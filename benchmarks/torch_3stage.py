from pathlib import Path
import sys

# Allow running as a script: `uv run benchmarks/torch_3stage.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.datasets import ensemble3_keys, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.preprocessing import csr_to_logit_tensor
from benchmarks.models.torch_3stage import Torch3Stage
from benchmarks.metrics import (
    load_csr,
    ndcg_at_k_dense,
    f1_at_k_dense,
    update_markdown_scoreboard,
)


def _fmt_tensor_stats(t: torch.Tensor) -> str:
    tt = t.detach().float().cpu()
    return (
        f"shape={tuple(tt.shape)} "
        f"min={tt.min().item():.6f} "
        f"max={tt.max().item():.6f} "
        f"mean={tt.mean().item():.6f} "
        f"std={tt.std(unbiased=False).item():.6f}"
    )


def _print_model_debug(model: Torch3Stage, *, prefix: str) -> None:
    with torch.no_grad():
        w = model.effective_w().detach().float().cpu()
        w_np = w.numpy()

        w_sum = float(w_np.sum())
        w_l1 = float(np.abs(w_np).sum())
        w_l2 = float(np.sqrt(np.square(w_np).sum()))

        b = float(model.bias.detach().float().cpu().item())

        print(
            f"{prefix} weights={w_np.round(6).tolist()} "
            f"(sum={w_sum:.6f}, l1={w_l1:.6f}, l2={w_l2:.6f}, "
            f"min={float(w_np.min()):.6f}, max={float(w_np.max()):.6f}) "
            f"bias={b:.6f}"
        )


DEVICE = get_device()
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 32
K_VALUES = (10, 1000)

PATIENCE = 2
MIN_EPOCHS = 2

EVAL_BATCH_SIZE = 512
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337


def _predict_in_batches(model: torch.nn.Module, x_cpu: torch.Tensor) -> torch.Tensor:
    """
    Run model forward pass over a CPU tensor in batches, moving only minibatches
    to DEVICE. Returns a CPU tensor of outputs.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_cpu),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
    )

    outs: list[torch.Tensor] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            out = model(xb)
            outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="yso-fi",
        choices=["yso-fi", "yso-en", "koko"],
        help="Dataset to benchmark",
    )
    args = parser.parse_args()
    dataset = str(args.dataset)

    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    e3 = ensemble3_keys(dataset)

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in e3]

    # Keep X_train on CPU; move only minibatches to GPU.
    # Convert base probabilities in [0,1] to logits for logit-space training.
    X_train = torch.stack([csr_to_logit_tensor(p, eps=1e-6) for p in train_preds], dim=1)

    # Keep Y_train on CPU (requested).
    Y_train = torch.from_numpy(y_train_true.toarray()).float()

    print(f"X_train: {_fmt_tensor_stats(X_train)}")
    print(
        f"Y_train: shape={tuple(Y_train.shape)} "
        f"pos_rate={float(Y_train.mean().item()):.6f} "
        f"nnz={int(y_train_true.nnz)}"
    )

    # Fixed random subset of train rows for per-epoch early stopping metric
    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_train = X_train.shape[0]
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_train)
    train_eval_idx = rng.choice(n_train, size=n_eval, replace=False)
    X_train_eval = X_train[train_eval_idx]
    y_train_true_eval = y_train_true[train_eval_idx]

    print("Loading test data...")

    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in e3]

    # Keep X_test on CPU; move to GPU only for evaluation forward pass.
    # Convert base probabilities in [0,1] to logits for logit-space inference.
    X_test = torch.stack([csr_to_logit_tensor(p, eps=1e-6) for p in test_preds], dim=1)

    print(f"X_test: {_fmt_tensor_stats(X_test)}")
    print(f"y_test_true: shape={y_test_true.shape} nnz={int(y_test_true.nnz)}")

    model = Torch3Stage(n_models=X_train.shape[1]).to(DEVICE)
    _print_model_debug(model, prefix="init")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
        eps=1e-8,
    )
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Early stopping: select best epoch by TRAIN NDCG@1000 (no test leakage)
    best_metric = float("-inf")
    best_epoch = None
    best_state = None
    best_train_metrics = None
    best_test_metrics = None
    best_n_used_train = None
    best_n_used_test = None
    epochs_no_improve = 0

    if EARLY_STOP_EVAL_ROWS <= 0:
        raise ValueError("EARLY_STOP_EVAL_ROWS must be positive")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_n = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output_train = model(xb)  # logits
            loss = criterion(output_train, yb)
            loss.backward()

            # Print gradient stats for the trainable mixture logits (alpha).
            alpha_grad = model.alpha.grad
            if alpha_grad is None:
                grad_stats = "grad=None"
            else:
                grad_stats = f"grad({_fmt_tensor_stats(alpha_grad)})"

            optimizer.step()

            epoch_loss_sum += float(loss.detach().item())
            epoch_loss_n += 1

        avg_loss = epoch_loss_sum / max(1, epoch_loss_n)

        # --- Train evaluation for early stopping (subset only) ---
        train_eval_output = _predict_in_batches(model, X_train_eval)
        train_ndcg1000, n_used_train = ndcg_at_k_dense(
            y_train_true_eval, train_eval_output, k=1000
        )

        # --- Test evaluation (batched; no CSR conversion) ---
        output_test = _predict_in_batches(model, X_test)
        test_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k_dense(y_test_true, output_test, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg

        f1, _ = f1_at_k_dense(y_test_true, output_test, k=5)
        test_metrics["f1@5"] = f1

        _print_model_debug(
            model,
            prefix=(
                f"epoch={epoch} loss={avg_loss:.6f} "
                f"train_ndcg@1000(subset)={train_ndcg1000:.6f} "
                f"train_used={n_used_train} {grad_stats} |"
            ),
        )

        current = train_ndcg1000
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Compute full train metrics only for the best epoch snapshot
            full_train_output = _predict_in_batches(model, X_train)
            best_train_metrics = {}
            for k in K_VALUES:
                ndcg, n_used_train_full = ndcg_at_k_dense(
                    y_train_true, full_train_output, k=k
                )
                best_train_metrics[f"ndcg@{k}"] = ndcg
            best_n_used_train = n_used_train_full

            best_test_metrics = test_metrics.copy()
            best_n_used_test = n_used_test
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch >= MIN_EPOCHS and epochs_no_improve >= PATIENCE:
            break

    if best_state is None or best_epoch is None:
        raise RuntimeError("Training failed to produce a best_state/best_epoch")

    model.load_state_dict(best_state)

    update_markdown_scoreboard(
        path=scoreboard_path,
        model=f"torch_3stage({','.join(e3)})",
        dataset=dataset,
        split="train",
        metrics=best_train_metrics,
        n_samples=best_n_used_train,
        epoch=best_epoch,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model=f"torch_3stage({','.join(e3)})",
        dataset=dataset,
        split="test",
        metrics=best_test_metrics,
        n_samples=best_n_used_test,
        epoch=best_epoch,
    )

    _print_model_debug(model, prefix="final")

    print(
        "\nFinal test metrics | "
        f"ndcg@10={best_test_metrics['ndcg@10']:.6f} | "
        f"ndcg@1000={best_test_metrics['ndcg@1000']:.6f} | "
        f"f1@5={best_test_metrics['f1@5']:.6f} | "
        f"epoch={best_epoch}"
    )
    print("\nSaved best result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
