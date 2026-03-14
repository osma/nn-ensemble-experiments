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
from benchmarks.preprocessing import csr_to_gamma_tensor, fit_source_gamma_from_csr
from benchmarks.datasets import get_dataset_config
from benchmarks.models.torch_3stage import Torch3Stage
from benchmarks.metrics import (
    load_csr,
    ndcg_at_k_dense,
    f1_at_k_dense,
    update_markdown_scoreboard,
)


def pairwise_logistic_ranking_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    *,
    n_pairs: int = 128,
    margin: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Pairwise logistic ranking loss with negative sampling.

    For each row:
      - Sample up to n_pairs positive labels and n_pairs negative labels.
      - Minimize: softplus(margin - (s_pos - s_neg))

    Notes:
    - Operates on *logits* (or any real-valued scores). No sigmoid needed.
    - Uses only sampled pairs for efficiency with huge label spaces.
    - Rows with no positives or no negatives contribute zero loss.
    """
    if logits.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: logits {tuple(logits.shape)} vs y_true {tuple(y_true.shape)}")
    if logits.ndim != 2:
        raise ValueError(f"Expected (B, L), got {tuple(logits.shape)}")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")

    device = logits.device
    B, L = logits.shape
    y = (y_true > 0).to(torch.bool)

    # Use a per-call generator so results can be reproducible if desired.
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    losses: list[torch.Tensor] = []

    for i in range(B):
        pos_idx = torch.nonzero(y[i], as_tuple=False).flatten()
        if pos_idx.numel() == 0:
            continue

        neg_idx = torch.nonzero(~y[i], as_tuple=False).flatten()
        if neg_idx.numel() == 0:
            continue

        m = int(min(n_pairs, pos_idx.numel(), neg_idx.numel()))
        if m <= 0:
            continue

        pos_sel = pos_idx[torch.randperm(pos_idx.numel(), generator=g, device=device)[:m]]
        neg_sel = neg_idx[torch.randperm(neg_idx.numel(), generator=g, device=device)[:m]]

        s_pos = logits[i, pos_sel]
        s_neg = logits[i, neg_sel]

        # softplus(margin - (s_pos - s_neg)) is a smooth version of max(0, margin - diff)
        diff = s_pos - s_neg
        losses.append(torch.nn.functional.softplus(float(margin) - diff).mean())

    if not losses:
        return logits.new_tensor(0.0)

    return torch.stack(losses).mean()


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
        w = model.global_w.detach().float().cpu().numpy()  # (M,)
        w_sum = float(w.sum())
        w_l1 = float(np.abs(w).sum())
        w_l2 = float(np.sqrt(np.square(w).sum()))

        delta_l2 = float(model.delta_l2().detach().float().cpu().item())
        bias_l2 = float(model.bias_l2().detach().float().cpu().item())

        print(
            f"{prefix} global_w={w.round(6).tolist()} "
            f"(sum={w_sum:.6f}, l1={w_l1:.6f}, l2={w_l2:.6f}, "
            f"min={float(w.min()):.6f}, max={float(w.max()):.6f}) "
            f"delta_l2={delta_l2:.6e} bias_l2={bias_l2:.6e}"
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

PAIRWISE_N_PAIRS = 128
PAIRWISE_MARGIN = 0.0
PAIRWISE_SEED = 1337

# Regularization (torch_mean_residual-style)
WEIGHT_DECAY = 0.0  # rely on explicit penalties
LAMBDA_GLOBAL_L2 = 1e-3
LAMBDA_DELTA_L2 = 1e-2
LAMBDA_BIAS_L2 = 1e-3


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

    # Per-source gamma correction fit on TRAIN prediction distributions (no labels).
    gammas = fit_source_gamma_from_csr(train_preds, quantile=0.95, target=0.5)
    print("Gamma per source:", {k: float(g) for k, g in zip(e3, gammas)})

    # Keep X_train on CPU; move only minibatches to GPU.
    # Apply gamma correction in probability space.
    X_train = torch.stack(
        [csr_to_gamma_tensor(p, gamma=float(g)) for p, g in zip(train_preds, gammas)],
        dim=1,
    )

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
    # Apply the SAME gammas fit from training distributions.
    X_test = torch.stack(
        [csr_to_gamma_tensor(p, gamma=float(g)) for p, g in zip(test_preds, gammas)],
        dim=1,
    )

    print(f"X_test: {_fmt_tensor_stats(X_test)}")
    print(f"y_test_true: shape={y_test_true.shape} nnz={int(y_test_true.nnz)}")

    cfg = get_dataset_config(dataset)
    init_global: torch.Tensor | None = None
    if cfg.ensemble3_init_weights is not None:
        init_global = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)
        if init_global.shape[0] != int(X_train.shape[1]):
            raise ValueError(
                f"ensemble3_init_weights has length {init_global.shape[0]}, but X_train has n_models={int(X_train.shape[1])}."
            )

    model = Torch3Stage(
        n_models=int(X_train.shape[1]),
        n_labels=int(X_train.shape[2]),
        init_global=init_global,
    ).to(DEVICE)
    _print_model_debug(model, prefix="init")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )
    criterion = None  # pairwise ranking loss (+ explicit regularization; see training loop)

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
            output_train = model(xb)  # scores
            loss_rank = pairwise_logistic_ranking_loss(
                output_train,
                yb,
                n_pairs=PAIRWISE_N_PAIRS,
                margin=PAIRWISE_MARGIN,
                seed=PAIRWISE_SEED + epoch,  # deterministic but changes per epoch
            )
            loss_reg = (
                float(LAMBDA_GLOBAL_L2) * model.global_l2()
                + float(LAMBDA_DELTA_L2) * model.delta_l2()
                + float(LAMBDA_BIAS_L2) * model.bias_l2()
            )
            loss = loss_rank + loss_reg
            loss.backward()

            # Print gradient stats for the trainable parameters.
            gw_grad = model.global_w.grad
            dw_grad = model.delta_w.grad
            b_grad = model.bias.grad

            grad_parts: list[str] = []
            if gw_grad is None:
                grad_parts.append("global_w_grad=None")
            else:
                grad_parts.append(f"global_w_grad({_fmt_tensor_stats(gw_grad)})")

            if dw_grad is None:
                grad_parts.append("delta_w_grad=None")
            else:
                # Avoid huge prints; just a couple of aggregates.
                dw = dw_grad.detach().float()
                grad_parts.append(
                    "delta_w_grad("
                    f"mean={float(dw.mean().cpu().item()):.6e} "
                    f"std={float(dw.std(unbiased=False).cpu().item()):.6e} "
                    f"min={float(dw.min().cpu().item()):.6e} "
                    f"max={float(dw.max().cpu().item()):.6e}"
                    ")"
                )

            if b_grad is None:
                grad_parts.append("bias_grad=None")
            else:
                bg = b_grad.detach().float()
                grad_parts.append(
                    "bias_grad("
                    f"mean={float(bg.mean().cpu().item()):.6e} "
                    f"std={float(bg.std(unbiased=False).cpu().item()):.6e} "
                    f"min={float(bg.min().cpu().item()):.6e} "
                    f"max={float(bg.max().cpu().item()):.6e}"
                    ")"
                )

            grad_stats = " ".join(grad_parts)

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

        # Print test metrics every epoch (requested; always on by default).
        print(
            "test | "
            f"epoch={epoch} "
            f"ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"f1@5={test_metrics['f1@5']:.6f} "
            f"used={n_used_test}"
        )

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
# Suggested commands to re-run:
# ./regenerate_scoreboard.sh --models torch_3stage
