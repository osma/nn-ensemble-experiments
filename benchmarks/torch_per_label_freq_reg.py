# STATUS: EXPERIMENTAL
# Purpose: torch_per_label with frequency-aware regularization (rare labels get stronger shrinkage).
from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label_freq_reg.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

from benchmarks.device import get_device
from benchmarks.metrics import (
    load_csr,
    ndcg_at_k_dense,
    f1_at_k_dense,
    update_markdown_scoreboard,
)

DEVICE = get_device()
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
K_VALUES = (10, 1000)

PATIENCE = 2
MIN_EPOCHS = 2

EVAL_BATCH_SIZE = 512
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337


def csr_to_dense_tensor(csr):
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


def _label_counts(y_true_csr: csr_matrix) -> np.ndarray:
    return np.asarray(y_true_csr.sum(axis=0)).reshape(-1).astype(np.float32)


def _freq_weight(counts: np.ndarray, mode: str) -> np.ndarray:
    """
    Per-label weights g(c) used to scale L2 penalties.

    mode:
      - "inv_sqrt": 1/sqrt(c+1)  (default; strong on rare labels)
      - "inv_log":  1/(log1p(c)+1)
      - "none":     all ones
    """
    if mode == "inv_sqrt":
        return 1.0 / np.sqrt(counts + 1.0)
    if mode == "inv_log":
        return 1.0 / (np.log1p(counts) + 1.0)
    if mode == "none":
        return np.ones_like(counts, dtype=np.float32)
    raise ValueError(f"Unknown freq mode: {mode!r}")


class PerLabelWeightedEnsemble(nn.Module):
    """
    Same architecture as torch_per_label: per-label linear ensemble with bias.

    score[l] = sum_m w[m,l] * x[m,l] + b[l]
    """

    def __init__(self, n_models: int, n_labels: int):
        super().__init__()
        self.n_models = n_models
        self.n_labels = n_labels
        self.weights = nn.Parameter(torch.full((n_models, n_labels), 1.0 / n_models))
        self.bias = nn.Parameter(torch.zeros(n_labels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input of shape (batch, n_models, n_labels), got {x.shape}"
            )
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected input with n_models={self.n_models}, "
                f"n_labels={self.n_labels}, got {x.shape}"
            )
        return (x * self.weights.unsqueeze(0)).sum(dim=1) + self.bias


def _sync_if_cuda() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class _Timer:
    def __init__(self):
        self.t0: float | None = None
        self.dt: float | None = None

    def __enter__(self):
        _sync_if_cuda()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _sync_if_cuda()
        assert self.t0 is not None
        self.dt = time.perf_counter() - self.t0


def _predict_in_batches(model: torch.nn.Module, x_cpu: torch.Tensor) -> torch.Tensor:
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
            logits = model(xb)
            outs.append(logits.detach().cpu())
    return torch.cat(outs, dim=0)


def _train_one(
    *,
    lambda_w: float,
    lambda_b: float,
    freq_mode: str,
    update_scoreboard: bool,
) -> dict[str, object]:
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr("data/train-output.npz")
    train_preds = [
        load_csr("data/train-bonsai.npz"),
        load_csr("data/train-fasttext.npz"),
        load_csr("data/train-mllm.npz"),
    ]

    X_train = torch.stack([csr_to_dense_tensor(p) for p in train_preds], dim=1)
    Y_train = csr_to_dense_tensor(y_train_true)

    # Frequency weights (CPU tensor, moved to DEVICE when used)
    counts = _label_counts(y_train_true)
    g = _freq_weight(counts, mode=freq_mode)  # (L,)
    g_t = torch.from_numpy(g).float()  # CPU

    # Fixed random subset of train rows for per-epoch early stopping metric
    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_train = X_train.shape[0]
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_train)
    train_eval_idx = rng.choice(n_train, size=n_eval, replace=False)
    X_train_eval = X_train[train_eval_idx]
    y_train_true_eval = y_train_true[train_eval_idx]

    print("Loading test data...")

    y_test_true = load_csr("data/test-output.npz")
    test_preds = [
        load_csr("data/test-bonsai.npz"),
        load_csr("data/test-fasttext.npz"),
        load_csr("data/test-mllm.npz"),
    ]
    X_test = torch.stack([csr_to_dense_tensor(p) for p in test_preds], dim=1)

    n_models = X_train.shape[1]
    n_labels = X_train.shape[2]

    model = PerLabelWeightedEnsemble(n_models=n_models, n_labels=n_labels).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.001,  # keep baseline behavior; freq-reg is additional
        eps=1e-8,
    )
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")
    print(
        f"Config | lambda_w={lambda_w:.6g} | lambda_b={lambda_b:.6g} | freq_mode={freq_mode}"
    )

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    best_metric = float("-inf")
    best_epoch = None
    best_state = None
    best_train_metrics = None
    best_test_metrics = None
    best_n_used_train = None
    best_n_used_test = None
    best_loss = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.perf_counter()

        model.train()
        with _Timer() as t_train_step:
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                logits = model(xb)

                bce = criterion(logits, yb)

                # Frequency-aware L2 penalties
                g_dev = g_t.to(DEVICE, non_blocking=True)

                w2_per_label = (model.weights * model.weights).sum(dim=0)  # (L,)
                b2_per_label = model.bias * model.bias  # (L,)

                reg_w = (g_dev * w2_per_label).mean()
                reg_b = (g_dev * b2_per_label).mean()

                loss = bce + (lambda_w * reg_w) + (lambda_b * reg_b)
                loss.backward()
                optimizer.step()

        # --- Train evaluation for early stopping (subset only) ---
        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)

        with _Timer() as t_ndcg_train:
            train_ndcg1000, n_used_train = ndcg_at_k_dense(
                y_train_true_eval, train_scores_eval, k=1000
            )

        # --- Test evaluation ---
        with _Timer() as t_pred_test:
            test_scores = _predict_in_batches(model, X_test)

        test_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg

        f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
        test_metrics["f1@5"] = f1

        epoch_dt = time.perf_counter() - epoch_t0

        def _dt(timer: _Timer) -> float:
            return float(timer.dt) if timer.dt is not None else 0.0

        print(
            f"Epoch {epoch:02d} timing | "
            f"train_step={_dt(t_train_step):.3f}s | "
            f"pred_train={_dt(t_pred_train):.3f}s | "
            f"ndcg_train@1000={train_ndcg1000:.6f} | "
            f"pred_test={_dt(t_pred_test):.3f}s | "
            f"loss={loss.item():.6f} | "
            f"total={epoch_dt:.3f}s"
        )

        current = train_ndcg1000
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_loss = float(loss.item())

            # Full train metrics only for best snapshot
            full_train_scores = _predict_in_batches(model, X_train)
            best_train_metrics = {}
            for k in K_VALUES:
                ndcg, n_used_train_full = ndcg_at_k_dense(
                    y_train_true, full_train_scores, k=k
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

    assert best_state is not None
    assert best_epoch is not None
    assert best_train_metrics is not None
    assert best_test_metrics is not None
    assert best_n_used_train is not None
    assert best_n_used_test is not None
    assert best_loss is not None

    model.load_state_dict(best_state)

    if update_scoreboard:
        update_markdown_scoreboard(
            path=scoreboard_path,
            model="torch_per_label_freq_reg",
            dataset="train",
            metrics=best_train_metrics,
            n_samples=best_n_used_train,
            epoch=best_epoch,
        )
        update_markdown_scoreboard(
            path=scoreboard_path,
            model="torch_per_label_freq_reg",
            dataset="test",
            metrics=best_test_metrics,
            n_samples=best_n_used_test,
            epoch=best_epoch,
        )

    print(
        "\nFinal test metrics | "
        f"ndcg@10={best_test_metrics['ndcg@10']:.6f} | "
        f"ndcg@1000={best_test_metrics['ndcg@1000']:.6f} | "
        f"f1@5={best_test_metrics['f1@5']:.6f} | "
        f"epoch={best_epoch} | "
        f"lambda_w={lambda_w:.6g} | lambda_b={lambda_b:.6g} | freq_mode={freq_mode}"
    )
    if update_scoreboard:
        print("\nSaved best result to SCOREBOARD.md")

    return {
        "lambda_w": float(lambda_w),
        "lambda_b": float(lambda_b),
        "freq_mode": str(freq_mode),
        "best_epoch": int(best_epoch),
        "best_sel_metric": float(best_metric),
        "best_loss": float(best_loss),
        "test_ndcg10": float(best_test_metrics["ndcg@10"]),
        "test_ndcg1000": float(best_test_metrics["ndcg@1000"]),
        "test_f1_5": float(best_test_metrics["f1@5"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a small grid over (lambda_w, lambda_b, freq_mode) and keep the best by train NDCG@1000.",
    )
    parser.add_argument("--lambda-w", type=float, default=0.0, help="L2 strength for weights")
    parser.add_argument("--lambda-b", type=float, default=0.0, help="L2 strength for bias")
    parser.add_argument(
        "--freq-mode",
        type=str,
        default="inv_sqrt",
        choices=["inv_sqrt", "inv_log", "none"],
        help="How to scale L2 by label frequency",
    )
    args = parser.parse_args()

    if not args.sweep:
        _train_one(
            lambda_w=float(args.lambda_w),
            lambda_b=float(args.lambda_b),
            freq_mode=str(args.freq_mode),
            update_scoreboard=True,
        )
        return

    # Small, conservative sweep grid. These are *additional* penalties on top of AdamW weight_decay.
    freq_modes = ["inv_sqrt", "inv_log"]
    lambda_w_grid = [0.0, 1e-5, 3e-5, 1e-4, 3e-4]
    lambda_b_grid = [0.0, 1e-6, 3e-6, 1e-5, 3e-5]

    results: list[dict[str, object]] = []
    best = None

    for fm in freq_modes:
        for lw in lambda_w_grid:
            for lb in lambda_b_grid:
                # Skip the fully-zero config for the second freq mode to reduce duplicate baseline runs.
                if fm != freq_modes[0] and lw == 0.0 and lb == 0.0:
                    continue

                print("\n" + "=" * 80)
                print(f"SWEEP RUN | freq_mode={fm} | lambda_w={lw} | lambda_b={lb}")
                print("=" * 80 + "\n")

                r = _train_one(
                    lambda_w=lw,
                    lambda_b=lb,
                    freq_mode=fm,
                    update_scoreboard=False,
                )
                results.append(r)

                if best is None or float(r["best_sel_metric"]) > float(best["best_sel_metric"]):
                    best = r

    assert best is not None

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY (sorted by selection metric desc)")
    print("=" * 80)
    results_sorted = sorted(results, key=lambda x: float(x["best_sel_metric"]), reverse=True)
    for r in results_sorted:
        print(
            f"sel={float(r['best_sel_metric']):.6f} | "
            f"epoch={int(r['best_epoch']):02d} | "
            f"freq_mode={r['freq_mode']} | "
            f"lambda_w={float(r['lambda_w']):.2e} | "
            f"lambda_b={float(r['lambda_b']):.2e} | "
            f"loss(best)={float(r['best_loss']):.6f} | "
            f"test ndcg@10={float(r['test_ndcg10']):.6f} | "
            f"test ndcg@1000={float(r['test_ndcg1000']):.6f} | "
            f"test f1@5={float(r['test_f1_5']):.6f}"
        )

    print("\nBest config by selection metric:")
    print(
        f"freq_mode={best['freq_mode']} | "
        f"lambda_w={float(best['lambda_w']):.6g} | "
        f"lambda_b={float(best['lambda_b']):.6g} | "
        f"epoch={int(best['best_epoch'])} | "
        f"sel={float(best['best_sel_metric']):.6f} | "
        f"loss(best)={float(best['best_loss']):.6f}"
    )

    print("\nRe-running best config to update SCOREBOARD.md...\n")
    _train_one(
        lambda_w=float(best["lambda_w"]),
        lambda_b=float(best["lambda_b"]),
        freq_mode=str(best["freq_mode"]),
        update_scoreboard=True,
    )


if __name__ == "__main__":
    main()
