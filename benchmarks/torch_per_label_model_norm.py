# STATUS: EXPERIMENTAL
# Purpose: torch_per_label with per-model normalization (train stats, nonzero-only).
from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label_model_norm.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.device import get_device
from benchmarks.metrics import (
    load_csr,
    ndcg_at_k_dense,
    f1_at_k_dense,
    update_markdown_scoreboard,
)

# ============================
# Model (same as torch_per_label)
# ============================


class PerLabelWeightedEnsemble(nn.Module):
    """
    Per-label weighted ensemble with bias.

    For each label l:
        score[l] = sum_m w[m, l] * x[m, l] + b[l]

    Input:
        x: (batch, M, L)
    Output:
        (batch, L) raw logits
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


# ============================
# Training / evaluation script
# ============================

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

EPS = 1e-6

# Robust scaling configuration
ROBUST_Q = 0.90  # use p90 of nonzero values per model (train only)


def csr_to_dense_log1p(csr) -> torch.Tensor:
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


def _compute_per_model_nonzero_robust_scale(X_train: torch.Tensor, q: float = ROBUST_Q) -> torch.Tensor:
    """
    Compute per-model robust scale over nonzero entries only.

    We use a high quantile (default p90) to represent the "typical high" score
    magnitude while being less sensitive than max and less fragile than std.

    Args:
        X_train: (N, M, L) CPU tensor
        q: quantile in (0,1], e.g. 0.9

    Returns:
        scale: (M,) CPU tensor (>= EPS)
    """
    if X_train.ndim != 3:
        raise ValueError(f"Expected X_train to have shape (N,M,L), got {X_train.shape}")
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0, 1]")

    n_models = X_train.shape[1]
    scale = torch.ones(n_models, dtype=torch.float32)

    for m in range(n_models):
        xm = X_train[:, m, :]  # (N, L)
        nz = xm[xm != 0.0]
        if nz.numel() == 0:
            scale[m] = 1.0
            continue

        # torch.quantile is available in modern torch; keep on CPU.
        s = torch.quantile(nz, q)

        # Guard against degenerate scales
        s_val = float(s.item())
        if not np.isfinite(s_val) or s_val <= EPS:
            scale[m] = 1.0
        else:
            scale[m] = s

    return scale


def _apply_per_model_scale_norm_inplace(X: torch.Tensor, scale: torch.Tensor) -> None:
    """
    Apply per-model scaling in-place to nonzero entries only:
        x <- x / scale[m]   for x != 0

    Keeps zeros as zeros to preserve sparsity semantics.

    Args:
        X:     (N, M, L) CPU tensor
        scale: (M,) CPU tensor
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X to have shape (N,M,L), got {X.shape}")
    if scale.ndim != 1:
        raise ValueError("scale must be a 1D tensor of shape (M,)")
    if X.shape[1] != scale.shape[0]:
        raise ValueError(f"Model dim mismatch: X has M={X.shape[1]}, scale={scale.shape}")

    for m in range(X.shape[1]):
        xm = X[:, m, :]  # view
        mask = xm != 0.0
        if mask.any():
            xm[mask] = xm[mask] / (scale[m] + EPS)


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


def main():
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr("data/train-output.npz")
    train_preds = [
        load_csr("data/train-bonsai.npz"),
        load_csr("data/train-fasttext.npz"),
        load_csr("data/train-mllm.npz"),
    ]

    # Build X_train on CPU (log1p)
    X_train = torch.stack([csr_to_dense_log1p(p) for p in train_preds], dim=1)

    # Y_train on CPU (labels are 0/1)
    Y_train = torch.from_numpy(y_train_true.toarray()).float()

    # Compute per-model robust scale from train only (nonzero entries)
    scale = _compute_per_model_nonzero_robust_scale(X_train, q=ROBUST_Q)
    print(
        f"Per-model nonzero robust scale (p{int(ROBUST_Q*100):d}) | "
        f"scale={scale.numpy().round(6).tolist()}"
    )

    # Apply scaling-only normalization to train in-place
    _apply_per_model_scale_norm_inplace(X_train, scale)

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

    X_test = torch.stack([csr_to_dense_log1p(p) for p in test_preds], dim=1)

    # Apply the same train-derived scaling to test in-place
    _apply_per_model_scale_norm_inplace(X_test, scale)

    n_models = X_train.shape[1]
    n_labels = X_train.shape[2]

    model = PerLabelWeightedEnsemble(n_models=n_models, n_labels=n_labels).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.001,
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

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.perf_counter()

        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # --- Train evaluation for early stopping (subset only) ---
        train_scores_eval = _predict_in_batches(model, X_train_eval)
        train_ndcg1000, _ = ndcg_at_k_dense(y_train_true_eval, train_scores_eval, k=1000)

        # --- Test evaluation (batched; no CSR conversion) ---
        test_scores = _predict_in_batches(model, X_test)
        test_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg

        f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
        test_metrics["f1@5"] = f1

        epoch_dt = time.perf_counter() - epoch_t0
        print(
            f"Epoch {epoch:02d} | "
            f"loss={loss.item():.6f} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} | "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"dt={epoch_dt:.3f}s"
        )

        current = train_ndcg1000
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Full train metrics only for the best epoch snapshot
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

    model.load_state_dict(best_state)

    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_per_label_model_norm",
        dataset="train",
        metrics=best_train_metrics,
        n_samples=best_n_used_train,
        epoch=best_epoch,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_per_label_model_norm",
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
        f"epoch={best_epoch}"
    )
    print("\nSaved best result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
