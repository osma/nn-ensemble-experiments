# STATUS: EXPERIMENTAL
# Purpose: Per-label ensemble where per-model trust is adjusted using label frequency features.
from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label_freq_gate.py`
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

# Regularization strength for the frequency-based residual weight correction.
# Higher -> pushes the model closer to plain torch_per_label behavior.
LAMBDA_DELTA = 1e-3


def csr_to_dense_tensor(csr):
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


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


def _label_freq_features(y_train_true_csr) -> torch.Tensor:
    """
    Build per-label frequency features from y_train_true (CSR).

    Returns:
        Tensor of shape (L, F) on CPU.
    """
    # counts per label (column sums)
    counts = np.asarray(y_train_true_csr.sum(axis=0)).reshape(-1).astype(np.float32)  # (L,)

    logf = np.log1p(counts)
    inv_logf = 1.0 / (1.0 + logf)
    is_zero = (counts == 0.0).astype(np.float32)

    # Rare-label indicator (helps the model learn a distinct regime vs just "small logf")
    is_rare_1 = (counts <= 1.0).astype(np.float32)
    is_rare_5 = (counts <= 5.0).astype(np.float32)

    feats = np.stack([logf, inv_logf, is_zero, is_rare_1, is_rare_5], axis=1)  # (L, 5)

    # Normalize logf for stability; keep indicator features as-is.
    if feats.shape[0] > 0:
        mu = feats[:, 0].mean()
        sd = feats[:, 0].std() + 1e-6
        feats[:, 0] = (feats[:, 0] - mu) / sd

    return torch.from_numpy(feats).float()  # CPU


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


class PerLabelFreqGatedEnsemble(nn.Module):
    """
    Per-label weighted ensemble with bias, where per-model trust is adjusted by
    label-frequency features.

    This version uses a *residual* correction to the per-label weights:

        w_eff[m,l] = base_w[m,l] + alpha * delta_w[m,l](freq[l])

    This is intentionally more stable than multiplicative gating, because it
    preserves the strong per-label baseline and lets frequency add a small,
    regularized adjustment.

    Input:
        x: (batch, M, L) log1p-scaled base predictions
    Output:
        (batch, L) raw logits
    """

    def __init__(
        self,
        n_models: int,
        n_labels: int,
        label_feats: torch.Tensor,
        hidden: int = 16,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        if label_feats.ndim != 2:
            raise ValueError(f"label_feats must be 2D (L,F), got {label_feats.shape}")
        if label_feats.shape[0] != n_labels:
            raise ValueError(
                f"label_feats has {label_feats.shape[0]} labels, expected {n_labels}"
            )

        self.n_models = n_models
        self.n_labels = n_labels

        # Base per-model, per-label weights (like torch_per_label)
        self.base_w = nn.Parameter(torch.full((n_models, n_labels), 1.0 / n_models))

        # Per-label bias
        self.bias = nn.Parameter(torch.zeros(n_labels))

        # Non-trainable label features (moves with .to(device))
        self.register_buffer("label_feats", label_feats)  # (L, F)
        fdim = int(label_feats.shape[1])

        # Per-label residual generator: (F) -> (M)
        self.delta = nn.Sequential(
            nn.Linear(fdim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_models),
        )

        # Small scaling for the residual; learnable but initialized small.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def delta_w(self) -> torch.Tensor:
        """
        Returns:
            delta_w: (M, L) residual weight correction derived from label_feats.
        """
        # (L, M) -> (M, L)
        dw = self.delta(self.label_feats).transpose(0, 1).contiguous()
        return dw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, n_models, n_labels), got {x.shape}")
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected n_models={self.n_models}, n_labels={self.n_labels}, got {x.shape}"
            )

        dw = self.delta_w()  # (M, L)
        w_eff = self.base_w + self.alpha * dw  # (M, L)

        out = (x * w_eff.unsqueeze(0)).sum(dim=1) + self.bias
        return out  # raw logits


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

    # Keep X_train on CPU; move only minibatches to GPU.
    X_train = torch.stack([csr_to_dense_tensor(p) for p in train_preds], dim=1)

    # Keep Y_train on CPU (requested).
    Y_train = csr_to_dense_tensor(y_train_true)

    # Label frequency features (CPU)
    label_feats = _label_freq_features(y_train_true)

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

    # Keep X_test on CPU; move to GPU only for evaluation forward pass.
    X_test = torch.stack([csr_to_dense_tensor(p) for p in test_preds], dim=1)

    n_models = X_train.shape[1]
    n_labels = X_train.shape[2]

    model = PerLabelFreqGatedEnsemble(
        n_models=n_models,
        n_labels=n_labels,
        label_feats=label_feats,
        hidden=16,
        alpha_init=0.1,
    ).to(DEVICE)

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
        with _Timer() as t_train_step:
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                logits = model(xb)

                # Regularize the frequency-based residual so it stays small unless useful.
                dw = model.delta_w()
                reg = (dw * dw).mean()

                loss = criterion(logits, yb) + (LAMBDA_DELTA * reg)
                loss.backward()
                optimizer.step()

        # --- Train evaluation for early stopping (subset only) ---
        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)

        with _Timer() as t_ndcg_train:
            train_ndcg1000, _ = ndcg_at_k_dense(
                y_train_true_eval, train_scores_eval, k=1000
            )

        # --- Test evaluation (batched; no CSR conversion) ---
        with _Timer() as t_pred_test:
            test_scores = _predict_in_batches(model, X_test)

        test_metrics = {}
        t_ndcg_test: dict[int, float] = {}
        for k in K_VALUES:
            with _Timer() as t:
                ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
            assert t.dt is not None
            t_ndcg_test[k] = t.dt
            test_metrics[f"ndcg@{k}"] = ndcg

        with _Timer() as t_f1:
            f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
        test_metrics["f1@5"] = f1

        epoch_dt = time.perf_counter() - epoch_t0

        def _dt(timer: _Timer) -> float:
            return float(timer.dt) if timer.dt is not None else 0.0

        print(
            f"Epoch {epoch:02d} timing | "
            f"train_step={_dt(t_train_step):.3f}s | "
            f"pred_train={_dt(t_pred_train):.3f}s | "
            f"ndcg_train@1000={_dt(t_ndcg_train):.3f}s | "
            f"pred_test={_dt(t_pred_test):.3f}s | "
            f"ndcg_test@10={t_ndcg_test.get(10, 0.0):.3f}s ndcg_test@1000={t_ndcg_test.get(1000, 0.0):.3f}s | "
            f"f1@5={_dt(t_f1):.3f}s | "
            f"loss={loss.item():.6f} | "
            f"alpha={float(model.alpha.detach().cpu()):.4f} | "
            f"total={epoch_dt:.3f}s"
        )

        current = train_ndcg1000
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Compute full train metrics only for the best epoch snapshot
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
    model.load_state_dict(best_state)

    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_per_label_freq_gate",
        dataset="train",
        metrics=best_train_metrics,
        n_samples=best_n_used_train,
        epoch=best_epoch,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_per_label_freq_gate",
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
