# STATUS: ACTIVE (variant)
# Purpose: torch_per_label variant that reparameterizes bias as global + per-label delta.
from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label_bias_global_plus_delta.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

from benchmarks.datasets import ensemble3_keys, get_dataset_config, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.metrics import (
    load_csr,
    evaluate_model_batched,
    update_markdown_scoreboard,
)
from benchmarks.preprocessing import SparseCSRDataset, log1p_transform


class PerLabelWeightedEnsembleBiasGlobalPlusDelta(nn.Module):
    """
    Per-label weighted ensemble with bias reparameterized as:
        bias[l] = bias_global + bias_delta[l]

    For each label l:
        score[l] = sum_m w[m, l] * x[m, l] + bias_global + bias_delta[l]

    Notes:
    - Inputs are log1p-preprocessed during preprocessing (see csr_to_dense_tensor).
    - Returns raw logits (no clamp, no sigmoid).
    - Intended for use with BCEWithLogitsLoss or ranking-aware losses.
    """

    def __init__(
        self,
        n_models: int,
        n_labels: int,
        init_model_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_models = n_models
        self.n_labels = n_labels

        if init_model_weights is None:
            init = torch.full((n_models,), 1.0 / n_models, dtype=torch.float32)
        else:
            if init_model_weights.ndim != 1 or init_model_weights.shape[0] != n_models:
                raise ValueError(
                    f"init_model_weights must have shape ({n_models},), got {tuple(init_model_weights.shape)}"
                )
            init = init_model_weights.to(dtype=torch.float32).clone()
            s = float(init.sum().item())
            if not np.isfinite(s) or s <= 0.0:
                raise ValueError("init_model_weights must sum to a positive finite value")
            init = init / init.sum()

        # Per-model, per-label weights (initialize each label with the same per-model weights)
        self.weights = nn.Parameter(init[:, None].repeat(1, n_labels))

        # Bias reparameterization
        self.bias_global = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.bias_delta = nn.Parameter(torch.zeros(n_labels, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, n_models, n_labels)
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input of shape (batch, n_models, n_labels), got {x.shape}"
            )
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected input with n_models={self.n_models}, "
                f"n_labels={self.n_labels}, got {x.shape}"
            )

        weighted = x * self.weights.unsqueeze(0)
        out = weighted.sum(dim=1) + self.bias_global + self.bias_delta
        return out


# ============================
# Training / evaluation script
# ============================

DEVICE = get_device()
EPOCHS = 20
K_VALUES = (10, 1000)

PATIENCE = 2
MIN_EPOCHS = 2

EVAL_BATCH_SIZE = 512
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

# Best hyperparameters (from tuning)
BEST_LR = 0.003
BEST_WEIGHT_DECAY = 0.0
BEST_BATCH_SIZE = 256

# Reproducibility for training shuffles / init
TRAIN_SEED = 0

# Dataset-specific initialization is defined in benchmarks.datasets.DatasetConfig
# as `ensemble3_init_weights` and consumed below.


# (Removed csr_to_dense_tensor, _Timer, and _predict_in_batches in favor of SparseCSRDataset and evaluate_model_batched)


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())


# (Removed _Timer and _predict_in_batches in favor of evaluate_model_batched)


def train_and_evaluate(
    *,
    dataset: str,
    ensemble_keys: tuple[str, str, str],
    lr: float,
    weight_decay: float,
    batch_size: int,
    train_loader: torch.utils.data.DataLoader,
    y_train_true: csr_matrix,
    train_eval_loader: torch.utils.data.DataLoader,
    y_train_true_eval: csr_matrix,
    test_loader: torch.utils.data.DataLoader,
    y_test_true: csr_matrix,
    full_train_loader: torch.utils.data.DataLoader,
) -> dict[str, object]:
    """
    Train a model with given hyperparameters and return the best snapshot
    selected by TRAIN subset NDCG@1000 (early stopping metric).

    Returns dict with:
      - best_metric (float): best train subset NDCG@1000
      - best_epoch (int)
      - best_train_metrics (dict[str,float]) computed on full train at best epoch
      - best_test_metrics (dict[str,float]) computed on test at best epoch
      - best_n_used_train (int)
      - best_n_used_test (int)
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    # Make each run deterministic-ish (init + dataloader shuffle)
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    n_models = len(ensemble_keys)
    n_labels = y_train_true.shape[1]

    init_weights: torch.Tensor | None = None
    cfg = get_dataset_config(dataset)
    if cfg.ensemble3 != ensemble_keys:
        raise ValueError(
            "Internal error: ensemble_keys does not match dataset config "
            f"(cfg.ensemble3={cfg.ensemble3}, ensemble_keys={ensemble_keys})"
        )
    if cfg.ensemble3_init_weights is not None:
        init_weights = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)
        if init_weights.shape[0] != n_models:
            raise ValueError(
                f"ensemble3_init_weights has length {init_weights.shape[0]}, but ensemble has n_models={n_models}."
            )

    model = PerLabelWeightedEnsembleBiasGlobalPlusDelta(
        n_models=n_models,
        n_labels=n_labels,
        init_model_weights=init_weights,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_metric = float("-inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    best_n_used_train: int | None = None
    best_n_used_test: int | None = None
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
        train_res_eval = evaluate_model_batched(model, train_eval_loader, y_train_true_eval, k_values=(1000,), device=DEVICE)
        train_ndcg1000 = train_res_eval["ndcg@1000"]

        # --- Test evaluation ---
        test_metrics = evaluate_model_batched(model, test_loader, y_test_true, k_values=K_VALUES, f1_k=5, device=DEVICE)

        epoch_dt = time.perf_counter() - epoch_t0

        print(
            f"[lr={lr:g} wd={weight_decay:g} bs={batch_size}] "
            f"Epoch {epoch:02d} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} | "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"total={epoch_dt:.3f}s"
        )

        current = train_ndcg1000
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Compute full train metrics only for the best epoch snapshot
            best_train_metrics_res = evaluate_model_batched(model, full_train_loader, y_train_true, k_values=K_VALUES, device=DEVICE)
            best_train_metrics = {k: v for k, v in best_train_metrics_res.items() if k.startswith("ndcg")}
            best_n_used_train = int(best_train_metrics_res["n_used"])

            best_test_metrics = test_metrics.copy()
            best_n_used_test = int(test_metrics["n_used"])

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

    # Load best snapshot before returning (useful if caller wants to reuse model later)
    model.load_state_dict(best_state)

    return {
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "best_train_metrics": best_train_metrics,
        "best_test_metrics": best_test_metrics,
        "best_n_used_train": int(best_n_used_train),
        "best_n_used_test": int(best_n_used_test),
    }


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

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_per_label_bias_global_plus_delta({','.join(ensemble_keys)})"

    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]

    n_samples_train = y_train_true.shape[0]

    # Datasets using SparseCSRDataset
    train_ds = SparseCSRDataset(train_preds, y_train_true, stack_dim=0, transform=lambda xy: (log1p_transform(xy[0]), xy[1]))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BEST_BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type == "cuda"))

    full_train_ds = SparseCSRDataset(train_preds, stack_dim=0, transform=log1p_transform)
    full_train_loader = torch.utils.data.DataLoader(full_train_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # Fixed random subset of train rows for per-epoch early stopping metric
    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_samples_train)
    train_eval_idx = rng.choice(n_samples_train, size=n_eval, replace=False)
    train_eval_preds = [p[train_eval_idx] for p in train_preds]
    y_train_true_eval = y_train_true[train_eval_idx]
    train_eval_ds = SparseCSRDataset(train_eval_preds, stack_dim=0, transform=log1p_transform)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    print("Loading test data...")
    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in ensemble_keys]
    test_ds = SparseCSRDataset(test_preds, stack_dim=0, transform=log1p_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    print(
        "Training with best hyperparameters | "
        f"lr={BEST_LR:g} | wd={BEST_WEIGHT_DECAY:g} | bs={BEST_BATCH_SIZE}"
    )

    result = train_and_evaluate(
        dataset=dataset,
        ensemble_keys=ensemble_keys,
        lr=BEST_LR,
        weight_decay=BEST_WEIGHT_DECAY,
        batch_size=BEST_BATCH_SIZE,
        train_loader=train_loader,
        y_train_true=y_train_true,
        train_eval_loader=train_eval_loader,
        y_train_true_eval=y_train_true_eval,
        test_loader=test_loader,
        y_test_true=y_test_true,
        full_train_loader=full_train_loader,
    )

    best_epoch = int(result["best_epoch"])
    best_metric = float(result["best_metric"])
    best_train_metrics = result["best_train_metrics"]
    best_test_metrics = result["best_test_metrics"]
    best_n_used_train = int(result["best_n_used_train"])
    best_n_used_test = int(result["best_n_used_test"])

    print("\n====================")
    print("Training complete")
    print("====================")
    print(
        "Best hyperparameters | "
        f"lr={BEST_LR:g} | wd={BEST_WEIGHT_DECAY:g} | bs={BEST_BATCH_SIZE} | "
        f"best_epoch={best_epoch} | "
        f"train_ndcg@1000(subset)={best_metric:.6f}"
    )
    print(
        "Best test metrics | "
        f"ndcg@10={float(best_test_metrics['ndcg@10']):.6f} | "
        f"ndcg@1000={float(best_test_metrics['ndcg@1000']):.6f} | "
        f"f1@5={float(best_test_metrics['f1@5']):.6f}"
    )

    # Update scoreboard with the best result
    update_markdown_scoreboard(
        path=scoreboard_path,
        model=model_name,
        dataset=dataset,
        split="train",
        metrics=best_train_metrics,
        n_samples=best_n_used_train,
        epoch=best_epoch,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model=model_name,
        dataset=dataset,
        split="test",
        metrics=best_test_metrics,
        n_samples=best_n_used_test,
        epoch=best_epoch,
    )

    print("\nSaved result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
