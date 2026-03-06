# STATUS: ACTIVE (recommended base model)
# Purpose: Best-performing per-label linear ensemble trained with BCE on raw logits.
from pathlib import Path
import json
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label.py`
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
    ndcg_at_k_dense,
    f1_at_k_dense,
    update_markdown_scoreboard,
)


class PerLabelWeightedEnsemble(nn.Module):
    """
    Per-label weighted ensemble with bias.

    For each label l:
        score[l] = sum_m w[m, l] * x[m, l] + b[l]

    Notes:
    - Applies a fixed sub-linear log1p transform to inputs.

    Input:
        x: (batch, M, L)
            M = number of base models
            L = number of labels

    Output:
        (batch, L) raw logits

    Notes:
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

        # Per-label bias
        self.bias = nn.Parameter(torch.zeros(n_labels))

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

        # Inputs are already log1p-scaled during preprocessing
        x_scaled = x

        weighted = x_scaled * self.weights.unsqueeze(0)
        out = weighted.sum(dim=1) + self.bias
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


def csr_to_dense_tensor(csr):
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())


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


def train_and_evaluate(
    *,
    dataset: str,
    ensemble_keys: tuple[str, str, str],
    lr: float,
    weight_decay: float,
    batch_size: int,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    y_train_true: csr_matrix,
    X_train_eval: torch.Tensor,
    y_train_true_eval: csr_matrix,
    X_test: torch.Tensor,
    y_test_true: csr_matrix,
) -> dict[str, object]:
    """
    Train a model with given hyperparameters and return the best snapshot
    selected by TRAIN subset NDCG@1000 (early stopping metric).

    Also returns a diagnostics payload (JSON-serializable) for the best epoch
    to help understand cross-dataset behavior.

    Returns dict with:
      - best_metric (float): best train subset NDCG@1000
      - best_epoch (int)
      - best_train_metrics (dict[str,float]) computed on full train at best epoch
      - best_test_metrics (dict[str,float]) computed on test at best epoch
      - best_n_used_train (int)
      - best_n_used_test (int)
      - timing (dict[str,float]) rough totals
      - diagnostics (dict[str,object]) best-epoch diagnostics
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    # Make each run deterministic-ish (init + dataloader shuffle)
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    n_models = X_train.shape[1]
    n_labels = X_train.shape[2]

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
                f"ensemble3_init_weights has length {init_weights.shape[0]}, but X_train has n_models={n_models}."
            )

    model = PerLabelWeightedEnsemble(
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

    # Unweighted BCE performs best for NDCG in this setup
    criterion = nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    best_metric = float("-inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    best_n_used_train: int | None = None
    best_n_used_test: int | None = None
    best_diag: dict[str, object] | None = None
    epochs_no_improve = 0

    t_total_train_step = 0.0
    t_total_pred_train = 0.0
    t_total_pred_test = 0.0
    t_total_metrics = 0.0

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.perf_counter()

        model.train()
        with _Timer() as t_train_step:
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        t_total_train_step += float(t_train_step.dt or 0.0)

        # --- Train evaluation for early stopping (subset only) ---
        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)
        t_total_pred_train += float(t_pred_train.dt or 0.0)

        with _Timer() as t_metric_train:
            train_ndcg1000, _n_used_train_eval = ndcg_at_k_dense(
                y_train_true_eval, train_scores_eval, k=1000
            )
        t_total_metrics += float(t_metric_train.dt or 0.0)

        # --- Test evaluation (batched; no CSR conversion) ---
        with _Timer() as t_pred_test:
            test_scores = _predict_in_batches(model, X_test)
        t_total_pred_test += float(t_pred_test.dt or 0.0)

        test_metrics: dict[str, float] = {}
        n_used_test: int | None = None
        with _Timer() as t_metric_test:
            for k in K_VALUES:
                ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
                test_metrics[f"ndcg@{k}"] = ndcg

            f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
            test_metrics["f1@5"] = f1
        t_total_metrics += float(t_metric_test.dt or 0.0)

        epoch_dt = time.perf_counter() - epoch_t0

        def _dt(timer: _Timer) -> float:
            return float(timer.dt) if timer.dt is not None else 0.0

        print(
            f"[lr={lr:g} wd={weight_decay:g} bs={batch_size}] "
            f"Epoch {epoch:02d} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} | "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"timing train_step={_dt(t_train_step):.3f}s pred_train={_dt(t_pred_train):.3f}s "
            f"pred_test={_dt(t_pred_test):.3f}s metrics={_dt(t_metric_train)+_dt(t_metric_test):.3f}s "
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
            n_used_train_full: int | None = None
            for k in K_VALUES:
                ndcg, n_used_train_full = ndcg_at_k_dense(
                    y_train_true, full_train_scores, k=k
                )
                best_train_metrics[f"ndcg@{k}"] = ndcg
            best_n_used_train = int(n_used_train_full or 0)

            best_test_metrics = test_metrics.copy()
            best_n_used_test = int(n_used_test or 0)

            # ---- Best-epoch diagnostics (weights + bias distributions) ----
            with torch.no_grad():
                w = model.weights.detach().cpu()  # (M, L)
                b = model.bias.detach().cpu()  # (L,)

            def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
                t64 = t.to(dtype=torch.float64)
                flat = t64.reshape(-1)
                if flat.numel() == 0:
                    return {"n": 0.0}
                q = torch.quantile(flat, torch.tensor([0.0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.0], dtype=torch.float64))
                return {
                    "n": float(flat.numel()),
                    "mean": float(flat.mean().item()),
                    "std": float(flat.std(unbiased=False).item()),
                    "min": float(q[0].item()),
                    "p01": float(q[1].item()),
                    "p05": float(q[2].item()),
                    "p50": float(q[3].item()),
                    "p95": float(q[4].item()),
                    "p99": float(q[5].item()),
                    "max": float(q[6].item()),
                }

            # Per-model weight means over labels (M,)
            w_mean_per_model = w.mean(dim=1).numpy().tolist()
            w_abs_mean_per_model = w.abs().mean(dim=1).numpy().tolist()

            # Per-label "dominant model" frequency
            dominant = torch.argmax(w, dim=0)  # (L,)
            dominant_counts = torch.bincount(dominant, minlength=w.shape[0]).to(dtype=torch.int64)
            dominant_frac = (dominant_counts.to(dtype=torch.float64) / float(w.shape[1])).numpy().tolist()

            best_diag = {
                "dataset": dataset,
                "ensemble_keys": list(ensemble_keys),
                "selected_by": "train_subset_ndcg@1000",
                "best_epoch": int(epoch),
                "best_train_subset_ndcg@1000": float(train_ndcg1000),
                "hyperparams": {"lr": float(lr), "weight_decay": float(weight_decay), "batch_size": int(batch_size)},
                "shapes": {"n_models": int(w.shape[0]), "n_labels": int(w.shape[1])},
                "weights": {
                    "overall": _tensor_stats(w),
                    "per_model_mean_over_labels": w_mean_per_model,
                    "per_model_mean_abs_over_labels": w_abs_mean_per_model,
                    "dominant_model_frac_over_labels": dominant_frac,
                    "n_negative": int((w < 0).sum().item()),
                },
                "bias": {
                    "overall": _tensor_stats(b),
                },
                "metrics_at_best_epoch": {
                    "train_subset": {"ndcg@1000": float(train_ndcg1000)},
                    "train_full": {k: float(v) for k, v in best_train_metrics.items()},
                    "test": {k: float(v) for k, v in best_test_metrics.items()},
                },
            }

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
    assert best_diag is not None

    # Load best snapshot before returning (useful if caller wants to reuse model later)
    model.load_state_dict(best_state)

    return {
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "best_train_metrics": best_train_metrics,
        "best_test_metrics": best_test_metrics,
        "best_n_used_train": int(best_n_used_train),
        "best_n_used_test": int(best_n_used_test),
        "timing": {
            "train_step_s": float(t_total_train_step),
            "pred_train_s": float(t_total_pred_train),
            "pred_test_s": float(t_total_pred_test),
            "metrics_s": float(t_total_metrics),
        },
        "diagnostics": best_diag,
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
    model_name = f"torch_per_label({','.join(ensemble_keys)})"

    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]

    # Keep X_train on CPU; move only minibatches to GPU.
    X_train = torch.stack([csr_to_dense_tensor(p) for p in train_preds], dim=1)

    # Keep Y_train on CPU (requested).
    Y_train = csr_to_dense_tensor(y_train_true)

    # Fixed random subset of train rows for per-epoch early stopping metric
    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_train = X_train.shape[0]
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_train)
    train_eval_idx = rng.choice(n_train, size=n_eval, replace=False)
    X_train_eval = X_train[train_eval_idx]
    y_train_true_eval = y_train_true[train_eval_idx]

    print("Loading test data...")

    # Keep y_test_true / Y_test on CPU (requested).
    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in ensemble_keys]

    # Keep X_test on CPU; move to GPU only for evaluation forward pass.
    X_test = torch.stack([csr_to_dense_tensor(p) for p in test_preds], dim=1)

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
        X_train=X_train,
        Y_train=Y_train,
        y_train_true=y_train_true,
        X_train_eval=X_train_eval,
        y_train_true_eval=y_train_true_eval,
        X_test=X_test,
        y_test_true=y_test_true,
    )

    best_epoch = int(result["best_epoch"])
    best_metric = float(result["best_metric"])
    best_train_metrics = result["best_train_metrics"]
    best_test_metrics = result["best_test_metrics"]
    best_n_used_train = int(result["best_n_used_train"])
    best_n_used_test = int(result["best_n_used_test"])
    diagnostics = result["diagnostics"]

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

    diag_dir = Path("diagnostics")
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / f"torch_per_label__{dataset}.best.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(f"Wrote diagnostics to {diag_path}")


if __name__ == "__main__":
    main()
