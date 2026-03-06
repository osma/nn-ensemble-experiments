# STATUS: ACTIVE (research variant)
# Purpose: torch_per_label + L1 regularization on (W - W0) to encourage per-label
#          sparse deviations from a strong initialization.
#
# Rationale:
# - Keeps the "per-label independent linear ensemble" form that works reliably.
# - L1 on delta encourages per-label model selection without simplex constraints.
from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label_l1_delta.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

from benchmarks.datasets import ensemble3_keys, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.metrics import (
    load_csr,
    ndcg_at_k_dense,
    f1_at_k_dense,
    update_markdown_scoreboard,
)
from benchmarks.preprocessing import csr_to_log1p_tensor

# ============================
# Model
# ============================


class PerLabelWeightedEnsemble(nn.Module):
    """
    Per-label weighted ensemble with bias.

    For each label l:
        score[l] = sum_m w[m, l] * x[m, l] + b[l]

    Input:
        x: (batch, M, L) already log1p-preprocessed.

    Output:
        (batch, L) raw logits.
    """

    def __init__(
        self,
        n_models: int,
        n_labels: int,
        init_model_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_models = int(n_models)
        self.n_labels = int(n_labels)

        if init_model_weights is None:
            init = torch.full((self.n_models,), 1.0 / self.n_models, dtype=torch.float32)
        else:
            if init_model_weights.ndim != 1 or init_model_weights.shape[0] != self.n_models:
                raise ValueError(
                    f"init_model_weights must have shape ({self.n_models},), got {tuple(init_model_weights.shape)}"
                )
            init = init_model_weights.to(dtype=torch.float32).clone()
            s = float(init.sum().item())
            if not np.isfinite(s) or s <= 0.0:
                raise ValueError("init_model_weights must sum to a positive finite value")
            init = init / init.sum()

        # Store W0 (M,) and expanded W0 (M,L) for delta regularization.
        self.register_buffer("w0", init)  # (M,)
        self.register_buffer("w0_mat", init[:, None].repeat(1, self.n_labels))  # (M,L)

        # Trainable per-model, per-label weights; initialize from w0.
        self.weights = nn.Parameter(self.w0_mat.clone())

        # Per-label bias.
        self.bias = nn.Parameter(torch.zeros(self.n_labels))

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

        weighted = x * self.weights.unsqueeze(0)
        return weighted.sum(dim=1) + self.bias

    def l1_delta(self) -> torch.Tensor:
        """
        Mean absolute deviation from initialization, averaged over all weights.
        """
        return (self.weights - self.w0_mat).abs().mean()


# ============================
# Training / evaluation script
# ============================

DEVICE = get_device()
EPOCHS = 10
K_VALUES = (10, 1000)

PATIENCE = 2
MIN_EPOCHS = 2

EVAL_BATCH_SIZE = 512
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

# Base hyperparameters (from torch_per_label tuning)
BEST_LR = 0.003
BEST_WEIGHT_DECAY = 0.0
BEST_BATCH_SIZE = 256

# Variant hyperparameter
DEFAULT_LAMBDA_L1 = 3e-4

# Reproducibility for training shuffles / init
TRAIN_SEED = 0

# Initial per-source weights for the *yso* 3-way ensemble order: [bonsai, fasttext, mllm]
YSo_INIT_SOURCE_WEIGHTS = torch.tensor([0.2418, 0.6090, 0.1492], dtype=torch.float32)


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
    lambda_l1: float,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    y_train_true: csr_matrix,
    X_train_eval: torch.Tensor,
    y_train_true_eval: csr_matrix,
    X_test: torch.Tensor,
    y_test_true: csr_matrix,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if lambda_l1 < 0:
        raise ValueError("lambda_l1 must be nonnegative")

    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    n_models = X_train.shape[1]
    n_labels = X_train.shape[2]

    init_weights: torch.Tensor | None = None
    if dataset in {"yso-fi", "yso-en"} and ensemble_keys == ("bonsai", "fasttext", "mllm"):
        if YSo_INIT_SOURCE_WEIGHTS.shape[0] != n_models:
            raise ValueError(
                f"YSo_INIT_SOURCE_WEIGHTS has length {YSo_INIT_SOURCE_WEIGHTS.shape[0]}, but X_train has n_models={n_models}."
            )
        init_weights = YSo_INIT_SOURCE_WEIGHTS

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
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.perf_counter()

        model.train()
        with _Timer() as t_train_step:
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss_bce = criterion(logits, yb)
                loss = loss_bce + (float(lambda_l1) * model.l1_delta())
                loss.backward()
                optimizer.step()

        # --- Train evaluation for early stopping (subset only) ---
        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)

        with _Timer() as t_metric_train:
            train_ndcg1000, _n_used_train_eval = ndcg_at_k_dense(
                y_train_true_eval, train_scores_eval, k=1000
            )

        # --- Test evaluation ---
        with _Timer() as t_pred_test:
            test_scores = _predict_in_batches(model, X_test)

        test_metrics: dict[str, float] = {}
        n_used_test: int | None = None
        with _Timer() as t_metric_test:
            for k in K_VALUES:
                ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
                test_metrics[f"ndcg@{k}"] = ndcg
            f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
            test_metrics["f1@5"] = f1

        epoch_dt = time.perf_counter() - epoch_t0

        def _dt(timer: _Timer) -> float:
            return float(timer.dt) if timer.dt is not None else 0.0

        print(
            f"[lr={lr:g} wd={weight_decay:g} bs={batch_size} l1={lambda_l1:g}] "
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

            # Full train metrics at best epoch
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

    return {
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "best_train_metrics": best_train_metrics,
        "best_test_metrics": best_test_metrics,
        "best_n_used_train": int(best_n_used_train),
        "best_n_used_test": int(best_n_used_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="yso-fi",
        choices=["yso-fi", "yso-en", "koko"],
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=DEFAULT_LAMBDA_L1,
        help="L1 regularization strength on mean(|W-W0|). Default tuned for stability.",
    )
    args = parser.parse_args()
    dataset = str(args.dataset)
    lambda_l1 = float(args.lambda_l1)

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_per_label_l1_delta({','.join(ensemble_keys)})"

    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]

    X_train = torch.stack([csr_to_log1p_tensor(p) for p in train_preds], dim=1)
    Y_train = torch.from_numpy(y_train_true.toarray()).float()

    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_train = X_train.shape[0]
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_train)
    train_eval_idx = rng.choice(n_train, size=n_eval, replace=False)
    X_train_eval = X_train[train_eval_idx]
    y_train_true_eval = y_train_true[train_eval_idx]

    print("Loading test data...")

    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in ensemble_keys]
    X_test = torch.stack([csr_to_log1p_tensor(p) for p in test_preds], dim=1)

    print(
        "Training | "
        f"lr={BEST_LR:g} | wd={BEST_WEIGHT_DECAY:g} | bs={BEST_BATCH_SIZE} | "
        f"lambda_l1={lambda_l1:g}"
    )

    result = train_and_evaluate(
        dataset=dataset,
        ensemble_keys=ensemble_keys,
        lr=BEST_LR,
        weight_decay=BEST_WEIGHT_DECAY,
        batch_size=BEST_BATCH_SIZE,
        lambda_l1=lambda_l1,
        X_train=X_train,
        Y_train=Y_train,
        y_train_true=y_train_true,
        X_train_eval=X_train_eval,
        y_train_true_eval=y_train_true_eval,
        X_test=X_test,
        y_test_true=y_test_true,
    )

    best_epoch = int(result["best_epoch"])
    best_train_metrics = result["best_train_metrics"]
    best_test_metrics = result["best_test_metrics"]
    best_n_used_train = int(result["best_n_used_train"])
    best_n_used_test = int(result["best_n_used_test"])

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

    print(
        "\nFinal test metrics | "
        f"ndcg@10={float(best_test_metrics['ndcg@10']):.6f} | "
        f"ndcg@1000={float(best_test_metrics['ndcg@1000']):.6f} | "
        f"f1@5={float(best_test_metrics['f1@5']):.6f} | "
        f"epoch={best_epoch}"
    )
    print("\nSaved result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
