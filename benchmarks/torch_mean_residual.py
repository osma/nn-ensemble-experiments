# STATUS: EXPERIMENTAL
# Purpose: Mean-like ensemble with per-label residual weights, strongly regularized.
#
# Rationale:
# - torch_mean is a strong baseline; this model keeps it as the "default" by using
#   global per-model weights + per-label residuals (shrinkage).
# - Trains on raw logits with BCEWithLogitsLoss (no output clamp), optimizing ranking.
#
# Form:
#   logits[b, l] = sum_m (w_global[m] + delta_w[m, l]) * x[b, m, l] + bias[l]
#
# Where:
#   - w_global is initialized from DatasetConfig.ensemble3_init_weights (if present),
#     else uniform.
#   - delta_w is initialized to 0 and regularized with L2 (weight decay-like, but explicit).
from __future__ import annotations

from pathlib import Path
import sys

# Allow running as a script: `uv run benchmarks/torch_mean_residual.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.datasets import ensemble3_keys, get_dataset_config, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.preprocessing import csr_to_log1p_tensor
from benchmarks.metrics import load_csr, ndcg_at_k_dense, f1_at_k_dense, update_markdown_scoreboard


DEVICE = get_device()

# Training defaults (intentionally similar to torch_per_label)
EPOCHS = 20
K_VALUES = (10, 1000)
PATIENCE = 2
MIN_EPOCHS = 2

# Batch sizes
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512

# Early stop uses train subset NDCG@1000
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

# Hyperparameters for "residual" approach
LR = 0.003
WEIGHT_DECAY = 0.0  # rely on explicit residual penalty
LAMBDA_DELTA_L2 = 1e-2  # strength of shrinkage of per-label residuals toward 0
LAMBDA_BIAS_L2 = 1e-3  # shrinkage for per-label bias (important for large label spaces)

# Reproducibility
TRAIN_SEED = 0


class MeanResidualEnsemble(nn.Module):
    """
    Mean-like ensemble with per-label residual weights.

    Input:
        x: (batch, M=3, L) log1p-preprocessed scores (non-negative)
    Output:
        logits: (batch, L) raw logits
    """

    def __init__(self, n_models: int, n_labels: int, init_global: torch.Tensor | None):
        super().__init__()
        if n_models != 3:
            raise ValueError("This experimental model is intended for 3-way ensembles only")
        self.n_models = int(n_models)
        self.n_labels = int(n_labels)

        if init_global is None:
            w0 = torch.full((n_models,), 1.0 / float(n_models), dtype=torch.float32)
        else:
            if init_global.ndim != 1 or init_global.shape[0] != n_models:
                raise ValueError(
                    f"init_global must have shape ({n_models},), got {tuple(init_global.shape)}"
                )
            w0 = init_global.to(dtype=torch.float32).clone()
            s = float(w0.sum().item())
            if not np.isfinite(s) or s <= 0.0:
                raise ValueError("init_global must sum to a positive finite value")
            w0 = w0 / w0.sum()

        # Global per-model weights (shared over labels). Learnable.
        self.global_w = nn.Parameter(w0)  # (M,)

        # Per-label residual weights initialized to 0. Learnable.
        self.delta_w = nn.Parameter(torch.zeros((n_models, n_labels), dtype=torch.float32))

        # Per-label bias in logit space (helps match label base rates).
        self.bias = nn.Parameter(torch.zeros((n_labels,), dtype=torch.float32))

    def effective_w(self) -> torch.Tensor:
        # (M, L)
        return self.global_w[:, None] + self.delta_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, M, L), got {tuple(x.shape)}")
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected x with (M={self.n_models}, L={self.n_labels}), got {tuple(x.shape)}"
            )

        w_eff = self.effective_w()  # (M, L)
        logits = (x * w_eff.unsqueeze(0)).sum(dim=1) + self.bias  # (B, L)
        return logits

    def delta_l2(self) -> torch.Tensor:
        # Mean squared residual for scale-invariant regularization.
        return (self.delta_w ** 2).mean()

    def bias_l2(self) -> torch.Tensor:
        # Mean squared bias for scale-invariant regularization.
        return (self.bias ** 2).mean()


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
        "--lambda-delta",
        type=float,
        default=LAMBDA_DELTA_L2,
        help="L2 shrinkage strength for per-label residual weights (delta_w)",
    )
    parser.add_argument(
        "--lambda-bias",
        type=float,
        default=LAMBDA_BIAS_L2,
        help="L2 shrinkage strength for per-label bias (bias)",
    )
    parser.add_argument(
        "--print-delta",
        action="store_true",
        help="Print delta_w diagnostics (delta_l2 and per-model mean |delta|) each epoch",
    )
    args = parser.parse_args()
    dataset = str(args.dataset)
    lambda_delta = float(args.lambda_delta)
    lambda_bias = float(args.lambda_bias)
    print_delta = bool(args.print_delta)

    # Deterministic-ish
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_mean_residual({','.join(ensemble_keys)})"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]

    # Keep X_train on CPU; move minibatches to GPU.
    X_train = torch.stack([csr_to_log1p_tensor(p) for p in train_preds], dim=1)

    # Targets are binary; use dense float for BCEWithLogitsLoss.
    Y_train = torch.from_numpy(y_train_true.toarray()).float()

    # Fixed random subset for early stopping
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

    n_models = int(X_train.shape[1])
    n_labels = int(X_train.shape[2])

    cfg = get_dataset_config(dataset)
    init_global: torch.Tensor | None = None
    if cfg.ensemble3_init_weights is not None:
        init_global = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)
        if init_global.shape[0] != n_models:
            raise ValueError(
                f"ensemble3_init_weights has length {init_global.shape[0]}, but X_train has n_models={n_models}."
            )

    model = MeanResidualEnsemble(n_models=n_models, n_labels=n_labels, init_global=init_global).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
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
        model.train()
        with _Timer() as t_train:
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss_main = criterion(logits, yb)
                loss_reg_delta = lambda_delta * model.delta_l2()
                loss_reg_bias = lambda_bias * model.bias_l2()
                loss_reg = loss_reg_delta + loss_reg_bias
                loss = loss_main + loss_reg
                loss.backward()
                optimizer.step()

        # --- Early stop metric: train subset NDCG@1000 ---
        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)
        train_ndcg1000, _n_used_train_eval = ndcg_at_k_dense(
            y_train_true_eval, train_scores_eval, k=1000
        )

        # --- Test metrics (prototype: OK to compute each epoch) ---
        with _Timer() as t_pred_test:
            test_scores = _predict_in_batches(model, X_test)

        test_metrics: dict[str, float] = {}
        n_used_test: int | None = None
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg
        f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
        test_metrics["f1@5"] = f1

        diag = ""
        if print_delta:
            with torch.no_grad():
                delta_l2 = float(model.delta_l2().detach().cpu().item())
                mean_abs_delta_per_model = (
                    model.delta_w.detach().abs().mean(dim=1).cpu().numpy()
                )

                bias_l2 = float(model.bias_l2().detach().cpu().item())
                mean_abs_bias = float(model.bias.detach().abs().mean().cpu().item())
                max_abs_bias = float(model.bias.detach().abs().max().cpu().item())

            diag = (
                " | "
                f"delta_l2={delta_l2:.6e} "
                f"mean_abs_delta=[{mean_abs_delta_per_model[0]:.3e},"
                f"{mean_abs_delta_per_model[1]:.3e},"
                f"{mean_abs_delta_per_model[2]:.3e}] "
                f"bias_l2={bias_l2:.6e} "
                f"mean_abs_bias={mean_abs_bias:.3e} "
                f"max_abs_bias={max_abs_bias:.3e}"
            )

        print(
            f"[lambda_delta={lambda_delta:g} lambda_bias={lambda_bias:g}] "
            f"Epoch {epoch:02d} | "
            f"loss={loss.item():.6f} (bce={loss_main.item():.6f} reg={loss_reg.item():.6f}) | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"timing train={float(t_train.dt or 0.0):.3f}s "
            f"pred_train={float(t_pred_train.dt or 0.0):.3f}s "
            f"pred_test={float(t_pred_test.dt or 0.0):.3f}s"
            f"{diag}"
        )

        current = float(train_ndcg1000)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Full train metrics computed only at best snapshot
            full_train_scores = _predict_in_batches(model, X_train)
            best_train_metrics = {}
            n_used_train_full: int | None = None
            for k in K_VALUES:
                ndcg, n_used_train_full = ndcg_at_k_dense(y_train_true, full_train_scores, k=k)
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
        f"ndcg@10={best_test_metrics['ndcg@10']:.6f} | "
        f"ndcg@1000={best_test_metrics['ndcg@1000']:.6f} | "
        f"f1@5={best_test_metrics['f1@5']:.6f} | "
        f"epoch={best_epoch}"
    )
    print("\nSaved best result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
