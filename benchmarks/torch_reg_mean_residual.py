# STATUS: EXPERIMENTAL
# Purpose: Mean-like ensemble with per-label residual weights, trained as a *regression*
#          model to predict numeric targets (0/1) with an unbounded score output.
#
# Rationale:
# - Similar architecture to torch_mean_residual, but replaces BCEWithLogitsLoss with MSELoss.
# - This isolates the effect of "regression-style" training while keeping ranking evaluation
#   (NDCG/F1) unchanged.
#
# Form:
#   score[b, l] = sum_m (w_global[m] * scale[m, l]) * x[b, m, l]
#
# Notes:
# - Targets are dense float tensors in {0,1}.
# - Outputs are unbounded real-valued scores; ranking uses raw scores.
from __future__ import annotations

from pathlib import Path
import sys

# Allow running as a script: `uv run benchmarks/torch_reg_mean_residual.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.datasets import ensemble3_keys, get_dataset_config, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.metrics import load_csr, ndcg_at_k_dense, f1_at_k_dense, update_markdown_scoreboard


def _tensor_stats_1d(t: torch.Tensor) -> dict[str, float]:
    """
    Lightweight numeric stats for debugging. Expects a 1D tensor on any device.
    Returns Python floats.

    Note:
      torch.quantile() can error on very large tensors on some builds/devices.
      For large tensors we compute quantiles on a deterministic subsample.
    """
    if t.ndim != 1:
        raise ValueError(f"_tensor_stats_1d expected 1D tensor, got shape {tuple(t.shape)}")
    if t.numel() == 0:
        return {"n": 0.0}

    x = t.detach().to(dtype=torch.float32)

    # Quantiles can be expensive / unsupported for huge arrays; subsample deterministically.
    # Keep this small to avoid surprising slowdowns during debugging.
    max_q_n = 2_000_000
    xq = x
    if x.numel() > max_q_n:
        # Deterministic stride-based subsample (no RNG, no extra deps).
        step = int(np.ceil(x.numel() / max_q_n))
        xq = x[::step]

    return {
        "n": float(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "p01": float(torch.quantile(xq, 0.01).item()),
        "p50": float(torch.quantile(xq, 0.50).item()),
        "p99": float(torch.quantile(xq, 0.99).item()),
        "max": float(x.max().item()),
    }


def _tensor_stats_all(t: torch.Tensor) -> dict[str, float]:
    """
    Stats for any tensor (flattens). Returns Python floats.
    """
    x = t.detach().reshape(-1)
    return _tensor_stats_1d(x)


DEVICE = get_device()

# Training defaults (intentionally similar to torch_mean_residual)
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

# Hyperparameters for multiplicative per-label scaling
LR = 0.003
WEIGHT_DECAY = 0.0  # rely on explicit scale penalty
LAMBDA_SCALE_L2 = 1e-2  # strength of shrinkage of per-label scales toward 1

# Reproducibility
TRAIN_SEED = 0


class MeanResidualEnsemble(nn.Module):
    """
    Mean-like ensemble with per-label multiplicative scaling factors.

    Input:
        x: (batch, M=3, L) raw base model scores (no log1p preprocessing assumed)
    Output:
        score: (batch, L) unbounded real-valued scores

    Form:
        score[b, l] = sum_m (global_w[m] * scale[m, l]) * x[b, m, l]

    Notes:
        - global_w is a softmax over learned logits (non-negative, sums to 1 over models).
        - scale is constrained to be positive and close to 1 via parameterization + regularization.
        - We intentionally do NOT renormalize weights per label across models (as requested).
    """

    # Keep this conservative; can be tuned via CLI later if needed.
    SCALE_ALPHA = 0.5  # scale in (1-alpha, 1+alpha)

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
        # We learn unconstrained logits and normalize with softmax in effective_w()
        # so that global weights are always non-negative and sum to 1 (over models).
        self._global_logits = nn.Parameter(torch.log(w0))  # (M,)

        # Per-label multiplicative scaling (around 1). Learnable.
        # scale_raw is unconstrained; scale = 1 + alpha * tanh(scale_raw) -> (1-alpha, 1+alpha)
        self.scale_raw = nn.Parameter(torch.zeros((n_models, n_labels), dtype=torch.float32))

    def effective_w(self) -> torch.Tensor:
        # (M, L)
        global_w = torch.softmax(self._global_logits, dim=0)  # (M,), sums to 1
        scale = 1.0 + self.SCALE_ALPHA * torch.tanh(self.scale_raw)  # (M, L), positive
        return global_w[:, None] * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, M, L), got {tuple(x.shape)}")
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected x with (M={self.n_models}, L={self.n_labels}), got {tuple(x.shape)}"
            )

        w_eff = self.effective_w()  # (M, L)
        score = (x * w_eff.unsqueeze(0)).sum(dim=1)  # (B, L)
        return score

    def scale_l2(self) -> torch.Tensor:
        # Mean squared deviation from 1, for scale-invariant regularization.
        scale = 1.0 + self.SCALE_ALPHA * torch.tanh(self.scale_raw)
        return ((scale - 1.0) ** 2).mean()



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
            scores = model(xb)
            outs.append(scores.detach().cpu())
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
        "--lambda-scale",
        type=float,
        default=LAMBDA_SCALE_L2,
        help="L2 shrinkage strength for per-label multiplicative scales (toward 1)",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=100.0,
        help="Weight multiplier for positive targets (y=1) in weighted MSE.",
    )
    args = parser.parse_args()
    dataset = str(args.dataset)
    lambda_scale = float(args.lambda_scale)
    pos_weight = float(args.pos_weight)
    if not np.isfinite(pos_weight) or pos_weight <= 0.0:
        raise ValueError("--pos-weight must be a positive finite float")
    if not np.isfinite(lambda_scale) or lambda_scale < 0.0:
        raise ValueError("--lambda-scale must be a finite non-negative float")

    # Deterministic-ish
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_reg_mean_residual({','.join(ensemble_keys)})"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]

    # Keep X_train on CPU; move minibatches to GPU.
    # Use raw base model scores directly (no log1p preprocessing).
    X_train = torch.stack([torch.from_numpy(p.toarray()).float() for p in train_preds], dim=1)

    # Regression targets are numeric 0/1.
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
    # Use raw base model scores directly (no log1p preprocessing).
    X_test = torch.stack([torch.from_numpy(p.toarray()).float() for p in test_preds], dim=1)

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

    model = MeanResidualEnsemble(n_models=n_models, n_labels=n_labels, init_global=init_global).to(
        DEVICE
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )

    # Weighted MSE to counter extreme class imbalance in dense {0,1} targets.
    # This reduces the degenerate "predict ~0 everywhere" optimum.
    def weighted_mse_loss(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if scores.shape != y.shape:
            raise ValueError(f"Shape mismatch: scores={tuple(scores.shape)} y={tuple(y.shape)}")
        w = torch.ones_like(y, dtype=scores.dtype, device=scores.device)
        w = torch.where(y > 0.0, w * pos_weight, w)
        return (w * (scores - y) ** 2).mean()

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
                scores = model(xb)
                loss_main = weighted_mse_loss(scores, yb)
                loss_reg_scale = lambda_scale * model.scale_l2()
                loss_reg = loss_reg_scale
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

        # --- Always-on debug diagnostics (no CLI flags) ---
        with torch.no_grad():
            # Parameters/regularization stats
            scale_l2 = float(model.scale_l2().detach().cpu().item())

            # Global weights (softmax-normalized; sums to 1 over models)
            w_global = torch.softmax(model._global_logits.detach(), dim=0)

            # Effective weights for diagnostics: w_eff = global_w * scale
            w_eff = model.effective_w().detach()

            # Scale tensor itself (to detect saturation at bounds)
            scale = (w_eff / (w_global[:, None] + 1e-12)).detach()

            # Output distribution on the early-stop subset (to spot scale issues)
            subset_scores = train_scores_eval.detach()

        # Per-model scale summaries (mean abs deviation from 1, max abs deviation)
        mean_abs_scale_dev_per_model = (scale - 1.0).abs().mean(dim=1).cpu().numpy()
        max_abs_scale_dev_per_model = (scale - 1.0).abs().amax(dim=1).cpu().numpy()

        # Detect if tanh is saturating: |scale_raw| large => tanh ~ +/-1
        mean_abs_scale_raw_per_model = model.scale_raw.detach().abs().mean(dim=1).cpu().numpy()
        max_abs_scale_raw_per_model = model.scale_raw.detach().abs().amax(dim=1).cpu().numpy()

        # Effective weight stats per model (helps catch runaway weight scaling without renorm)
        mean_w_eff_per_model = w_eff.mean(dim=1).cpu().numpy()
        max_w_eff_per_model = w_eff.amax(dim=1).cpu().numpy()

        diag = (
            " | "
            f"scale_l2={scale_l2:.6e} "
            f"mean_abs_scale_dev=[{mean_abs_scale_dev_per_model[0]:.3e},"
            f"{mean_abs_scale_dev_per_model[1]:.3e},"
            f"{mean_abs_scale_dev_per_model[2]:.3e}] "
            f"max_abs_scale_dev=[{max_abs_scale_dev_per_model[0]:.3e},"
            f"{max_abs_scale_dev_per_model[1]:.3e},"
            f"{max_abs_scale_dev_per_model[2]:.3e}] "
            f"mean_w_eff=[{mean_w_eff_per_model[0]:.3e},"
            f"{mean_w_eff_per_model[1]:.3e},"
            f"{mean_w_eff_per_model[2]:.3e}] "
            f"max_w_eff=[{max_w_eff_per_model[0]:.3e},"
            f"{max_w_eff_per_model[1]:.3e},"
            f"{max_w_eff_per_model[2]:.3e}] "
            f"max_abs_scale_raw=[{max_abs_scale_raw_per_model[0]:.3e},"
            f"{max_abs_scale_raw_per_model[1]:.3e},"
            f"{max_abs_scale_raw_per_model[2]:.3e}] "
        )

        w_stats = _tensor_stats_1d(w_global)
        eff_stats = _tensor_stats_all(w_eff)
        scale_stats = _tensor_stats_all(scale)
        s_stats = _tensor_stats_all(subset_scores)

        extra = (
            "\n"
            "  debug:\n"
            f"    global_w: mean={w_stats['mean']:.6f} std={w_stats['std']:.6f} "
            f"min={w_stats['min']:.6f} p50={w_stats['p50']:.6f} max={w_stats['max']:.6f}\n"
            f"    w_eff:    mean={eff_stats['mean']:.6e} std={eff_stats['std']:.6e} "
            f"min={eff_stats['min']:.6e} p50={eff_stats['p50']:.6e} max={eff_stats['max']:.6e}\n"
            f"    scale:    mean={scale_stats['mean']:.6e} std={scale_stats['std']:.6e} "
            f"min={scale_stats['min']:.6e} p50={scale_stats['p50']:.6e} max={scale_stats['max']:.6e}\n"
            f"    scores:   mean={s_stats['mean']:.6e} std={s_stats['std']:.6e} "
            f"min={s_stats['min']:.6e} p50={s_stats['p50']:.6e} max={s_stats['max']:.6e}"
        )

        print(
            f"[lambda_scale={lambda_scale:g} pos_weight={pos_weight:g}] "
            f"Epoch {epoch:02d} | "
            f"loss={loss.item():.6f} (wmse={loss_main.item():.6f} reg={loss_reg.item():.6f}) | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"timing train={float(t_train.dt or 0.0):.3f}s "
            f"pred_train={float(t_pred_train.dt or 0.0):.3f}s "
            f"pred_test={float(t_pred_test.dt or 0.0):.3f}s"
            f"{diag}"
            f"{extra}"
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
