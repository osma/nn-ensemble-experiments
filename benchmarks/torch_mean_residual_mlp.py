# STATUS: EXPERIMENTAL
# Purpose: torch_mean_residual + a small cross-label MLP correction applied only to "active" labels.
#
# Rationale:
# - torch_mean_residual is a strong baseline (logit space, BCEWithLogitsLoss, explicit shrinkage).
# - Some datasets (notably koko) benefit from cross-label influence (co-occurrence effects).
# - We add a small MLP that produces *additive logit deltas* for active labels only.
#
# Key design constraints (to avoid overfitting):
# - MLP is a *residual* correction: its output head is zero-initialized.
# - A learnable gating scalar alpha starts at 0 (via parameter log_alpha=0 -> alpha=0).
# - We add an explicit L2 penalty on the MLP output deltas (mean(delta^2)).
# - Only active labels (present in train truth or any train predictor nnz) get MLP corrections.
#
# Form:
#   base_logits[b, l] = sum_m (w_global[m] + delta_w[m, l]) * x[b, m, l] + bias[l]
#   delta_active[b, la] = MLP(flatten(x[b, :, active_labels]))
#   logits[b, active_labels] = base_logits[b, active_labels] + alpha * delta_active[b, :]
#   logits[b, inactive_labels] = base_logits[b, inactive_labels]
#
# Training:
# - BCEWithLogitsLoss on logits (no clamp/sigmoid)
# - Early stopping by train subset NDCG@1000 (no test leakage for selection)
# - Metrics printed each epoch (train subset + test) for convenience
from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_mean_residual_mlp.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csr_matrix

from benchmarks.datasets import ensemble3_keys, get_dataset_config, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.preprocessing import csr_to_log1p_tensor
from benchmarks.metrics import load_csr, ndcg_at_k_dense, f1_at_k_dense, update_markdown_scoreboard

DEVICE = get_device()

# Training defaults (intentionally similar to torch_mean_residual / torch_per_label)
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

# Base residual hyperparameters (same intent as torch_mean_residual)
LR = 0.003
WEIGHT_DECAY = 0.01  # match torch_nn_split-style stabilization (curbs MLP weight growth)
LAMBDA_DELTA_L2 = 1e-2  # shrinkage for per-label residual weights
LAMBDA_BIAS_L2 = 1e-3  # shrinkage for per-label bias

# MLP hyperparameters (controlled capacity)
MLP_HIDDEN_DIM = 32
MLP_DROPOUT = 0.5
LAMBDA_MLP_OUT_L2 = 1e-4  # penalize mean(delta_active^2) to keep corrections small
LAMBDA_MLP_DELTA_BIAS_L2 = 1e-3  # penalize delta_layer.bias to avoid acting like an extra per-label bias

# Reproducibility
TRAIN_SEED = 0


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


def _label_active_mask(y_train_true: csr_matrix, train_preds: list[csr_matrix]) -> np.ndarray:
    """
    Active if label appears at least once in:
      - train truth (y_train_true.indices), OR
      - any train prediction matrix (pred.indices)

    "Any nnz counts" (no thresholding).
    """
    n_labels = int(y_train_true.shape[1])
    truth_active = np.zeros(n_labels, dtype=bool)
    if y_train_true.nnz:
        truth_active[np.unique(y_train_true.indices)] = True

    pred_active = np.zeros(n_labels, dtype=bool)
    for p in train_preds:
        if p.nnz:
            pred_active[np.unique(p.indices)] = True

    return truth_active | pred_active


def _csr_avg_nnz_per_row(x: csr_matrix) -> float:
    if x.shape[0] == 0:
        return 0.0
    return float(np.mean(np.diff(x.indptr)))


def _delta_active_stats(
    model: "MeanResidualMLPEnsemble", x_cpu_subset: torch.Tensor
) -> tuple[float, float]:
    """
    Compute mean(|alpha*delta_active|) and p95(|alpha*delta_active|) over a CPU subset.

    We sample up to 1e6 elements for robust quantile estimation.
    """
    if model.n_active == 0:
        return 0.0, 0.0

    model.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_cpu_subset),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
    )

    sum_abs = 0.0
    n_abs = 0

    max_samples = 1_000_000
    samples: list[torch.Tensor] = []

    with torch.no_grad():
        alpha = float(model.alpha().detach().cpu().item())
        for (xb_cpu,) in loader:
            xb = xb_cpu.to(DEVICE, non_blocking=True)
            delta = model.mlp_delta_active(xb)  # (B, L_active)
            a = (alpha * delta).abs().detach().cpu().reshape(-1)

            sum_abs += float(a.sum().item())
            n_abs += int(a.numel())

            if max_samples > 0:
                remaining = max_samples - sum(int(s.numel()) for s in samples)
                if remaining <= 0:
                    max_samples = 0
                else:
                    if a.numel() <= remaining:
                        samples.append(a)
                    else:
                        idx = torch.randperm(a.numel())[:remaining]
                        samples.append(a.index_select(0, idx))

    if n_abs == 0:
        return 0.0, 0.0

    mean_abs = sum_abs / float(n_abs)

    if not samples:
        return float(mean_abs), 0.0

    v = torch.cat(samples, dim=0)
    v, _ = torch.sort(v)
    q_idx = min(int(round(0.95 * (v.numel() - 1))), v.numel() - 1)
    p95_abs = float(v[q_idx].item())
    return float(mean_abs), float(p95_abs)


def _tensor_stats_1d(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().reshape(-1).to(dtype=torch.float32)
    if x.numel() == 0:
        return {"n": 0.0}
    # Deterministic subsample for quantiles (avoid slow/unsupported on huge tensors)
    max_q = 2_000_000
    if x.numel() > max_q:
        step = int(np.ceil(x.numel() / max_q))
        xq = x[::step]
    else:
        xq = x
    return {
        "n": float(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "p50": float(torch.quantile(xq, 0.50).item()),
        "p99": float(torch.quantile(xq, 0.99).item()),
        "max": float(x.max().item()),
        "mean_abs": float(x.abs().mean().item()),
    }


def _prob_saturation_stats(p: torch.Tensor) -> dict[str, float]:
    p = p.detach()
    if p.numel() == 0:
        return {"n": 0.0}
    n = float(p.numel())
    return {
        "p<1e-6": float((p < 1e-6).sum().item()) / n,
        "p<1e-4": float((p < 1e-4).sum().item()) / n,
        "p<1e-3": float((p < 1e-3).sum().item()) / n,
        "p>0.5": float((p > 0.5).sum().item()) / n,
        "p>0.9": float((p > 0.9).sum().item()) / n,
    }


def _decomp_debug(model: "MeanResidualMLPEnsemble", x_cpu_subset: torch.Tensor, *, eps: float) -> str:
    """
    One-pass diagnostic on the early-stop subset to attribute failures:
    base vs mlp contribution vs saturation.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_cpu_subset),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
    )

    base_chunks: list[torch.Tensor] = []
    full_chunks: list[torch.Tensor] = []
    mlp_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        alpha = float(model.alpha().detach().cpu().item())
        for (xb_cpu,) in loader:
            xb = xb_cpu.to(DEVICE, non_blocking=True)

            base = model.base_logits(xb)  # (B, L)
            full = model(xb)  # (B, L)

            base_chunks.append(base.detach().cpu())
            full_chunks.append(full.detach().cpu())

            if model.n_active > 0:
                delta_active = model.mlp_delta_active(xb)  # (B, L_active)
                mlp = (alpha * delta_active).detach().cpu()
                mlp_chunks.append(mlp)

    base_all = torch.cat(base_chunks, dim=0)
    full_all = torch.cat(full_chunks, dim=0)

    if mlp_chunks:
        mlp_all = torch.cat(mlp_chunks, dim=0)  # only active labels
    else:
        mlp_all = torch.zeros((base_all.shape[0], 0), dtype=torch.float32)

    # Probability view (for saturation diagnostics)
    p = torch.sigmoid(full_all)
    p = torch.clamp(p, min=eps, max=1.0 - eps)

    b = _tensor_stats_1d(base_all)
    f = _tensor_stats_1d(full_all)
    if mlp_all.numel():
        m = _tensor_stats_1d(mlp_all)
    else:
        m = {"n": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0, "mean_abs": 0.0}
    sat = _prob_saturation_stats(p)

    # Parameter diagnostics (cheap, but crucial for “is it the bias?”)
    dlb = _tensor_stats_1d(model.delta_layer.bias.detach().cpu())
    dlw_mean_abs = float(model.delta_layer.weight.detach().abs().mean().cpu().item())
    bb = _tensor_stats_1d(model.bias.detach().cpu())

    return (
        "\n  decomp_debug:\n"
        f"    base_logits: mean={b['mean']:.3e} std={b['std']:.3e} p50={b['p50']:.3e} p99={b['p99']:.3e} min={b['min']:.3e} max={b['max']:.3e}\n"
        f"    mlp_add(active): mean={m['mean']:.3e} std={m['std']:.3e} p50={m['p50']:.3e} p99={m['p99']:.3e} min={m['min']:.3e} max={m['max']:.3e} mean_abs={m['mean_abs']:.3e}\n"
        f"    full_logits: mean={f['mean']:.3e} std={f['std']:.3e} p50={f['p50']:.3e} p99={f['p99']:.3e} min={f['min']:.3e} max={f['max']:.3e}\n"
        f"    probs(clamped): p<1e-6={sat['p<1e-6']:.4f} p<1e-4={sat['p<1e-4']:.4f} p<1e-3={sat['p<1e-3']:.4f} p>0.5={sat['p>0.5']:.4f} p>0.9={sat['p>0.9']:.4f}\n"
        f"    params: base_bias mean={bb['mean']:.3e} p99={bb['p99']:.3e} min={bb['min']:.3e} max={bb['max']:.3e} | "
        f"mlp_delta_bias mean={dlb['mean']:.3e} p99={dlb['p99']:.3e} min={dlb['min']:.3e} max={dlb['max']:.3e} | "
        f"mlp_delta_weight mean_abs={dlw_mean_abs:.3e}\n"
    )


class MeanResidualMLPEnsemble(nn.Module):
    """
    Mean-residual ensemble (logit space) with an MLP cross-label correction
    applied only to active labels.

    Input:
        x: (B, M=3, L) log1p-preprocessed scores (non-negative)
    Output:
        logits: (B, L) raw logits
    """

    def __init__(
        self,
        *,
        n_models: int,
        n_labels: int,
        active_idx: torch.Tensor,  # int64, sorted ascending
        init_global: torch.Tensor | None,
        mlp_hidden_dim: int,
        mlp_dropout: float,
    ):
        super().__init__()
        if n_models != 3:
            raise ValueError("This experimental model is intended for 3-way ensembles only")
        if active_idx.ndim != 1:
            raise ValueError("active_idx must be 1D")

        self.n_models = int(n_models)
        self.n_labels = int(n_labels)
        self.register_buffer("active_idx", active_idx.long())
        self.n_active = int(self.active_idx.numel())

        # --- Base torch_mean_residual parameters ---
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

        # Global weights are represented as unconstrained logits and normalized with softmax
        # so they are always non-negative and sum to 1.
        self.global_logits = nn.Parameter(torch.log(w0 + 1e-12))  # (M,)
        self.delta_w = nn.Parameter(torch.zeros((n_models, n_labels), dtype=torch.float32))  # (M, L)
        self.bias = nn.Parameter(torch.zeros((n_labels,), dtype=torch.float32))  # (L,)

        # --- MLP correction on active labels only ---
        # Gate alpha: bounded to avoid MLP dominating the base model.
        # alpha = alpha_max * sigmoid(log_alpha)
        self.log_alpha = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        self.alpha_max = 0.1

        # MLP:
        #   flatten (M * L_active) -> hidden_dim -> L_active
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(float(mlp_dropout))
        self.hidden = nn.Linear(self.n_models * self.n_active, int(mlp_hidden_dim))
        self.dropout2 = nn.Dropout(float(mlp_dropout))
        self.delta_layer = nn.Linear(int(mlp_hidden_dim), self.n_active)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Base parameters: delta_w and bias are already zero-init by construction.

        # MLP: start with no correction.
        # - delta layer weights/bias to zero ensures delta_active == 0 at init regardless of hidden output.
        nn.init.zeros_(self.delta_layer.weight)
        nn.init.zeros_(self.delta_layer.bias)

        # Hidden layer: default init is fine; it won't matter until delta_layer learns non-zero weights.

        # Gate alpha starts small but non-zero so MLP can start learning,
        # while still being strongly bounded by alpha_max.
        with torch.no_grad():
            self.log_alpha.fill_(-3.0)

    def alpha(self) -> torch.Tensor:
        # Bounded positive gate so the MLP cannot take over.
        # With log_alpha=-3, sigmoid ~ 0.047, so alpha starts ~0.0047 when alpha_max=0.1.
        return float(self.alpha_max) * torch.sigmoid(self.log_alpha)

    def global_w(self) -> torch.Tensor:
        # Always sums to 1.
        return torch.softmax(self.global_logits, dim=0)

    def effective_w(self) -> torch.Tensor:
        return self.global_w()[:, None] + self.delta_w  # (M, L)

    def base_logits(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.effective_w()  # (M, L)
        return (x * w_eff.unsqueeze(0)).sum(dim=1) + self.bias  # (B, L)

    def mlp_delta_active(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, L)
        if self.n_active == 0:
            # Maintain device & dtype
            return x.new_zeros((x.shape[0], 0))
        x_active = x.index_select(dim=2, index=self.active_idx)  # (B, M, L_active)
        h = self.flatten(x_active)  # (B, M*L_active)
        h = self.dropout1(h)
        h = F.relu(self.hidden(h))
        h = self.dropout2(h)
        return self.delta_layer(h)  # (B, L_active)

    def forward(self, x: torch.Tensor, *, return_delta_active: bool = False):
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, M, L), got {tuple(x.shape)}")
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected x with (M={self.n_models}, L={self.n_labels}), got {tuple(x.shape)}"
            )

        logits = self.base_logits(x)

        if self.n_active == 0:
            if return_delta_active:
                return logits, x.new_zeros((x.shape[0], 0))
            return logits

        delta_active = self.mlp_delta_active(x)  # (B, L_active)
        alpha = self.alpha()

        # Add correction only to active labels.
        logits = logits.clone()  # avoid in-place on graph view
        logits.index_add_(dim=1, index=self.active_idx, source=alpha * delta_active)

        if return_delta_active:
            return logits, delta_active
        return logits

    def delta_l2(self) -> torch.Tensor:
        return (self.delta_w ** 2).mean()

    def bias_l2(self) -> torch.Tensor:
        return (self.bias ** 2).mean()


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
        "--loss",
        type=str,
        default="prob_epsclamp",
        choices=["logits", "prob_epsclamp"],
        help=(
            "Training loss variant. "
            "'logits' uses BCEWithLogitsLoss on raw logits. "
            "'prob_epsclamp' uses sigmoid(logits) -> clamp(eps,1-eps) -> BCELoss."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Epsilon for probability clamping when --loss=prob_epsclamp",
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
        "--lambda-mlp-out",
        type=float,
        default=LAMBDA_MLP_OUT_L2,
        help="L2 penalty strength for mean(delta_active^2) (keeps MLP corrections small)",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=MLP_HIDDEN_DIM,
        help="Hidden/bottleneck dimension for MLP correction (capacity control)",
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=MLP_DROPOUT,
        help="Dropout probability for MLP correction",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debugging diagnostics each epoch",
    )
    args = parser.parse_args()

    dataset = str(args.dataset)
    loss_kind = str(args.loss)
    eps = float(args.eps)
    if loss_kind == "prob_epsclamp" and not (0.0 < eps < 0.5):
        raise ValueError("--eps must satisfy 0 < eps < 0.5")
    lambda_delta = float(args.lambda_delta)
    lambda_bias = float(args.lambda_bias)
    lambda_mlp_out = float(args.lambda_mlp_out)
    mlp_hidden_dim = int(args.mlp_hidden_dim)
    mlp_dropout = float(args.mlp_dropout)
    debug = bool(args.debug)

    # Deterministic-ish
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_mean_residual_mlp({','.join(ensemble_keys)})"
    if loss_kind != "prob_epsclamp":
        model_name = f"{model_name}[loss={loss_kind}]"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]

    # Active labels based on train-only data.
    active_mask = _label_active_mask(y_train_true, train_preds)
    active_idx_np = np.flatnonzero(active_mask).astype(np.int64)
    active_idx = torch.from_numpy(active_idx_np)

    n_labels = int(y_train_true.shape[1])
    n_active = int(active_idx_np.size)
    n_inactive = int(n_labels - n_active)
    print(
        "Label activity | "
        f"n_labels={n_labels} "
        f"active={n_active} ({(100.0*n_active/max(1,n_labels)):.2f}%) "
        f"inactive={n_inactive}"
    )

    print(
        "Train sparsity | "
        f"truth avg nnz/row={_csr_avg_nnz_per_row(y_train_true):.2f} | "
        + " | ".join(
            f"{k} avg nnz/row={_csr_avg_nnz_per_row(p):.2f}"
            for k, p in zip(ensemble_keys, train_preds, strict=True)
        )
    )

    # Keep X_train on CPU; move minibatches to GPU.
    X_train = torch.stack([csr_to_log1p_tensor(p) for p in train_preds], dim=1)

    # Targets are binary; use dense float for BCEWithLogitsLoss.
    Y_train = torch.from_numpy(y_train_true.toarray()).float()

    # Fixed random subset for early stopping
    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_train = int(X_train.shape[0])
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_train)
    train_eval_idx = rng.choice(n_train, size=n_eval, replace=False)
    X_train_eval = X_train[train_eval_idx]
    y_train_true_eval = y_train_true[train_eval_idx]

    print("Loading test data...")
    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in ensemble_keys]
    X_test = torch.stack([csr_to_log1p_tensor(p) for p in test_preds], dim=1)

    n_models = int(X_train.shape[1])
    if n_models != 3:
        raise ValueError(f"Expected 3-way ensemble input (M=3), got M={n_models}")

    cfg = get_dataset_config(dataset)
    init_global: torch.Tensor | None = None
    if cfg.ensemble3_init_weights is not None:
        init_global = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)
        if init_global.shape[0] != n_models:
            raise ValueError(
                f"ensemble3_init_weights has length {init_global.shape[0]}, but X_train has n_models={n_models}."
            )

    model = MeanResidualMLPEnsemble(
        n_models=n_models,
        n_labels=n_labels,
        active_idx=active_idx,
        init_global=init_global,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_dropout=mlp_dropout,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )
    # --- Loss weighting for extreme class imbalance ---
    # Per-label pos_weight (neg/pos) makes false negatives more expensive and
    # prevents the degenerate "push everything down" solution that can tank ranking.
    #
    # Clamp to avoid huge weights for ultra-rare labels.
    pos = np.asarray(y_train_true.sum(axis=0)).ravel().astype(np.float32)  # (L,)
    n_rows = float(y_train_true.shape[0])
    neg = n_rows - pos
    pos_weight = neg / np.maximum(pos, 1.0)
    pos_weight = np.minimum(pos_weight, 100.0).astype(np.float32)
    pos_weight_t = torch.from_numpy(pos_weight).to(DEVICE)

    criterion_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    criterion_prob = nn.BCELoss()

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
            last_loss: float | None = None
            last_loss_bce: float | None = None
            last_loss_reg: float | None = None
            last_loss_reg_mlp: float | None = None

            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits, delta_active = model(xb, return_delta_active=True)

                # Always train with weighted logits-BCE for stability under imbalance.
                # Keep prob_epsclamp as an optional debug/ablation mode.
                if loss_kind == "prob_epsclamp":
                    probs = torch.sigmoid(logits)
                    probs = torch.clamp(probs, min=eps, max=1.0 - eps)
                    loss_main = criterion_prob(probs, yb)
                else:
                    loss_main = criterion_logits(logits, yb)

                loss_reg_delta = lambda_delta * model.delta_l2()
                loss_reg_bias = lambda_bias * model.bias_l2()

                # Encourage MLP correction to stay small (in logit units).
                if delta_active.numel() == 0:
                    loss_reg_mlp = logits.new_tensor(0.0)
                else:
                    alpha = model.alpha()
                    # Penalize actual applied correction magnitude.
                    loss_reg_mlp = lambda_mlp_out * (alpha * delta_active).pow(2).mean()

                # Prevent delta_layer.bias from acting like an extra per-label bias vector.
                loss_reg_mlp_bias = LAMBDA_MLP_DELTA_BIAS_L2 * (model.delta_layer.bias ** 2).mean()

                loss_reg = loss_reg_delta + loss_reg_bias + loss_reg_mlp + loss_reg_mlp_bias
                loss = loss_main + loss_reg

                loss.backward()
                optimizer.step()

                last_loss = float(loss.item())
                last_loss_bce = float(loss_main.item())
                last_loss_reg = float(loss_reg.item())
                last_loss_reg_mlp = float(loss_reg_mlp.item())

        # --- Early stop metric: train subset NDCG@1000 ---
        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)
        train_ndcg1000, _n_used_train_eval = ndcg_at_k_dense(
            y_train_true_eval, train_scores_eval, k=1000
        )
        train_ndcg10, _ = ndcg_at_k_dense(y_train_true_eval, train_scores_eval, k=10)

        # --- Test metrics (observational; not used for selection) ---
        with _Timer() as t_pred_test:
            test_scores = _predict_in_batches(model, X_test)

        test_metrics: dict[str, float] = {}
        n_used_test: int | None = None
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg
        f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
        test_metrics["f1@5"] = f1

        # Diagnostics
        with torch.no_grad():
            w_global = model.global_w().detach().cpu().numpy().tolist()
            alpha_val = float(model.alpha().detach().cpu().item())
            delta_l2 = float(model.delta_l2().detach().cpu().item())
            bias_l2 = float(model.bias_l2().detach().cpu().item())
            mean_abs_delta_per_model = model.delta_w.detach().abs().mean(dim=1).cpu().numpy()
            mean_abs_bias = float(model.bias.detach().abs().mean().cpu().item())
            max_abs_bias = float(model.bias.detach().abs().max().cpu().item())

        delta_mean_abs, delta_p95_abs = _delta_active_stats(model, X_train_eval)

        dbg = ""
        if debug:
            dbg = (
                " | "
                f"delta_l2={delta_l2:.6e} "
                f"mean_abs_delta=[{mean_abs_delta_per_model[0]:.3e},"
                f"{mean_abs_delta_per_model[1]:.3e},"
                f"{mean_abs_delta_per_model[2]:.3e}] "
                f"bias_l2={bias_l2:.6e} "
                f"mean_abs_bias={mean_abs_bias:.3e} "
                f"max_abs_bias={max_abs_bias:.3e} "
                f"mlp|alpha*delta| mean={delta_mean_abs:.6f} p95={delta_p95_abs:.6f}"
            )

        decomp = _decomp_debug(model, X_train_eval, eps=eps)

        print(
            f"[loss={loss_kind}{f' eps={eps:g}' if loss_kind == 'prob_epsclamp' else ''} "
            f"lambda_delta={lambda_delta:g} lambda_bias={lambda_bias:g} lambda_mlp_out={lambda_mlp_out:g}] "
            f"Epoch {epoch:02d} | "
            f"loss={float(last_loss or 0.0):.6f} "
            f"(bce={float(last_loss_bce or 0.0):.6f} reg={float(last_loss_reg or 0.0):.6f} "
            f"reg_mlp={float(last_loss_reg_mlp or 0.0):.6f}) | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} "
            f"train_ndcg@10(subset)={train_ndcg10:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"global_w={','.join(f'{w:.4f}' for w in w_global)} "
            f"alpha={alpha_val:.6f} | "
            f"timing train={float(t_train.dt or 0.0):.3f}s "
            f"pred_train={float(t_pred_train.dt or 0.0):.3f}s "
            f"pred_test={float(t_pred_test.dt or 0.0):.3f}s"
            f"{dbg}"
            f"{decomp}"
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
