# STATUS: EXPERIMENTAL
# Purpose: Two-stage model.
#   Stage 1: torch_per_label (per-label linear ensemble in logit space) trained to its own early stop.
#   Stage 2: Freeze stage-1 weights/bias, then train a cross-label low-rank residual ("mix")
#            applied ONLY on active labels.
#
# Stage-2 residual is non-symmetric:
#   input  features: (C=4, L_active) where
#       - channels 0..2: raw base predictors (log1p-preprocessed)
#       - channel 3:     stage-1 base logits (active labels)
#   output residual: (1, L_active) delta logits, added to base logits.
#
# Cross-label coupling is provided by low-rank factors over labels:
#   U: (L_active, r)  label -> rank
#   V: (L_active, r)  rank  -> label
#   W: (C, r)         channel mixing in rank space
#
# Training:
#   - Stage 1: same as torch_per_label (BCEWithLogitsLoss), early stop on train subset NDCG@1000.
#   - Stage 2: BCEWithLogitsLoss on (base + delta) logits, early stop on train subset NDCG@1000.
#   - No explicit delta penalty beyond standard AdamW weight decay (requested).
#
# Notes:
#   - Inactive labels are *exactly* stage-1 predictions (no changes).
#   - This script will (re)train stage 1 each run to ensure the early-stopped best snapshot
#     matches the current data/code, then proceed to stage 2.
from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label_residual_lowrank_mix_active.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

from benchmarks.datasets import ensemble3_keys, get_dataset_config, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.metrics import load_csr, ndcg_at_k_dense, f1_at_k_dense, update_markdown_scoreboard
from benchmarks.torch_per_label import (
    PerLabelWeightedEnsemble,
    train_and_evaluate as train_per_label_and_eval,
)

DEVICE = get_device()


def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
    """
    Small helper for debugging parameter / activation health.
    Uses float64 reductions for numeric stability; works on CPU tensors.

    NOTE: torch.quantile() can raise on very large tensors; for robustness we
    downsample when needed.
    """
    t64 = t.detach().to(dtype=torch.float64)
    flat = t64.reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return {"n": 0.0}

    # Guard against `RuntimeError: quantile() input tensor is too large` by sampling.
    # 1e6 float64 values ~= 8MB, safe for debugging.
    max_q = 1_000_000
    if n > max_q:
        idx = torch.randperm(n, device=flat.device)[:max_q]
        flat_q = flat.index_select(0, idx)
    else:
        flat_q = flat

    q = torch.quantile(
        flat_q,
        torch.tensor([0.0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.0], dtype=torch.float64),
    )

    return {
        "n": float(n),
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


def _fmt_stats(s: dict[str, float]) -> str:
    if not s or s.get("n", 0.0) == 0.0:
        return "n=0"
    return (
        f"mean={s['mean']:.4g} std={s['std']:.4g} "
        f"min={s['min']:.4g} p50={s['p50']:.4g} p95={s['p95']:.4g} max={s['max']:.4g}"
    )

# Match torch_per_label defaults / policy
K_VALUES = (10, 1000)
PATIENCE = 2
MIN_EPOCHS = 2
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

# Stage-2 training defaults (kept modest; stage-1 is already strong)
EPOCHS_STAGE2 = 12
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512

# Stage-2 stability tweaks:
# - Lower LR (residual mixing is sensitive; high LR quickly destroys rankings).
# - Use a *learnable, bounded* scalar gate on delta so the residual starts as a near-no-op
#   but can grow if it finds a useful correction.
# - Prefer weight decay / bounded delta over an explicit delta L2 penalty (which can
#   overly suppress learning).
LR_STAGE2 = 1e-4
WEIGHT_DECAY_STAGE2 = 0.01
TRAIN_SEED = 0

# Learnable gate configuration (effective gate in (0, DELTA_GATE_MAX))
DELTA_GATE_INIT = 0.01
DELTA_GATE_MAX = 0.2

# Optional explicit delta penalty (disabled by default; re-enable if collapse returns).
LAMBDA_DELTA_L2 = 0.0

# Optional: clamp ungated residual logits to a reasonable bound for safety.
DELTA_CLAMP = 0.5

# Low-rank residual settings
RANK = 32
N_CHANNELS = 4  # 3 base predictors + base logits


def csr_to_log1p_tensor(csr: csr_matrix) -> torch.Tensor:
    """Match torch_per_label preprocessing: log1p(clamp(x,0))."""
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


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
    Same policy as torch_nn_split:
      Active if label appears at least once in:
        - train truth (y_train_true.indices), OR
        - any train prediction matrix (pred.indices)
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


class LowRankResidualMixActive(nn.Module):
    """
    Stage-2 residual head:
      inputs:  (B, C=4, L_active)
      outputs: (B, L_active) delta logits

    Non-symmetric mapping (C channels -> 1 delta channel) with cross-label coupling via low rank.
    """

    def __init__(self, *, n_channels: int, n_active: int, rank: int):
        super().__init__()
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if n_active < 0:
            raise ValueError("n_active must be >= 0")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.n_channels = int(n_channels)
        self.n_active = int(n_active)
        self.rank = int(rank)

        # Low-rank factors over labels (active subset)
        # Init:
        # - U small random: learns how to project label axis into rank space.
        # - V small random (NOT zeros): allows learning to begin while staying near-no-op.
        # - W small random: channel mixing in rank space.
        self.U = nn.Parameter(0.01 * torch.randn(self.n_active, self.rank))
        self.V = nn.Parameter(1e-3 * torch.randn(self.n_active, self.rank))

        # Channel mixing in rank space (C -> r), elementwise per rank component.
        self.W = nn.Parameter(0.01 * torch.randn(self.n_channels, self.rank))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim != 3:
            raise ValueError(f"Expected feats shape (B,C,L_active), got {tuple(feats.shape)}")
        if int(feats.shape[1]) != self.n_channels:
            raise ValueError(f"Expected C={self.n_channels}, got {int(feats.shape[1])}")
        if int(feats.shape[2]) != self.n_active:
            raise ValueError(f"Expected L_active={self.n_active}, got {int(feats.shape[2])}")

        if self.n_active == 0:
            # Return empty (B,0) tensor on same device/dtype.
            return feats.new_zeros((int(feats.shape[0]), 0))

        # Project label axis to rank per channel:
        #   Z[b,c,k] = sum_l feats[b,c,l] * U[l,k]
        # Implement as (B*C, L) @ (L, r) -> (B*C, r) -> (B,C,r)
        B = int(feats.shape[0])
        x2 = feats.reshape(B * self.n_channels, self.n_active)
        Z = x2 @ self.U  # (B*C, r)
        Z = Z.reshape(B, self.n_channels, self.rank)  # (B,C,r)

        # Mix channels down to one rank vector (elementwise weighting over channels):
        #   h[b,k] = sum_c Z[b,c,k] * W[c,k]
        h = torch.sum(Z * self.W.unsqueeze(0), dim=1)  # (B, r)

        # Project back to labels:
        #   delta[b,l] = sum_k h[b,k] * V[l,k]
        delta = h @ self.V.t()  # (B, L_active)

        # Center residual per sample to prevent a degenerate global logit shift
        # (which can collapse rankings by pushing all active-label logits down).
        if delta.numel():
            delta = delta - delta.mean(dim=1, keepdim=True)

        return delta


class TwoStagePerLabelLowRankMixActive(nn.Module):
    """
    Wrapper model:
      - frozen base per-label ensemble in logits space
      - trainable low-rank cross-label residual on active labels only
    """

    def __init__(
        self,
        *,
        base: PerLabelWeightedEnsemble,
        active_idx: torch.Tensor,  # int64
        rank: int,
        delta_gate_init: float = DELTA_GATE_INIT,
    ):
        super().__init__()
        if active_idx.ndim != 1:
            raise ValueError("active_idx must be 1D")
        self.base = base
        self.register_buffer("active_idx", active_idx.long())
        self.n_labels = int(base.n_labels)
        self.n_models = int(base.n_models)
        self.n_active = int(active_idx.numel())

        # Learnable, bounded scalar gate:
        #   gate = DELTA_GATE_MAX * sigmoid(raw_gate)
        # Initialize raw_gate so that gate ~= delta_gate_init.
        init = float(delta_gate_init)
        maxv = float(DELTA_GATE_MAX)
        if not (0.0 < init < maxv):
            raise ValueError(f"delta_gate_init must be in (0, {maxv}), got {init}")

        p = init / maxv
        raw_init = float(np.log(p / (1.0 - p)))
        self.raw_delta_gate = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        self.residual = LowRankResidualMixActive(
            n_channels=N_CHANNELS,
            n_active=self.n_active,
            rank=rank,
        )

    def delta_gate(self) -> torch.Tensor:
        return float(DELTA_GATE_MAX) * torch.sigmoid(self.raw_delta_gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M=3, L) log1p-preprocessed base predictors.
        returns: (B, L) logits
        """
        base_logits = self.base(x)  # (B,L)

        if self.n_active == 0:
            return base_logits

        x_active = x.index_select(dim=2, index=self.active_idx)  # (B,M,L_active)
        base_logits_active = base_logits.index_select(dim=1, index=self.active_idx)  # (B,L_active)

        feats = torch.cat([x_active, base_logits_active.unsqueeze(1)], dim=1)  # (B, 4, L_active)

        delta_active = self.residual(feats)  # (B,L_active)
        if DELTA_CLAMP is not None:
            delta_active = delta_active.clamp(-float(DELTA_CLAMP), float(DELTA_CLAMP))

        out = base_logits.clone()
        out.index_add_(
            dim=1,
            index=self.active_idx,
            source=(delta_active * self.delta_gate()),
        )
        return out


def _predict_in_batches_logits(model: nn.Module, x_cpu: torch.Tensor) -> torch.Tensor:
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
    args = parser.parse_args()
    dataset = str(args.dataset)

    # Deterministic-ish
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    cfg = get_dataset_config(dataset)
    e3 = ensemble3_keys(dataset)
    model_name = f"torch_per_label_residual_lowrank_mix_active({','.join(e3)})"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in e3]

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
            for k, p in zip(e3, train_preds, strict=True)
        )
    )

    # Inputs use torch_per_label preprocessing (log1p).
    X_train = torch.stack([csr_to_log1p_tensor(p) for p in train_preds], dim=1)
    # Targets: keep as dense float (0/1), consistent with torch_per_label.
    Y_train = csr_to_log1p_tensor(y_train_true)  # y_true data is 0/1; log1p keeps 0/0.693... (not desired)
    # Fix: use raw dense 0/1 for BCEWithLogitsLoss target
    Y_train = torch.from_numpy(y_train_true.toarray()).float()

    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_train = int(X_train.shape[0])
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_train)
    train_eval_idx = rng.choice(n_train, size=n_eval, replace=False)
    X_train_eval = X_train[train_eval_idx]
    y_train_true_eval = y_train_true[train_eval_idx]

    print("Loading test data...")
    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in e3]
    X_test = torch.stack([csr_to_log1p_tensor(p) for p in test_preds], dim=1)

    # -----------------
    # Stage 1: per-label
    # -----------------
    print("\n====================")
    print("Stage 1: torch_per_label training")
    print("====================")

    result_stage1 = train_per_label_and_eval(
        dataset=dataset,
        ensemble_keys=e3,
        lr=0.003,          # match torch_per_label BEST_LR
        weight_decay=0.0,  # match torch_per_label BEST_WEIGHT_DECAY
        batch_size=256,    # match torch_per_label BEST_BATCH_SIZE
        X_train=X_train,
        Y_train=Y_train,
        y_train_true=y_train_true,
        X_train_eval=X_train_eval,
        y_train_true_eval=y_train_true_eval,
        X_test=X_test,
        y_test_true=y_test_true,
    )

    best_state = result_stage1["best_state"]
    assert isinstance(best_state, dict)
    base_weights = best_state["weights"]
    base_bias = best_state["bias"]

    # ---- Stage-1 debug dump (always) ----
    # Helps diagnose cases where stage-2 diverges due to a pathological stage-1 solution.
    with torch.no_grad():
        w_cpu = base_weights.detach().cpu()
        b_cpu = base_bias.detach().cpu()

        w_stats = _tensor_stats(w_cpu)
        b_stats = _tensor_stats(b_cpu)
        w_mean_per_model = w_cpu.mean(dim=1).to(dtype=torch.float64).numpy().tolist()
        w_abs_mean_per_model = w_cpu.abs().mean(dim=1).to(dtype=torch.float64).numpy().tolist()
        w_neg = int((w_cpu < 0).sum().item())

        dominant = torch.argmax(w_cpu, dim=0)
        dominant_counts = torch.bincount(dominant, minlength=w_cpu.shape[0]).to(dtype=torch.int64)
        dominant_frac = (dominant_counts.to(dtype=torch.float64) / float(w_cpu.shape[1])).numpy().tolist()

    print("\nStage 1 debug | per-label base parameters (best snapshot)")
    print(f"  base.weights: {_fmt_stats(w_stats)} | n_negative={w_neg}")
    print(f"  base.bias:    {_fmt_stats(b_stats)}")
    print(f"  base.weights per-model mean:      {', '.join(f'{v:.6f}' for v in w_mean_per_model)}")
    print(f"  base.weights per-model mean|abs|: {', '.join(f'{v:.6f}' for v in w_abs_mean_per_model)}")
    print(f"  base.weights dominant model frac: {', '.join(f'{v:.4f}' for v in dominant_frac)}")

    # Export checkpoint (same location as torch_per_label) for reuse / debugging.
    ckpt_dir = Path(".cache") / "warmstarts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"torch_per_label__{dataset}.best.pt"
    torch.save({"weights": base_weights.detach().cpu(), "bias": base_bias.detach().cpu()}, ckpt_path)
    print(f"Stage 1: wrote checkpoint to {ckpt_path}")

    # Instantiate frozen base.
    n_models = int(X_train.shape[1])
    if n_models != 3:
        raise ValueError(f"Expected 3-way ensemble input (M=3), got M={n_models}")
    init_weights: torch.Tensor | None = None
    if cfg.ensemble3_init_weights is not None:
        init_weights = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)

    base = PerLabelWeightedEnsemble(n_models=n_models, n_labels=n_labels, init_model_weights=init_weights).to(DEVICE)
    base.load_state_dict({"weights": base_weights.to(DEVICE), "bias": base_bias.to(DEVICE)})
    base.requires_grad_(False)
    base.eval()

    # Quick consistency check: ensure loaded module produces finite outputs.
    with torch.no_grad():
        xb0 = X_train_eval[: min(8, int(X_train_eval.shape[0]))].to(DEVICE, non_blocking=True)
        logits0 = base(xb0).detach().cpu()
        s0 = _tensor_stats(logits0)
    print(f"Stage 1 debug | base logits on small train subset: {_fmt_stats(s0)}")

    # -----------------
    # Stage 2: residual
    # -----------------
    print("\n====================")
    print("Stage 2: low-rank cross-label residual (active labels only)")
    print("====================")

    model = TwoStagePerLabelLowRankMixActive(
        base=base,
        active_idx=active_idx.to(DEVICE),
        rank=RANK,
        delta_gate_init=DELTA_GATE_INIT,
    ).to(DEVICE)

    # ---- Stage-2 debug dump (initial params) ----
    with torch.no_grad():
        U = model.residual.U.detach().cpu()
        V = model.residual.V.detach().cpu()
        W = model.residual.W.detach().cpu()
    print("\nStage 2 debug | initial residual parameters")
    print(f"  residual.U:   {_fmt_stats(_tensor_stats(U))}")
    print(f"  residual.V:   {_fmt_stats(_tensor_stats(V))}")
    print(f"  residual.W:   {_fmt_stats(_tensor_stats(W))}")

    optimizer = optim.AdamW(
        [
            {"params": model.residual.parameters(), "weight_decay": WEIGHT_DECAY_STAGE2},
            {"params": [model.raw_delta_gate], "weight_decay": 0.0},
        ],
        lr=LR_STAGE2,
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
    best_state2: dict[str, torch.Tensor] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    best_n_used_train: int | None = None
    best_n_used_test: int | None = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        model.train()
        with _Timer() as t_train:
            last_loss: float | None = None
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)

                delta_l2 = logits.new_tensor(0.0)
                if LAMBDA_DELTA_L2:
                    # Optional explicit delta regularization (stability):
                    # Penalize the (gated) residual magnitude on active labels so the residual
                    # cannot quickly dominate and destroy ranking structure.
                    with torch.no_grad():
                        base_logits = base(xb)

                    if model.n_active == 0:
                        delta_l2 = logits.new_tensor(0.0)
                    else:
                        logits_active = logits.index_select(dim=1, index=model.active_idx)
                        base_active = base_logits.index_select(dim=1, index=model.active_idx)
                        delta_active = logits_active - base_active  # includes gating
                        delta_l2 = (delta_active * delta_active).mean()

                loss = criterion(logits, yb) + (float(LAMBDA_DELTA_L2) * delta_l2)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())

        with _Timer() as t_pred_train:
            train_logits_eval = _predict_in_batches_logits(model, X_train_eval)
        train_ndcg1000, _n_used_train_eval = ndcg_at_k_dense(
            y_train_true_eval, train_logits_eval, k=1000
        )
        train_ndcg10, _ = ndcg_at_k_dense(y_train_true_eval, train_logits_eval, k=10)

        with _Timer() as t_pred_test:
            test_logits = _predict_in_batches_logits(model, X_test)

        test_metrics: dict[str, float] = {}
        n_used_test: int | None = None
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_logits, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg
        f1, _ = f1_at_k_dense(y_test_true, test_logits, k=5)
        test_metrics["f1@5"] = f1

        # --- More debug: residual magnitude + parameter health ---
        # Compute delta logits stats on the eval subset (cheap, helps catch divergence).
        with torch.no_grad():
            if model.n_active == 0:
                delta_s = {"n": 0.0}
            else:
                xb_dbg = X_train_eval.to(DEVICE, non_blocking=True)
                base_logits_dbg = base(xb_dbg)
                x_active_dbg = xb_dbg.index_select(dim=2, index=model.active_idx)
                base_logits_active_dbg = base_logits_dbg.index_select(dim=1, index=model.active_idx)
                feats_dbg = torch.cat([x_active_dbg, base_logits_active_dbg.unsqueeze(1)], dim=1)
                delta_dbg = model.residual(feats_dbg).detach().cpu()
                delta_s = _tensor_stats(delta_dbg)

            U_s = _tensor_stats(model.residual.U.detach().cpu())
            V_s = _tensor_stats(model.residual.V.detach().cpu())
            W_s = _tensor_stats(model.residual.W.detach().cpu())

        with torch.no_grad():
            gate_val = float(model.delta_gate().detach().cpu().item())

        print(
            f"Stage2 Epoch {epoch:02d} | "
            f"loss={float(last_loss or 0.0):.6f} (gate={gate_val:.6f} gate_max={DELTA_GATE_MAX:g} delta_clamp={DELTA_CLAMP} lambda_delta_l2={LAMBDA_DELTA_L2:g} lr={LR_STAGE2:g}) | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} train_ndcg@10(subset)={train_ndcg10:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"delta_logits_active: {_fmt_stats(delta_s)} | "
            f"U: {_fmt_stats(U_s)} | V: {_fmt_stats(V_s)} | W: {_fmt_stats(W_s)} | "
            f"timing train={float(t_train.dt or 0.0):.3f}s "
            f"pred_train={float(t_pred_train.dt or 0.0):.3f}s "
            f"pred_test={float(t_pred_test.dt or 0.0):.3f}s"
        )

        current = float(train_ndcg1000)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state2 = {k: v.detach().clone() for k, v in model.residual.state_dict().items()}

            # Full train metrics at best snapshot
            full_train_logits = _predict_in_batches_logits(model, X_train)
            best_train_metrics = {}
            n_used_train_full: int | None = None
            for k in K_VALUES:
                ndcg, n_used_train_full = ndcg_at_k_dense(y_train_true, full_train_logits, k=k)
                best_train_metrics[f"ndcg@{k}"] = ndcg
            best_n_used_train = int(n_used_train_full or 0)

            best_test_metrics = test_metrics.copy()
            best_n_used_test = int(n_used_test or 0)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch >= MIN_EPOCHS and epochs_no_improve >= PATIENCE:
            break

    assert best_state2 is not None
    assert best_epoch is not None
    assert best_train_metrics is not None
    assert best_test_metrics is not None
    assert best_n_used_train is not None
    assert best_n_used_test is not None

    model.residual.load_state_dict(best_state2)

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
