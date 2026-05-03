# STATUS: EXPERIMENTAL
# Purpose: torch_per_label_mlp variant with a low-rank bottleneck in the stage-2 output projection.
#   Stage 1: torch_per_label (per-label linear ensemble in logit space) trained to its own early stop.
#   Stage 2: Freeze stage-1 weights/bias, then train an active-label-only residual MLP that can
#            make small cross-label adjustments (e.g. "A and B high -> boost C").
#
# Variant change (ONLY):
#   Replace stage-2 projection hidden_dim -> n_active with hidden_dim -> RANK -> n_active
#   to encourage low-rank structure in the cross-label delta.
#
# Safety goals for stage 2:
#   - Must not compromise stage-1 quality.
#   - Residual starts as a near-no-op via (a) zero-initialized final layer and
#     (b) a learnable bounded scalar gate initialized small.
#
# Stage-2 features (active labels only):
#   - Channels 0..2: raw base predictors (log1p-preprocessed)
#   - Channel 3:     stage-1 base logits (active labels)
#
# Training:
#   - Stage 1: BCEWithLogitsLoss, early stop on train subset NDCG@1000 (same as torch_per_label).
#   - Stage 2: BCEWithLogitsLoss on (base + gated_delta) logits, early stop on train subset NDCG@1000.
#   - No explicit delta penalty (requested); only AdamW weight decay.
#
# Debug:
#   - Always prints gate value, residual delta stats, parameter stats, and timing each epoch.
from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label_mlp_rank_bottleneck.py`
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
from benchmarks.torch_per_label import (
    PerLabelWeightedEnsemble,
    train_and_evaluate as train_per_label_and_eval,
)

DEVICE = get_device()

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

LR_STAGE2 = 5e-4
WEIGHT_DECAY_STAGE2 = 0.01
TRAIN_SEED = 0

# Learnable gate configuration (effective gate in (0, DELTA_GATE_MAX))
DELTA_GATE_INIT = 0.02
DELTA_GATE_MAX = 0.2

# Safety: smoothly bound ungated residual outputs to a reasonable range (via tanh).
# (We avoid hard clamp because it can pin the residual at the bound with zero gradients.)
DELTA_CLAMP = 0.5

# MLP hyperparameters
HIDDEN_DIM = 128
DROPOUT_RATE = 0.5

# Low-rank bottleneck hyperparameter (variant-specific)
RANK_BOTTLENECK = 32

# 3 base predictors + base logits
N_CHANNELS = 4


# (Removed csr_to_log1p_tensor, _Timer, and _predict_in_batches_logits in favor of SparseCSRDataset and evaluate_model_batched)


def _label_active_mask(y_train_true: csr_matrix, train_preds: list[csr_matrix]) -> np.ndarray:
    """
    Same policy as torch_nn_split and torch_per_label_residual_lowrank_mix_active:
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


def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
    """
    Robust-ish summary for debugging (CPU/float64 reductions).
    Uses sampling for quantiles if tensor is huge.
    """
    t64 = t.detach().to(dtype=torch.float64)
    flat = t64.reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return {"n": 0.0}

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


class ResidualMLPActive(nn.Module):
    """
    Stage-2 residual head (rank-bottleneck variant):
      inputs:  (B, C=4, L_active)
      outputs: (B, L_active) delta logits

    Dense cross-label adjustment via a per-sample MLP over the flattened features,
    with a low-rank bottleneck in the final projection: hidden_dim -> rank -> n_active.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        n_active: int,
        hidden_dim: int,
        dropout_rate: float,
        rank: int,
    ):
        super().__init__()
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if n_active < 0:
            raise ValueError("n_active must be >= 0")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if not (0.0 <= float(dropout_rate) < 1.0):
            raise ValueError("dropout_rate must be in [0,1)")

        self.n_channels = int(n_channels)
        self.n_active = int(n_active)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.rank = int(rank)

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.n_channels * self.n_active, self.hidden_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        # Rank bottleneck: hidden -> rank -> n_active
        self.fc2a = nn.Linear(self.hidden_dim, self.rank)
        self.fc2b = nn.Linear(self.rank, self.n_active)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Default init for fc1; start as near-no-op by forcing final projection to output ~0.
        nn.init.kaiming_uniform_(self.fc1.weight, a=np.sqrt(5))
        if self.fc1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.fc1.bias, -bound, bound)

        # Default init for fc2a; force fc2b to output exactly 0 at init (policy A).
        nn.init.kaiming_uniform_(self.fc2a.weight, a=np.sqrt(5))
        if self.fc2a.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2a.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.fc2a.bias, -bound, bound)

        nn.init.zeros_(self.fc2b.weight)
        nn.init.zeros_(self.fc2b.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim != 3:
            raise ValueError(f"Expected feats shape (B,C,L_active), got {tuple(feats.shape)}")
        if int(feats.shape[1]) != self.n_channels:
            raise ValueError(f"Expected C={self.n_channels}, got {int(feats.shape[1])}")
        if int(feats.shape[2]) != self.n_active:
            raise ValueError(f"Expected L_active={self.n_active}, got {int(feats.shape[2])}")

        if self.n_active == 0:
            return feats.new_zeros((int(feats.shape[0]), 0))

        x = self.flatten(feats)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc2a(x))
        return self.fc2b(x)


class TwoStagePerLabelMLPActive(nn.Module):
    """
    Wrapper model:
      - frozen base per-label ensemble in logits space
      - trainable residual MLP on active labels only, gated by a bounded scalar
    """

    def __init__(
        self,
        *,
        base: PerLabelWeightedEnsemble,
        active_idx: torch.Tensor,  # int64
        hidden_dim: int,
        dropout_rate: float,
        delta_gate_init: float = DELTA_GATE_INIT,
        rank_bottleneck: int = RANK_BOTTLENECK,
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
        init = float(delta_gate_init)
        maxv = float(DELTA_GATE_MAX)
        if not (0.0 < init < maxv):
            raise ValueError(f"delta_gate_init must be in (0, {maxv}), got {init}")

        p = init / maxv
        raw_init = float(np.log(p / (1.0 - p)))
        self.raw_delta_gate = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        self.residual = ResidualMLPActive(
            n_channels=N_CHANNELS,
            n_active=self.n_active,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            rank=rank_bottleneck,
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

        # Residual head outputs an (unbounded) per-label adjustment signal.
        # We:
        #  1) smoothly bound it with tanh (keeps gradients non-zero vs hard clamp),
        #  2) apply it as a multiplicative reweighting of the *base logits* on active labels,
        #     rather than an additive shift (reduces "push all active logits down" failure modes).
        delta_active = self.residual(feats)  # (B,L_active)
        if DELTA_CLAMP is not None:
            delta_active = float(DELTA_CLAMP) * torch.tanh(delta_active / float(DELTA_CLAMP))

        # Center per sample to prevent uniform scaling/shifting of the entire active-label block.
        # This constrains the residual to "redistribute" scores among active labels.
        delta_active = delta_active - delta_active.mean(dim=1, keepdim=True)

        gate = self.delta_gate()

        out = base_logits.clone()
        # Reweight active logits: out_active = base_active * (1 + gate * delta)
        # This preserves base ordering when delta≈0 and avoids a blanket additive offset.
        out_active = base_logits_active * (1.0 + gate * delta_active)
        out.index_copy_(dim=1, index=self.active_idx, source=out_active)
        return out


# (Removed _predict_in_batches_logits in favor of evaluate_model_batched)


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
    model_name = f"torch_per_label_mlp_rank_bottleneck({','.join(e3)})"
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

    n_samples_train = y_train_true.shape[0]
    n_labels = int(y_train_true.shape[1])
    n_models = len(train_preds)

    # Datasets using SparseCSRDataset
    train_ds = SparseCSRDataset(train_preds, y_train_true, stack_dim=0, transform=lambda xy: (log1p_transform(xy[0]), xy[1]))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type == "cuda"))

    full_train_ds = SparseCSRDataset(train_preds, stack_dim=0, transform=log1p_transform)
    full_train_loader = torch.utils.data.DataLoader(full_train_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_samples_train)
    train_eval_idx = rng.choice(n_samples_train, size=n_eval, replace=False)
    train_eval_preds = [p[train_eval_idx] for p in train_preds]
    y_train_true_eval = y_train_true[train_eval_idx]
    train_eval_ds = SparseCSRDataset(train_eval_preds, stack_dim=0, transform=log1p_transform)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    print("Loading test data...")
    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in e3]
    test_ds = SparseCSRDataset(test_preds, stack_dim=0, transform=log1p_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # -----------------
    # Stage 1: per-label
    # -----------------
    print("\n====================")
    print("Stage 1: torch_per_label training")
    print("====================")

    init_weights_s1: torch.Tensor | None = None
    if cfg.ensemble3_init_weights is not None:
        init_weights_s1 = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)

    model_s1 = PerLabelWeightedEnsemble(
        n_models=n_models,
        n_labels=n_labels,
        init_model_weights=init_weights_s1,
    ).to(DEVICE)
    optimizer_s1 = optim.AdamW(model_s1.parameters(), lr=0.003, weight_decay=0.0)

    result_stage1 = train_per_label_and_eval(
        dataset=dataset,
        ensemble_keys=e3,
        lr=0.003,          # match torch_per_label BEST_LR
        weight_decay=0.0,  # match torch_per_label BEST_WEIGHT_DECAY
        batch_size=256,    # match torch_per_label BEST_BATCH_SIZE
        train_loader=train_loader,
        y_train_true=y_train_true,
        train_eval_loader=train_eval_loader,
        y_train_true_eval=y_train_true_eval,
        test_loader=test_loader,
        y_test_true=y_test_true,
        full_train_loader=full_train_loader,
        model=model_s1,
        optimizer=optimizer_s1,
    )

    best_state = result_stage1["best_state"]
    assert isinstance(best_state, dict)
    base_weights = best_state["weights"]
    base_bias = best_state["bias"]

    # Instantiate frozen base.
    n_models = len(train_preds)
    base = PerLabelWeightedEnsemble(n_models=n_models, n_labels=n_labels, init_model_weights=init_weights_s1).to(DEVICE)
    base.load_state_dict({"weights": base_weights.to(DEVICE), "bias": base_bias.to(DEVICE)})
    base.requires_grad_(False)
    base.eval()

    # -----------------
    # Stage 2: residual MLP
    # -----------------
    print("\n====================")
    print("Stage 2: residual MLP (active labels only; rank bottleneck)")
    print("====================")
    print(f"Rank bottleneck: {RANK_BOTTLENECK}")

    model = TwoStagePerLabelMLPActive(
        base=base,
        active_idx=active_idx.to(DEVICE),
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        delta_gate_init=DELTA_GATE_INIT,
        rank_bottleneck=RANK_BOTTLENECK,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        [
            {"params": model.residual.parameters(), "weight_decay": WEIGHT_DECAY_STAGE2},
            {"params": [model.raw_delta_gate], "weight_decay": 0.0},
        ],
        lr=LR_STAGE2,
        eps=1e-8,
    )
    criterion = nn.BCEWithLogitsLoss()

    criterion = nn.BCEWithLogitsLoss()

    best_metric = float("-inf")
    best_epoch: int | None = None
    best_state2: dict[str, torch.Tensor] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    best_n_used_train: int | None = None
    best_n_used_test: int | None = None
    epochs_no_improve = 0

    # Cache a fixed debug subset for consistent per-epoch delta stats.
    batch_dbg = next(iter(train_eval_loader))
    X_dbg = batch_dbg.to(DEVICE, non_blocking=True)

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        epoch_t0 = time.perf_counter()

        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # --- Train evaluation for early stopping (subset only) ---
        train_res_eval = evaluate_model_batched(model, train_eval_loader, y_train_true_eval, k_values=(10, 1000), device=DEVICE)
        train_ndcg1000 = train_res_eval["ndcg@1000"]
        train_ndcg10 = train_res_eval["ndcg@10"]

        # --- Test evaluation ---
        test_metrics = evaluate_model_batched(model, test_loader, y_test_true, k_values=K_VALUES, f1_k=5, device=DEVICE)

        # ---- Debug (always printed every epoch) ----
        with torch.no_grad():
            gate_val = float(model.delta_gate().detach().cpu().item())

            if model.n_active == 0:
                delta_stats = {"n": 0.0}
                delta_g_stats = {"n": 0.0}
                fc1_stats = _tensor_stats(model.residual.fc1.weight.detach().cpu())
                fc2a_stats = _tensor_stats(model.residual.fc2a.weight.detach().cpu())
                fc2b_stats = _tensor_stats(model.residual.fc2b.weight.detach().cpu())
            else:
                # Activation stats
                batch = next(iter(train_eval_loader))
                xb_dbg = X_dbg
                base_logits_dbg = base(xb_dbg)
                base_active = base_logits_dbg.index_select(dim=1, index=model.active_idx)

                x_active_dbg = xb_dbg.index_select(dim=2, index=model.active_idx)
                feats_dbg = torch.cat([x_active_dbg, base_active.unsqueeze(1)], dim=1)
                delta_dbg = model.residual(feats_dbg)
                if DELTA_CLAMP is not None:
                    delta_dbg = float(DELTA_CLAMP) * torch.tanh(delta_dbg / float(DELTA_CLAMP))

                delta_dbg = delta_dbg - delta_dbg.mean(dim=1, keepdim=True)
                delta_stats = _tensor_stats(delta_dbg.detach().cpu())
                delta_g_stats = _tensor_stats((delta_dbg * model.delta_gate()).detach().cpu())

                # Parameter stats
                fc1_stats = _tensor_stats(model.residual.fc1.weight.detach().cpu())
                fc1b_stats = _tensor_stats(model.residual.fc1.bias.detach().cpu())
                fc2a_stats = _tensor_stats(model.residual.fc2a.weight.detach().cpu())
                fc2ab_stats = _tensor_stats(model.residual.fc2a.bias.detach().cpu())
                fc2b_stats = _tensor_stats(model.residual.fc2b.weight.detach().cpu())
                fc2bb_stats = _tensor_stats(model.residual.fc2b.bias.detach().cpu())

                flat_dbg = model.residual.flatten(feats_dbg)
                h1_dbg = torch.relu(model.residual.fc1(flat_dbg))
                h1_stats = _tensor_stats(h1_dbg.detach().cpu())
                h2_dbg = torch.relu(model.residual.fc2a(model.residual.dropout2(h1_dbg)))
                h2_stats = _tensor_stats(h2_dbg.detach().cpu())

        epoch_dt = time.perf_counter() - epoch_t0

        print(
            f"Stage2 Epoch {epoch:02d} | "
            f"gate={gate_val:.8f} gate_max={DELTA_GATE_MAX:g} delta_clamp={DELTA_CLAMP} "
            f"rank={RANK_BOTTLENECK} lr={LR_STAGE2:g} wd={WEIGHT_DECAY_STAGE2:g} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} train_ndcg@10(subset)={train_ndcg10:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"delta(tanh-bounded, centered): {_fmt_stats(delta_stats)} | gated_delta(mult): {_fmt_stats(delta_g_stats)} | "
            f"h1(relu(fc1)): {_fmt_stats(h1_stats)} | h2(relu(fc2a)): {_fmt_stats(h2_stats)} | "
            f"fc1.w: {_fmt_stats(fc1_stats)} | fc1.b: {_fmt_stats(fc1b_stats)} | "
            f"fc2a.w: {_fmt_stats(fc2a_stats)} | fc2a.b: {_fmt_stats(fc2ab_stats)} | "
            f"fc2b.w: {_fmt_stats(fc2b_stats)} | fc2b.b: {_fmt_stats(fc2bb_stats)} | "
            f"total={epoch_dt:.3f}s"
        )

        current = float(train_ndcg1000)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state2 = {k: v.detach().clone() for k, v in model.state_dict().items()}

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

    assert best_state2 is not None
    assert best_epoch is not None
    assert best_train_metrics is not None
    assert best_test_metrics is not None
    assert best_n_used_train is not None
    assert best_n_used_test is not None

    model.load_state_dict(best_state2)

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
