# STATUS: EXPERIMENTAL
# Purpose: Two-stage model.
#   Stage 1: torch_per_label (per-label linear ensemble in logit space) trained to its own early stop.
#   Stage 2: Freeze stage-1 weights/bias, then train an active-label-only residual MLP that can
#            make small cross-label adjustments (e.g. "A and B high -> boost C").
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

# Allow running as a script: `uv run benchmarks/torch_per_label_mlp.py`
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

# Safety: clamp ungated residual logits to a reasonable bound.
DELTA_CLAMP = 0.5

# MLP hyperparameters
HIDDEN_DIM = 128
DROPOUT_RATE = 0.5

# 3 base predictors + base logits
N_CHANNELS = 4


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
    Stage-2 residual head:
      inputs:  (B, C=4, L_active)
      outputs: (B, L_active) delta logits

    Dense cross-label adjustment via a per-sample MLP over the flattened features.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        n_active: int,
        hidden_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if n_active < 0:
            raise ValueError("n_active must be >= 0")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if not (0.0 <= float(dropout_rate) < 1.0):
            raise ValueError("dropout_rate must be in [0,1)")

        self.n_channels = int(n_channels)
        self.n_active = int(n_active)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.n_channels * self.n_active, self.hidden_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_active)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Default init for fc1; start as near-no-op by forcing fc2 to output ~0.
        nn.init.kaiming_uniform_(self.fc1.weight, a=np.sqrt(5))
        if self.fc1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.fc1.bias, -bound, bound)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

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
        return self.fc2(x)


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
    model_name = f"torch_per_label_mlp({','.join(e3)})"
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
    # Targets: raw dense 0/1 for BCEWithLogitsLoss.
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

    # -----------------
    # Stage 2: residual MLP
    # -----------------
    print("\n====================")
    print("Stage 2: residual MLP (active labels only)")
    print("====================")

    model = TwoStagePerLabelMLPActive(
        base=base,
        active_idx=active_idx.to(DEVICE),
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        delta_gate_init=DELTA_GATE_INIT,
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

    # Cache a fixed debug subset for consistent per-epoch delta stats.
    X_dbg = X_train_eval

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        epoch_t0 = time.perf_counter()

        model.train()
        with _Timer() as t_train:
            last_loss: float | None = None
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
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

        # ---- Debug (always printed every epoch) ----
        with torch.no_grad():
            gate_val = float(model.delta_gate().detach().cpu().item())

            if model.n_active == 0:
                delta_stats = {"n": 0.0}
                delta_g_stats = {"n": 0.0}
                fc1_stats = _tensor_stats(model.residual.fc1.weight.detach().cpu())
                fc2_stats = _tensor_stats(model.residual.fc2.weight.detach().cpu())
            else:
                xb_dbg = X_dbg.to(DEVICE, non_blocking=True)
                base_logits_dbg = base(xb_dbg)
                base_active = base_logits_dbg.index_select(dim=1, index=model.active_idx)

                x_active_dbg = xb_dbg.index_select(dim=2, index=model.active_idx)
                feats_dbg = torch.cat([x_active_dbg, base_active.unsqueeze(1)], dim=1)
                delta_dbg = model.residual(feats_dbg)
                if DELTA_CLAMP is not None:
                    delta_dbg = delta_dbg.clamp(-float(DELTA_CLAMP), float(DELTA_CLAMP))

                delta_stats = _tensor_stats(delta_dbg.detach().cpu())
                delta_g_stats = _tensor_stats((delta_dbg * model.delta_gate()).detach().cpu())

                # Parameter stats (to diagnose "delta collapses to constant" failures)
                fc1_stats = _tensor_stats(model.residual.fc1.weight.detach().cpu())
                fc1b_stats = _tensor_stats(model.residual.fc1.bias.detach().cpu())
                fc2_stats = _tensor_stats(model.residual.fc2.weight.detach().cpu())
                fc2b_stats = _tensor_stats(model.residual.fc2.bias.detach().cpu())

                # Activation stats (to see whether fc1 output is saturating / dead)
                flat_dbg = model.residual.flatten(feats_dbg)
                h1_dbg = torch.relu(model.residual.fc1(flat_dbg))
                h1_stats = _tensor_stats(h1_dbg.detach().cpu())

        epoch_dt = time.perf_counter() - epoch_t0

        print(
            f"Stage2 Epoch {epoch:02d} | "
            f"loss={float(last_loss or 0.0):.6f} | "
            f"gate={gate_val:.8f} gate_max={DELTA_GATE_MAX:g} delta_clamp={DELTA_CLAMP} "
            f"lr={LR_STAGE2:g} wd={WEIGHT_DECAY_STAGE2:g} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} train_ndcg@10(subset)={train_ndcg10:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"delta: {_fmt_stats(delta_stats)} | gated_delta: {_fmt_stats(delta_g_stats)} | "
            f"h1(relu(fc1)): {_fmt_stats(h1_stats)} | "
            f"fc1.w: {_fmt_stats(fc1_stats)} | fc1.b: {_fmt_stats(fc1b_stats)} | "
            f"fc2.w: {_fmt_stats(fc2_stats)} | fc2.b: {_fmt_stats(fc2b_stats)} | "
            f"timing train={float(t_train.dt or 0.0):.3f}s "
            f"pred_train={float(t_pred_train.dt or 0.0):.3f}s "
            f"pred_test={float(t_pred_test.dt or 0.0):.3f}s "
            f"total={epoch_dt:.3f}s"
        )

        current = float(train_ndcg1000)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state2 = {k: v.detach().clone() for k, v in model.state_dict().items()}

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
