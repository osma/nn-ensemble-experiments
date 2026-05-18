# STATUS: EXPERIMENTAL
# Purpose: Two-stage training for torch_nn_split_per_label.
#
# Stage 1: Train only the base layers (conv + scale_raw) until early stopping.
# Stage 2: Unfreeze all parameters (MLP starts zero-initialized) and continue
#          training until early stopping.
#
# No intermediate checkpoint files — best model state is kept in memory.
#
# Model:
#   mean_all = Conv1d over sources (initialized from dataset init weights if available)
#   scale_active = bounded_scale(scale_raw)  # per-label per-source scaling [0.1x, 10x]
#   mean_active = sum_m w[m] * x_active[m,:] * scale[m,:]
#   delta_active = MLP(flatten(x_active))    # MLP sees raw inputs, init to zero
#   out_active = clamp(mean_active + delta_active, [0,1])
#   out_inactive = clamp(mean_inactive, [0,1])
#   out = stitch active/inactive into full label space
#
# Training:
#   - BCELoss on probabilities
#   - Early stopping by train subset NDCG@1000 (same policy as other models)
#   - Default max 12 epochs per stage
#
# Diagnostics:
#   - Prints label activity stats and sparsity stats
#   - Prints conv weights, per-label scale stats, delta stats, and train/test metrics each epoch
from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_nn_split_per_label_staged.py`
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
from benchmarks.metrics import (
    load_csr,
    evaluate_model_batched,
    update_markdown_scoreboard,
)
from benchmarks.preprocessing import SparseCSRDataset, sqrt_transform

DEVICE = get_device()

# Training defaults
EPOCHS = 30
K_VALUES = (10, 1000)
PATIENCE = 5
MIN_EPOCHS = 4

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512

EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

LR = 1e-3
LR_STAGE2 = 5e-4  # Moderate LR for Stage 2 MLP fine-tuning
WEIGHT_DECAY = 0.01
TRAIN_SEED = 0

# Model hyperparameters (smaller than torch_nn)
HIDDEN_DIM = 64
DROPOUT_RATE = 0.5

# Per-label scaling bounds for active labels
SCALE_MIN = 0.1
SCALE_MAX = 10.0


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
    # CSR nnz per row: diff of indptr
    return float(np.mean(np.diff(x.indptr)))


def _bounded_scale_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """
    Map unconstrained raw values to a positive multiplicative scale in [SCALE_MIN, SCALE_MAX].

    We use exp + clamp in log-space:
        scale = exp(clamp(raw, log(min), log(max)))
    """
    lo = float(np.log(SCALE_MIN))
    hi = float(np.log(SCALE_MAX))
    return torch.exp(torch.clamp(raw, min=lo, max=hi))


class NNSplitPerLabelEnsembleModel(nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        n_labels: int,
        active_idx: torch.Tensor,  # int64, sorted ascending
        hidden_dim: int,
        dropout_rate: float,
        init_global: torch.Tensor | None,
        warm_start: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        if active_idx.ndim != 1:
            raise ValueError("active_idx must be 1D")
        self.register_buffer("active_idx", active_idx.long())
        self.n_labels = int(n_labels)
        self.n_active = int(active_idx.numel())
        self.source_dim = int(source_dim)

        self.model_config = {
            "source_dim": self.source_dim,
            "n_labels": self.n_labels,
            "n_active": self.n_active,
            "hidden_dim": int(hidden_dim),
            "dropout_rate": float(dropout_rate),
            "scale_min": float(SCALE_MIN),
            "scale_max": float(SCALE_MAX),
        }

        # Global mean-like mixer for all labels (conv weights constrained via softmax in forward).
        self.conv = nn.Conv1d(self.source_dim, 1, 1, bias=False)

        # Per-label (active-only) per-source scaling applied before mean mixing.
        # Shape: (M, L_active). Parameterized in log-space (raw) and bounded in forward.
        self.scale_raw = nn.Parameter(torch.zeros(self.source_dim, self.n_active))

        # Small MLP only for active labels (MLP sees *raw* x_active per request).
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden = nn.Linear(self.source_dim * self.n_active, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.delta_layer = nn.Linear(hidden_dim, self.n_active)

        self.reset_parameters(init_global=init_global, warm_start=warm_start)

    def reset_parameters(
        self,
        *,
        init_global: torch.Tensor | None,
        warm_start: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Initialize parameters.

        If `warm_start` is provided (from torch_per_label), we initialize:
          - conv weights to the mean per-model weight across labels (a good global prior)
          - scale_raw for active labels so that mean mixing approximates torch_per_label on active labels

        We intentionally keep the MLP delta initialized to zero so the model starts as a
        mostly-linear ensemble and only learns corrections if beneficial.
        """
        with torch.no_grad():
            if warm_start is not None:
                ws_w = warm_start["weights"]  # (M, L)
                ws_b = warm_start["bias"]  # (L,)

                if int(ws_w.shape[0]) != self.source_dim:
                    raise ValueError(
                        f"Warm start weights have M={int(ws_w.shape[0])}, expected source_dim={self.source_dim}"
                    )
                if int(ws_w.shape[1]) != self.n_labels:
                    raise ValueError(
                        f"Warm start weights have L={int(ws_w.shape[1])}, expected n_labels={self.n_labels}"
                    )
                if int(ws_b.shape[0]) != self.n_labels:
                    raise ValueError(
                        f"Warm start bias has L={int(ws_b.shape[0])}, expected n_labels={self.n_labels}"
                    )

                # 1) Global conv: use per-model mean weight across labels, normalized.
                w_mean = ws_w.mean(dim=1)  # (M,)
                w_mean = torch.clamp(w_mean, min=1e-12)
                w_mean = w_mean / w_mean.sum()
                self.conv.weight.copy_(w_mean.reshape(1, self.source_dim, 1).to(self.conv.weight))

                # 2) Active-label scaling: set scale so that (w_global * scale) ~= per-label weights.
                #    For each active label l and model m:
                #      desired_scale[m,l] = ws_w[m,l] / w_global[m]
                #    Then set scale_raw = log(clamp(desired_scale, [SCALE_MIN,SCALE_MAX])).
                #    This makes the initial mean_active close to torch_per_label (ignoring bias and clamp).
                w_global = w_mean.to(ws_w)  # (M,)
                w_active = ws_w.index_select(dim=1, index=self.active_idx.to(ws_w.device))  # (M, L_active)

                denom = w_global.unsqueeze(1).clamp(min=1e-12)
                desired_scale = w_active / denom
                desired_scale = torch.clamp(desired_scale, min=SCALE_MIN, max=SCALE_MAX)
                self.scale_raw.copy_(torch.log(desired_scale).to(self.scale_raw))

            else:
                if init_global is not None:
                    if init_global.numel() != self.source_dim:
                        raise ValueError("init_global must have length source_dim")
                    w = init_global.reshape(1, self.source_dim, 1).to(self.conv.weight)
                    self.conv.weight.copy_(w)
                else:
                    self.conv.weight.fill_(1.0 / float(self.source_dim))

                # scale_raw=0 => scale=1 (no-op)
                self.scale_raw.zero_()

        # Start as pure mean mixer: delta == 0
        nn.init.zeros_(self.delta_layer.weight)
        nn.init.zeros_(self.delta_layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs to have shape (B, M, L), got {tuple(inputs.shape)}")
        if int(inputs.shape[1]) != self.source_dim:
            raise ValueError(f"Expected M={self.source_dim}, got {int(inputs.shape[1])}")
        if int(inputs.shape[2]) != self.n_labels:
            raise ValueError(f"Expected L={self.n_labels}, got {int(inputs.shape[2])}")

        # Global convex weights (sum to 1).
        w = torch.softmax(self.conv.weight[:, :, 0], dim=1)  # (1, M)

        # Base mean for all labels (inactive labels always use this).
        mean_all = torch.sum(inputs * w.unsqueeze(-1), dim=1)  # (B, L)

        if self.n_active == 0:
            return torch.clamp(mean_all, min=0.0, max=1.0)

        # Active slice.
        x_active = inputs.index_select(dim=2, index=self.active_idx)  # (B, M, L_active)

        # Per-label scaling for active labels before mean mixing.
        # scale_raw: (M, L_active) -> scale: (M, L_active) in [SCALE_MIN, SCALE_MAX]
        scale = _bounded_scale_from_raw(self.scale_raw).to(x_active)
        x_active_adj = x_active * scale.unsqueeze(0)  # (B, M, L_active)

        mean_active = torch.sum(x_active_adj * w.unsqueeze(-1), dim=1)  # (B, L_active)

        # MLP delta uses raw x_active per request.
        x = self.flatten(x_active)
        x = self.dropout1(x)
        x = F.relu(self.hidden(x))
        x = self.dropout2(x)
        delta_active = self.delta_layer(x)  # (B, L_active)

        out_active = torch.clamp(mean_active + delta_active, min=0.0, max=1.0)

        out = torch.clamp(mean_all, min=0.0, max=1.0)
        out.index_copy_(dim=1, index=self.active_idx, source=out_active)
        return out


def _delta_stats(
    model: NNSplitPerLabelEnsembleModel, loader: torch.utils.data.DataLoader
) -> tuple[float, float]:

    sum_abs = 0.0
    n_abs = 0

    max_samples = 1_000_000
    samples: list[torch.Tensor] = []

    with torch.no_grad():
        for xb in loader:
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb = xb.to(DEVICE, non_blocking=True)
            x_active = xb.index_select(dim=2, index=model.active_idx)

            x = model.flatten(x_active)
            x = model.dropout1(x)
            x = F.relu(model.hidden(x))
            x = model.dropout2(x)
            delta = model.delta_layer(x)

            a = delta.abs().detach().cpu().reshape(-1)

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


def _scale_stats(model: NNSplitPerLabelEnsembleModel) -> tuple[float, float, float]:
    """
    Return (mean_scale, p95_scale, max_scale) over the learned active-label scales.
    """
    if model.n_active == 0:
        return 1.0, 1.0, 1.0
    with torch.no_grad():
        s = _bounded_scale_from_raw(model.scale_raw.detach()).reshape(-1).cpu()
        mean_s = float(s.mean().item())
        # quantile on <= (3*L_active) which is safe; but keep it consistent with sampling approach.
        s_sorted, _ = torch.sort(s)
        q_idx = min(int(round(0.95 * (s_sorted.numel() - 1))), s_sorted.numel() - 1)
        p95_s = float(s_sorted[q_idx].item())
        max_s = float(s_sorted[-1].item())
        return mean_s, p95_s, max_s


def _train_stage(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    train_eval_loader: torch.utils.data.DataLoader,
    full_train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    y_train_true: csr_matrix,
    y_train_true_eval: csr_matrix,
    y_test_true: csr_matrix,
    *,
    stage: str,
    max_epochs: int,
    patience: int,
    min_epochs: int,
) -> tuple[dict[str, torch.Tensor], int, float, dict[str, float], dict[str, float], int, int]:
    """
    Train a single stage. Returns (best_state, best_epoch, best_metric, best_train_metrics,
    best_test_metrics, best_n_used_train, best_n_used_test).
    """
    best_metric = float("-inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    best_n_used_train: int | None = None
    best_n_used_test: int | None = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        last_loss: float | None = None
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        # Early stop metric: train subset NDCG@1000
        train_res_eval = evaluate_model_batched(
            model, train_eval_loader, y_train_true_eval, k_values=(10, 1000), device=DEVICE
        )
        train_ndcg1000 = train_res_eval["ndcg@1000"]
        train_ndcg10 = train_res_eval["ndcg@10"]

        # Test metrics
        test_metrics = evaluate_model_batched(
            model, test_loader, y_test_true, k_values=K_VALUES, f1_k=5, device=DEVICE
        )

        with torch.no_grad():
            conv_w = (
                torch.softmax(model.conv.weight.detach()[:, :, 0], dim=1)
                .reshape(-1)
                .cpu()
                .numpy()
                .tolist()
            )
        scale_mean, scale_p95, scale_max = _scale_stats(model)
        delta_mean_abs, delta_p95_abs = _delta_stats(model, train_eval_loader)

        epoch_dt = time.perf_counter() - epoch_t0
        print(
            f"[{stage.upper():7s}] Epoch {epoch:02d} | "
            f"loss={float(last_loss or 0.0):.6f} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} "
            f"train_ndcg@10(subset)={train_ndcg10:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"conv_w={','.join(f'{w:.4f}' for w in conv_w)} | "
            f"scale mean={scale_mean:.4f} p95={scale_p95:.4f} max={scale_max:.4f} | "
            f"delta|x| mean={delta_mean_abs:.6f} p95={delta_p95_abs:.6f} | "
            f"total={epoch_dt:.3f}s"
        )

        current = float(train_ndcg1000)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Full train metrics computed only at best snapshot
            best_train_metrics_res = evaluate_model_batched(
                model, full_train_loader, y_train_true, k_values=K_VALUES, device=DEVICE
            )
            best_train_metrics = {k: v for k, v in best_train_metrics_res.items() if k.startswith("ndcg")}
            best_n_used_train = int(best_train_metrics_res["n_used"])

            best_test_metrics = test_metrics.copy()
            best_n_used_test = int(test_metrics["n_used"])
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch >= min_epochs and epochs_no_improve >= patience:
            break

    assert best_state is not None
    assert best_epoch is not None
    assert best_train_metrics is not None
    assert best_test_metrics is not None
    assert best_n_used_train is not None
    assert best_n_used_test is not None

    return best_state, best_epoch, best_metric, best_train_metrics, best_test_metrics, best_n_used_train, best_n_used_test


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
    model_base = "torch_nn_split_per_label_staged"
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
    n_models = len(train_preds)

    # Datasets using SparseCSRDataset
    train_ds = SparseCSRDataset(train_preds, y_train_true, stack_dim=0, transform=lambda xy: (sqrt_transform(xy[0]), xy[1]))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type == "cuda"))

    full_train_ds = SparseCSRDataset(train_preds, stack_dim=0, transform=sqrt_transform)
    full_train_loader = torch.utils.data.DataLoader(full_train_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_samples_train)
    train_eval_idx = rng.choice(n_samples_train, size=n_eval, replace=False)
    train_eval_preds = [p[train_eval_idx] for p in train_preds]
    y_train_true_eval = y_train_true[train_eval_idx]
    train_eval_ds = SparseCSRDataset(train_eval_preds, stack_dim=0, transform=sqrt_transform)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    print("Loading test data...")
    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in e3]
    test_ds = SparseCSRDataset(test_preds, stack_dim=0, transform=sqrt_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    if n_models != 3:
        raise ValueError(f"Expected 3-way ensemble input (M=3), got M={n_models}")

    init_global: torch.Tensor | None = None
    if cfg.ensemble3_init_weights is not None:
        init_global = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)

    model = NNSplitPerLabelEnsembleModel(
        source_dim=n_models,
        n_labels=n_labels,
        active_idx=active_idx,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        init_global=init_global,
        warm_start=None,  # Stage 1 learns base from scratch
    ).to(DEVICE)

    criterion = nn.BCELoss()

    # =========================================================================
    # STAGE 1: Train base only (conv + scale_raw), freeze MLP
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: Training base layers (conv + scale_raw) only")
    print("=" * 80)

    for name, param in model.named_parameters():
        if name not in ("conv.weight", "scale_raw"):
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )

    (
        best_state_s1,
        best_epoch_s1,
        best_metric_s1,
        best_train_s1,
        best_test_s1,
        best_n_used_train_s1,
        best_n_used_test_s1,
    ) = _train_stage(
        model,
        optimizer,
        criterion,
        train_loader,
        train_eval_loader,
        full_train_loader,
        test_loader,
        y_train_true,
        y_train_true_eval,
        y_test_true,
        stage="stage1",
        max_epochs=EPOCHS,
        patience=PATIENCE,
        min_epochs=MIN_EPOCHS,
    )

    print(f"\nStage 1 complete: best epoch={best_epoch_s1}, train_ndcg@1000={best_metric_s1:.6f}")
    print(f"  Train NDCG@10={best_train_s1.get('ndcg@10', 0):.6f}  NDCG@1000={best_train_s1.get('ndcg@1000', 0):.6f}")
    print(f"  Test  NDCG@10={best_test_s1['ndcg@10']:.6f}  NDCG@1000={best_test_s1['ndcg@1000']:.6f}  F1@5={best_test_s1['f1@5']:.6f}")

    # =========================================================================
    # STAGE 2: Unfreeze all parameters, continue training
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: Unfreezing all parameters and continuing training")
    print("=" * 80)

    for param in model.parameters():
        param.requires_grad = True

    # Load best state from stage 1
    model.load_state_dict(best_state_s1)

    # Reset optimizer to include all parameters with lower LR for Stage 2
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR_STAGE2,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )

    (
        best_state_s2,
        best_epoch_s2,
        best_metric_s2,
        best_train_s2,
        best_test_s2,
        best_n_used_train_s2,
        best_n_used_test_s2,
    ) = _train_stage(
        model,
        optimizer,
        criterion,
        train_loader,
        train_eval_loader,
        full_train_loader,
        test_loader,
        y_train_true,
        y_train_true_eval,
        y_test_true,
        stage="stage2",
        max_epochs=EPOCHS,
        patience=PATIENCE,
        min_epochs=MIN_EPOCHS,
    )

    print(f"\nStage 2 complete: best epoch={best_epoch_s2}, train_ndcg@1000={best_metric_s2:.6f}")
    print(f"  Train NDCG@10={best_train_s2.get('ndcg@10', 0):.6f}  NDCG@1000={best_train_s2.get('ndcg@1000', 0):.6f}")
    print(f"  Test  NDCG@10={best_test_s2['ndcg@10']:.6f}  NDCG@1000={best_test_s2['ndcg@1000']:.6f}  F1@5={best_test_s2['f1@5']:.6f}")

    # Use stage 2 best model for final results
    model.load_state_dict(best_state_s2)

    # Save Stage 1 results to scoreboard with _s1 suffix (before parentheses)
    model_s1 = f"{model_base}_s1({','.join(e3)})"
    update_markdown_scoreboard(
        path=scoreboard_path,
        model=model_s1,
        dataset=dataset,
        split="train",
        metrics=best_train_s1,
        n_samples=best_n_used_train_s1,
        epoch=best_epoch_s1,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model=model_s1,
        dataset=dataset,
        split="test",
        metrics=best_test_s1,
        n_samples=best_n_used_test_s1,
        epoch=best_epoch_s1,
    )

    # Save Stage 2 results to scoreboard with _s2 suffix (before parentheses)
    model_s2 = f"{model_base}_s2({','.join(e3)})"
    update_markdown_scoreboard(
        path=scoreboard_path,
        model=model_s2,
        dataset=dataset,
        split="train",
        metrics=best_train_s2,
        n_samples=best_n_used_train_s2,
        epoch=best_epoch_s2,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model=model_s2,
        dataset=dataset,
        split="test",
        metrics=best_test_s2,
        n_samples=best_n_used_test_s2,
        epoch=best_epoch_s2,
    )

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(
        f"Stage 1 best | epoch={best_epoch_s1} | "
        f"test_ndcg@10={best_test_s1['ndcg@10']:.6f} | "
        f"test_ndcg@1000={best_test_s1['ndcg@1000']:.6f} | "
        f"f1@5={best_test_s1['f1@5']:.6f}"
    )
    print(
        f"Stage 2 best | epoch={best_epoch_s2} | "
        f"test_ndcg@10={best_test_s2['ndcg@10']:.6f} | "
        f"test_ndcg@1000={best_test_s2['ndcg@1000']:.6f} | "
        f"f1@5={best_test_s2['f1@5']:.6f}"
    )
    print(f"\nSaved best result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
