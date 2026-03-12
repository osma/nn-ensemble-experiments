# STATUS: EXPERIMENTAL
# Purpose: torch_nn_split + per-label (per-source) scaling for active labels.
#
# Adds a learned per-label scaling matrix S[m, l_active] applied to the inputs
# before the global mean mixer, enabling label-specific "trust" adjustments for
# each source model (boost/suppress) without introducing cross-label effects for
# inactive labels.
#
# Active labels:
#   x_active_adj = x_active * scale(S_raw)  (bounded to [0.1x, 10x])
#   mean_active  = sum_m w[m] * x_active_adj[m, :]
#   delta_active = MLP(flatten(x_active))   (MLP sees raw inputs, per request)
#   out_active   = clamp(mean_active + delta_active, [0,1])
#
# Inactive labels:
#   out_inactive = mean_inactive only (no per-label scaling, no MLP)
#
# Training:
#   - BCELoss on probabilities
#   - Early stopping by train subset NDCG@1000 (same policy)
#   - Default max epochs reduced to 12
#
# Diagnostics:
#   - Prints label activity stats and sparsity stats
#   - Prints conv weights, per-label scale stats, delta stats, and train/test metrics each epoch
from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_nn_split_per_label.py`
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
from benchmarks.metrics import load_csr, ndcg_at_k_dense, f1_at_k_dense, update_markdown_scoreboard

DEVICE = get_device()

# Training defaults
EPOCHS = 12
K_VALUES = (10, 1000)
PATIENCE = 2
MIN_EPOCHS = 2

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512

EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

LR = 1e-3
WEIGHT_DECAY = 0.01
TRAIN_SEED = 0

# Model hyperparameters (smaller than torch_nn)
HIDDEN_DIM = 64
DROPOUT_RATE = 0.5

# Per-label scaling bounds for active labels (requested)
SCALE_MIN = 0.1
SCALE_MAX = 10.0


def _load_torch_per_label_checkpoint(path: str | Path, *, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Load a `torch_per_label` checkpoint saved via `torch.save(...)`.

    Expected keys:
      - "weights": (M, L) float tensor
      - "bias":    (L,)   float tensor
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    obj = torch.load(str(p), map_location=device)

    if isinstance(obj, dict) and "weights" in obj and "bias" in obj:
        weights = obj["weights"]
        bias = obj["bias"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        # Best-effort: support saving a raw model state_dict from torch_per_label.
        if "weights" not in sd or "bias" not in sd:
            raise ValueError("Checkpoint has state_dict but is missing 'weights'/'bias'.")
        weights = sd["weights"]
        bias = sd["bias"]
    else:
        raise ValueError("Unsupported checkpoint format. Expected dict with keys {'weights','bias'} or {'state_dict': ...}.")

    if not isinstance(weights, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError("Invalid checkpoint: 'weights' and 'bias' must be torch.Tensors.")

    if weights.ndim != 2 or bias.ndim != 1:
        raise ValueError(f"Invalid shapes in checkpoint: weights={tuple(weights.shape)}, bias={tuple(bias.shape)}")

    return {"weights": weights.to(device=device), "bias": bias.to(device=device)}


def csr_to_sqrt_tensor(csr: csr_matrix) -> torch.Tensor:
    """
    Convert CSR predictions to a dense torch tensor with fixed sqrt preprocessing:
        sqrt(clamp(x, 0))
    """
    x = torch.from_numpy(csr.toarray()).float()
    return torch.sqrt(torch.clamp(x, min=0.0))


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
            out = model(xb)
            outs.append(out.detach().cpu())
    return torch.cat(outs, dim=0)


def _delta_stats(
    model: NNSplitPerLabelEnsembleModel, x_cpu_subset: torch.Tensor
) -> tuple[float, float]:
    """
    Compute mean(|delta|) and p95(|delta|) over a CPU subset.

    Note: torch.quantile() can error on very large tensors (implementation limits).
    We therefore compute p95 approximately by sampling a bounded number of elements.
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
        for (xb_cpu,) in loader:
            xb = xb_cpu.to(DEVICE, non_blocking=True)
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
        "--warm-start-torch-per-label",
        type=str,
        default="",
        help=(
            "Optional path to a torch_per_label checkpoint (created by benchmarks/torch_per_label.py). "
            "If provided, initializes conv weights and active-label scales from that checkpoint."
        ),
    )
    args = parser.parse_args()
    dataset = str(args.dataset)

    # Deterministic-ish
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    cfg = get_dataset_config(dataset)
    e3 = ensemble3_keys(dataset)
    model_name = f"torch_nn_split_per_label({','.join(e3)})"
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

    X_train = torch.stack([csr_to_sqrt_tensor(p) for p in train_preds], dim=1)
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
    X_test = torch.stack([csr_to_sqrt_tensor(p) for p in test_preds], dim=1)

    n_models = int(X_train.shape[1])
    if n_models != 3:
        raise ValueError(f"Expected 3-way ensemble input (M=3), got M={n_models}")

    init_global: torch.Tensor | None = None
    if cfg.ensemble3_init_weights is not None:
        init_global = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)

    warm_start: dict[str, torch.Tensor] | None = None
    if str(args.warm_start_torch_per_label).strip():
        warm_start = _load_torch_per_label_checkpoint(args.warm_start_torch_per_label, device=DEVICE)
        print(f"Warm start: loaded torch_per_label checkpoint from {args.warm_start_torch_per_label}")

    model = NNSplitPerLabelEnsembleModel(
        source_dim=n_models,
        n_labels=n_labels,
        active_idx=active_idx,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        init_global=init_global,
        warm_start=warm_start,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )
    criterion = nn.BCELoss()

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
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())

        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)
        train_ndcg1000, _n_used_train_eval = ndcg_at_k_dense(
            y_train_true_eval, train_scores_eval, k=1000
        )
        train_ndcg10, _ = ndcg_at_k_dense(y_train_true_eval, train_scores_eval, k=10)

        with _Timer() as t_pred_test:
            test_scores = _predict_in_batches(model, X_test)

        test_metrics: dict[str, float] = {}
        n_used_test: int | None = None
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg
        f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
        test_metrics["f1@5"] = f1

        with torch.no_grad():
            conv_w = (
                torch.softmax(model.conv.weight.detach()[:, :, 0], dim=1)
                .reshape(-1)
                .cpu()
                .numpy()
                .tolist()
            )
        scale_mean, scale_p95, scale_max = _scale_stats(model)
        delta_mean_abs, delta_p95_abs = _delta_stats(model, X_train_eval)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={float(last_loss or 0.0):.6f} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} "
            f"train_ndcg@10(subset)={train_ndcg10:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"conv_w={','.join(f'{w:.4f}' for w in conv_w)} | "
            f"scale mean={scale_mean:.4f} p95={scale_p95:.4f} max={scale_max:.4f} | "
            f"delta|x| mean={delta_mean_abs:.6f} p95={delta_p95_abs:.6f} | "
            f"timing train={float(t_train.dt or 0.0):.3f}s "
            f"pred_train={float(t_pred_train.dt or 0.0):.3f}s "
            f"pred_test={float(t_pred_test.dt or 0.0):.3f}s"
        )

        current = float(train_ndcg1000)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

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
