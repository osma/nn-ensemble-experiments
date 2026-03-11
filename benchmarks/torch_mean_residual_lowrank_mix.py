# STATUS: EXPERIMENTAL
# Purpose: Combine torch_mean_residual (strong, stable logit-space baseline) with
# a *structured* cross-label correction inspired by torch_nn_split, but safer than
# a flattened MLP over (models × labels).
#
# Approach:
# - Keep the full torch_mean_residual base:
#     base_logits[b, l] = sum_m (w_global[m] + delta_w[m, l]) * x[b, m, l] + bias[l]
# - Add a low-rank label-mixing residual *only on active labels*:
#     p = sigmoid(base_logits)  (ranking-invariant monotone transform)
#     delta_active = (p_active @ U) @ V^T
#     logits[active] = base_logits[active] + alpha * delta_active
#
# Safety rails:
# - delta head is zero-initialized (V starts at 0) => no correction at init
# - bounded alpha gate: alpha = alpha_max * sigmoid(log_alpha)
# - explicit L2 penalty only on the *applied* correction: mean((alpha*delta)^2)
#
# Training:
# - BCEWithLogitsLoss on logits (no pos_weight, per request)
# - Early stopping by train subset NDCG@1000
from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_mean_residual_lowrank_mix.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

from benchmarks.datasets import ensemble3_keys, get_dataset_config, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.preprocessing import csr_to_log1p_tensor
from benchmarks.metrics import load_csr, ndcg_at_k_dense, f1_at_k_dense, update_markdown_scoreboard

DEVICE = get_device()

# Training defaults (intentionally similar to torch_mean_residual)
EPOCHS = 20
K_VALUES = (10, 1000)
PATIENCE = 2
MIN_EPOCHS = 2

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512

EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

# Base residual hyperparameters
LR = 0.003
WEIGHT_DECAY = 0.0  # rely on explicit residual penalties
LAMBDA_DELTA_L2 = 1e-2
LAMBDA_BIAS_L2 = 1e-3

# Low-rank mixing hyperparameters
MIX_RANK = 32
# Keep the mixer very small by default. After enabling learnable U init, the mix path
# can easily learn a degenerate "push everything down" solution under class imbalance.
# Use a smaller gate by default; you can still raise this via --alpha-max if needed.
ALPHA_MAX = 0.01
# Stronger penalty on the *applied* mix output (alpha * delta_active).
# This helps prevent early collapse while still allowing the mixer to contribute.
LAMBDA_MIX_OUT_L2 = 1e-3

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


def _mix_active_stats(
    model: "MeanResidualLowRankMixEnsemble", x_cpu_subset: torch.Tensor
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
            base = model.base_logits(xb)
            delta = model.mix_delta_active(base)  # (B, L_active)

            # Match the model's centering behavior used in forward():
            delta = delta - delta.mean(dim=1, keepdim=True)

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


class MeanResidualLowRankMixEnsemble(nn.Module):
    """
    torch_mean_residual base + low-rank label mixing residual on active labels.

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
        mix_rank: int,
        alpha_max: float,
    ):
        super().__init__()
        if n_models != 3:
            raise ValueError("This experimental model is intended for 3-way ensembles only")
        if active_idx.ndim != 1:
            raise ValueError("active_idx must be 1D")
        if mix_rank < 1:
            raise ValueError("mix_rank must be >= 1")

        self.n_models = int(n_models)
        self.n_labels = int(n_labels)

        self.register_buffer("active_idx", active_idx.long())
        self.n_active = int(self.active_idx.numel())

        # --- Base torch_mean_residual parameters (logit space) ---
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

        # Constrain global weights to a convex combination via softmax (matches torch_mean_residual_mlp)
        self.global_logits = nn.Parameter(torch.log(w0 + 1e-12))  # (M,)
        self.delta_w = nn.Parameter(torch.zeros((n_models, n_labels), dtype=torch.float32))  # (M, L)
        self.bias = nn.Parameter(torch.zeros((n_labels,), dtype=torch.float32))  # (L,)

        # --- Low-rank mixing residual (active labels only) ---
        self.mix_rank = int(mix_rank)
        self.log_alpha = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))
        self.alpha_max = float(alpha_max)

        # delta_active = (p_active @ U) @ V^T
        # U: (L_active, r), V: (L_active, r)
        self.U = nn.Parameter(torch.zeros((self.n_active, self.mix_rank), dtype=torch.float32))
        self.V = nn.Parameter(torch.zeros((self.n_active, self.mix_rank), dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Base params: delta_w and bias already start at 0 by construction.

        # Mixing: start with *effective* no correction, but allow gradients to flow.
        # If both U and V are initialized to 0, then:
        #   h = p_active @ U == 0 and delta = h @ V^T == 0,
        # and both U and V receive zero gradients (dead path).
        #
        # Instead:
        # - Initialize U to small random values so h is non-zero (enables gradients into V).
        # - Keep V at 0 so delta == 0 at init (strict residual / do-no-harm start).
        # - Also start alpha smaller to avoid the mix path immediately dominating logits.
        nn.init.normal_(self.U, mean=0.0, std=1e-3)
        nn.init.zeros_(self.V)

        with torch.no_grad():
            self.log_alpha.fill_(-5.0)

    def alpha(self) -> torch.Tensor:
        return float(self.alpha_max) * torch.sigmoid(self.log_alpha)

    def global_w(self) -> torch.Tensor:
        return torch.softmax(self.global_logits, dim=0)  # (M,)

    def effective_w(self) -> torch.Tensor:
        return self.global_w()[:, None] + self.delta_w  # (M, L)

    def base_logits(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.effective_w()
        return (x * w_eff.unsqueeze(0)).sum(dim=1) + self.bias  # (B, L)

    def mix_delta_active(self, base_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute low-rank mixing residual for active labels from base logits.

        base_logits: (B, L)
        returns: (B, L_active)
        """
        if self.n_active == 0:
            return base_logits.new_zeros((base_logits.shape[0], 0))

        p = torch.sigmoid(base_logits)  # (B, L)
        p_active = p.index_select(dim=1, index=self.active_idx)  # (B, L_active)

        # (B, r)
        h = p_active @ self.U
        # (B, L_active)
        return h @ self.V.transpose(0, 1)

    def forward(self, x: torch.Tensor, *, return_delta_active: bool = False):
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, M, L), got {tuple(x.shape)}")
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected x with (M={self.n_models}, L={self.n_labels}), got {tuple(x.shape)}"
            )

        base = self.base_logits(x)

        if self.n_active == 0:
            if return_delta_active:
                return base, base.new_zeros((base.shape[0], 0))
            return base

        delta_active = self.mix_delta_active(base)  # (B, L_active)

        # Prevent the mixer from learning a degenerate global shift (e.g. "push everything down")
        # by forcing the per-example mean correction over active labels to be exactly 0.
        # This makes the mixer purely redistributive over active labels.
        delta_active = delta_active - delta_active.mean(dim=1, keepdim=True)

        alpha = self.alpha()

        logits = base.clone()
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
        "--lambda-mix-out",
        type=float,
        default=LAMBDA_MIX_OUT_L2,
        help="L2 penalty strength for mean((alpha*delta_active)^2) (keeps mixing corrections small)",
    )
    parser.add_argument(
        "--mix-weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight_decay applied only to mixer parameters (U,V)",
    )
    parser.add_argument(
        "--mix-rank",
        type=int,
        default=MIX_RANK,
        help="Low-rank mixing rank (capacity control)",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=ALPHA_MAX,
        help="Maximum mixing gate alpha (alpha = alpha_max * sigmoid(log_alpha))",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostics (mix output magnitude and parameter norms) each epoch",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug diagnostics (debug is enabled by default)",
    )
    args = parser.parse_args()

    dataset = str(args.dataset)
    lambda_delta = float(args.lambda_delta)
    lambda_bias = float(args.lambda_bias)
    lambda_mix_out = float(args.lambda_mix_out)
    mix_rank = int(args.mix_rank)
    alpha_max = float(args.alpha_max)
    mix_weight_decay = float(args.mix_weight_decay)

    # Default debug ON (can be disabled with --no-debug).
    debug = (not bool(args.no_debug)) or bool(args.debug)

    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_mean_residual_lowrank_mix({','.join(ensemble_keys)})"
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

    X_train = torch.stack([csr_to_log1p_tensor(p) for p in train_preds], dim=1)
    Y_train = torch.from_numpy(y_train_true.toarray()).float()

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

    model = MeanResidualLowRankMixEnsemble(
        n_models=n_models,
        n_labels=n_labels,
        active_idx=active_idx,
        init_global=init_global,
        mix_rank=mix_rank,
        alpha_max=alpha_max,
    ).to(DEVICE)

    # Use parameter groups so we can apply weight decay only to the mixing parameters.
    # This helps curb fast growth of U/V without changing the base residual dynamics.
    optimizer = optim.AdamW(
        [
            {
                "params": [model.global_logits, model.delta_w, model.bias, model.log_alpha],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [model.U, model.V],
                "weight_decay": mix_weight_decay,
            },
        ],
        lr=LR,
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
            last_loss: float | None = None
            last_loss_bce: float | None = None
            last_loss_reg: float | None = None
            last_loss_reg_mix: float | None = None

            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits, delta_active = model(xb, return_delta_active=True)

                loss_main = criterion(logits, yb)
                loss_reg_delta = lambda_delta * model.delta_l2()
                loss_reg_bias = lambda_bias * model.bias_l2()

                if delta_active.numel() == 0:
                    loss_reg_mix = logits.new_tensor(0.0)
                else:
                    alpha = model.alpha()
                    loss_reg_mix = lambda_mix_out * (alpha * delta_active).pow(2).mean()

                loss_reg = loss_reg_delta + loss_reg_bias + loss_reg_mix
                loss = loss_main + loss_reg

                loss.backward()
                optimizer.step()

                last_loss = float(loss.item())
                last_loss_bce = float(loss_main.item())
                last_loss_reg = float(loss_reg.item())
                last_loss_reg_mix = float(loss_reg_mix.item())

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

        dbg = ""
        if debug:
            mix_mean_abs, mix_p95_abs = _mix_active_stats(model, X_train_eval)
            with torch.no_grad():
                u_l2 = float((model.U ** 2).mean().detach().cpu().item()) if model.n_active > 0 else 0.0
                v_l2 = float((model.V ** 2).mean().detach().cpu().item()) if model.n_active > 0 else 0.0
            dbg = (
                f" mix|alpha*delta| mean={mix_mean_abs:.6f} p95={mix_p95_abs:.6f}"
                f" U_l2={u_l2:.3e} V_l2={v_l2:.3e}"
            )

        with torch.no_grad():
            w_global = model.global_w().detach().cpu().numpy().tolist()
            alpha_val = float(model.alpha().detach().cpu().item())

        print(
            f"[lambda_delta={lambda_delta:g} lambda_bias={lambda_bias:g} "
            f"lambda_mix_out={lambda_mix_out:g} mix_rank={mix_rank} alpha_max={alpha_max:g}] "
            f"Epoch {epoch:02d} | "
            f"loss={float(last_loss or 0.0):.6f} "
            f"(bce={float(last_loss_bce or 0.0):.6f} reg={float(last_loss_reg or 0.0):.6f} "
            f"reg_mix={float(last_loss_reg_mix or 0.0):.6f}) | "
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
