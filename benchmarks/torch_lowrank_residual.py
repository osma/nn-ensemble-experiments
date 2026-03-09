# STATUS: EXPERIMENTAL
# Purpose: Single-model ensemble combining:
#   - torch_mean_residual's strong global prior + bias + shrinkage
#   - controlled cross-label coupling via low-rank per-label residual weights
#
# Form:
#   delta_w = U @ V     (U: MxR, V: RxL)
#   logits[b,l] = sum_m (w_global[m] + delta_w[m,l]) * x[b,m,l] + bias[l]
#
# Notes:
# - Inputs are sqrt-preprocessed outside the model (same convention as torch_nn).
# - Output is clamped to [0,1] and trained with BCELoss (torch_nn-like), rather than logits/BCEWithLogitsLoss.
# - Residual is initialized to zero by initializing V to zeros (U random), similar to torch_nn's
#   "delta starts at 0" behavior.
# - Early stopping is by TRAIN subset NDCG@1000 (no test leakage), matching other scripts.
from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_lowrank_residual.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from benchmarks.datasets import ensemble3_keys, get_dataset_config, pred_path, truth_path
from benchmarks.device import get_device
from scipy.sparse import csr_matrix

from benchmarks.preprocessing import csr_to_log1p_tensor
from benchmarks.metrics import (
    load_csr,
    ndcg_at_k_dense,
    f1_at_k_dense,
    update_markdown_scoreboard,
)

DEVICE = get_device()

# Training defaults (intentionally similar to torch_mean_residual / torch_per_label)
EPOCHS = 20
K_VALUES = (10, 1000)
PATIENCE = 2
MIN_EPOCHS = 2

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512

EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

# Optimizer hyperparams (match best-known range)
LR = 0.003
WEIGHT_DECAY = 0.0  # rely on explicit penalties

# Regularization strengths (tunable via CLI)
DEFAULT_LAMBDA_UV_L2 = 1e-2
DEFAULT_LAMBDA_BIAS_L2 = 1e-3


def _tensor_stats_1d(t: torch.Tensor) -> dict[str, float]:
    """
    Lightweight numeric stats for debugging. Expects a 1D tensor on any device.
    Returns Python floats.
    """
    if t.ndim != 1:
        raise ValueError(f"_tensor_stats_1d expected 1D tensor, got shape {tuple(t.shape)}")
    if t.numel() == 0:
        return {"n": 0.0}

    x = t.detach().to(dtype=torch.float32)
    return {
        "n": float(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "p01": float(torch.quantile(x, 0.01).item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "p99": float(torch.quantile(x, 0.99).item()),
        "max": float(x.max().item()),
    }


def _tensor_stats_all(t: torch.Tensor) -> dict[str, float]:
    """
    Stats for any tensor (flattens). Returns Python floats.
    """
    return _tensor_stats_1d(t.detach().reshape(-1))


def _fraction_at_bounds01(x: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    """
    For probability-like tensors in [0,1], compute how much mass is near 0 or 1.
    Helps detect clamp/saturation.
    """
    if x.numel() == 0:
        return {"frac_le_eps": 0.0, "frac_ge_1m_eps": 0.0}
    z = x.detach()
    return {
        "frac_le_eps": float((z <= eps).to(dtype=torch.float32).mean().item()),
        "frac_ge_1m_eps": float((z >= (1.0 - eps)).to(dtype=torch.float32).mean().item()),
    }


def csr_to_sqrt_tensor(csr: csr_matrix) -> torch.Tensor:
    """
    Convert CSR predictions to a dense torch tensor with fixed sqrt preprocessing:
        sqrt(clamp(x, 0))

    This matches torch_nn's input convention.
    """
    x = torch.from_numpy(csr.toarray()).float()
    return torch.sqrt(torch.clamp(x, min=0.0))

# Model capacity
DEFAULT_RANK = 32

# Reproducibility
TRAIN_SEED = 0


class LowRankResidualEnsemble(nn.Module):
    """
    Low-rank residual ensemble with global weights + per-label bias.

    Input:
        x: (batch, M, L) log1p-preprocessed scores (non-negative)
    Output:
        logits: (batch, L) raw logits
    """

    def __init__(
        self,
        *,
        n_models: int,
        n_labels: int,
        rank: int,
        init_global: torch.Tensor | None,
    ):
        super().__init__()
        if n_models < 1:
            raise ValueError("n_models must be positive")
        if n_labels < 1:
            raise ValueError("n_labels must be positive")
        if rank < 1:
            raise ValueError("rank must be positive")

        self.n_models = int(n_models)
        self.n_labels = int(n_labels)
        self.rank = int(rank)

        # Global per-model weights.
        if init_global is None:
            w0 = torch.full((self.n_models,), 1.0 / float(self.n_models), dtype=torch.float32)
        else:
            if init_global.ndim != 1 or init_global.shape[0] != self.n_models:
                raise ValueError(
                    f"init_global must have shape ({self.n_models},), got {tuple(init_global.shape)}"
                )
            w0 = init_global.to(dtype=torch.float32).clone()
            s = float(w0.sum().item())
            if not np.isfinite(s) or s <= 0.0:
                raise ValueError("init_global must sum to a positive finite value")
            w0 = w0 / w0.sum()

        self.global_w = nn.Parameter(w0)  # (M,)

        # Low-rank residual factors.
        # Initialize V to zeros so delta_w starts at exactly 0 (like torch_nn delta init).
        self.U = nn.Parameter(torch.empty((self.n_models, self.rank), dtype=torch.float32))
        self.V = nn.Parameter(torch.zeros((self.rank, self.n_labels), dtype=torch.float32))
        nn.init.normal_(self.U, mean=0.0, std=0.1)

        # Per-label bias.
        self.bias = nn.Parameter(torch.zeros((self.n_labels,), dtype=torch.float32))

    def delta_w(self) -> torch.Tensor:
        # (M, L)
        return self.U @ self.V

    def effective_w(self) -> torch.Tensor:
        # (M, L)
        return self.global_w[:, None] + self.delta_w()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, M, L), got {tuple(x.shape)}")
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected x with (M={self.n_models}, L={self.n_labels}), got {tuple(x.shape)}"
            )

        w_eff = self.effective_w()  # (M, L)
        out = (x * w_eff.unsqueeze(0)).sum(dim=1) + self.bias  # (B, L)
        return torch.clamp(out, min=0.0, max=1.0)

    def uv_l2(self) -> torch.Tensor:
        # Mean squared norm for scale-invariant regularization.
        return (self.U.pow(2).mean() + self.V.pow(2).mean()) / 2.0

    def bias_l2(self) -> torch.Tensor:
        return self.bias.pow(2).mean()


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
        "--rank",
        type=int,
        default=DEFAULT_RANK,
        help="Low-rank dimension R for delta_w = U@V.",
    )
    parser.add_argument(
        "--lambda-uv",
        type=float,
        default=DEFAULT_LAMBDA_UV_L2,
        help="L2 shrinkage strength for low-rank residual factors U,V.",
    )
    parser.add_argument(
        "--lambda-bias",
        type=float,
        default=DEFAULT_LAMBDA_BIAS_L2,
        help="L2 shrinkage strength for per-label bias.",
    )
    parser.add_argument(
        "--print-delta",
        action="store_true",
        help="Print simple diagnostics about delta_w magnitude each epoch.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print extra debugging diagnostics each epoch: parameter stats, "
            "output distribution and saturation, and gradient norms (to catch clamp/vanishing)."
        ),
    )
    args = parser.parse_args()

    dataset = str(args.dataset)
    rank = int(args.rank)
    lambda_uv = float(args.lambda_uv)
    lambda_bias = float(args.lambda_bias)
    print_delta = bool(args.print_delta)
    debug = bool(args.debug)

    if rank < 1:
        raise ValueError("rank must be positive")
    if lambda_uv < 0:
        raise ValueError("lambda_uv must be nonnegative")
    if lambda_bias < 0:
        raise ValueError("lambda_bias must be nonnegative")

    # Deterministic-ish
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_lowrank_residual({','.join(ensemble_keys)})"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]
    X_train = torch.stack([csr_to_sqrt_tensor(p) for p in train_preds], dim=1)

    Y_train = torch.from_numpy(y_train_true.toarray()).float()

    if debug:
        # Basic data sanity checks
        with torch.no_grad():
            x_stats = _tensor_stats_all(X_train)
            y_stats = _tensor_stats_all(Y_train)
            y_pos_rate = float((Y_train > 0.0).to(dtype=torch.float32).mean().item())
        print(
            "debug: data\n"
            f"  X_train stats: mean={x_stats['mean']:.6e} std={x_stats['std']:.6e} "
            f"min={x_stats['min']:.6e} p50={x_stats['p50']:.6e} p99={x_stats['p99']:.6e} max={x_stats['max']:.6e}\n"
            f"  Y_train stats: mean={y_stats['mean']:.6e} std={y_stats['std']:.6e} "
            f"min={y_stats['min']:.6e} p50={y_stats['p50']:.6e} p99={y_stats['p99']:.6e} max={y_stats['max']:.6e}\n"
            f"  Y_train positive rate: {y_pos_rate:.6e}"
        )

    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_train = int(X_train.shape[0])
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_train)
    train_eval_idx = rng.choice(n_train, size=n_eval, replace=False)
    X_train_eval = X_train[train_eval_idx]
    y_train_true_eval = y_train_true[train_eval_idx]

    print("Loading test data...")
    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in ensemble_keys]
    X_test = torch.stack([csr_to_sqrt_tensor(p) for p in test_preds], dim=1)

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

    model = LowRankResidualEnsemble(
        n_models=n_models,
        n_labels=n_labels,
        rank=rank,
        init_global=init_global,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )
    criterion = nn.BCELoss()

    if debug:
        # Confirm we start with an exactly-zero residual (due to V=0 init).
        with torch.no_grad():
            dw = model.delta_w().detach()
            dw_abs_mean = float(dw.abs().mean().cpu().item())
            dw_abs_max = float(dw.abs().max().cpu().item())
        print(
            "debug: init\n"
            f"  delta_w abs mean={dw_abs_mean:.6e} abs max={dw_abs_max:.6e} (expect ~0)\n"
            f"  global_w={model.global_w.detach().cpu().numpy()}"
        )

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
        epoch_t0 = time.perf_counter()

        model.train()
        with _Timer() as t_train:
            last_batch_debug: dict[str, float] | None = None

            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # Forward
                probs = model(xb)

                # Sanity: for BCELoss we must be in [0,1]
                if debug:
                    if torch.any(probs < 0.0) or torch.any(probs > 1.0) or torch.any(torch.isnan(probs)):
                        raise RuntimeError("Model output contains values outside [0,1] or NaNs")

                loss_main = criterion(probs, yb)
                loss_reg_uv = lambda_uv * model.uv_l2()
                loss_reg_bias = lambda_bias * model.bias_l2()
                loss_reg = loss_reg_uv + loss_reg_bias
                loss = loss_main + loss_reg

                loss.backward()

                # Capture gradient norms on the last minibatch (cheap, catches vanishing/exploding).
                if debug:
                    def _grad_norm(p: torch.Tensor | None) -> float:
                        if p is None or p.grad is None:
                            return 0.0
                        return float(p.grad.detach().norm().cpu().item())

                    last_batch_debug = {
                        "loss_main": float(loss_main.detach().cpu().item()),
                        "loss_reg": float(loss_reg.detach().cpu().item()),
                        "grad_global_w": _grad_norm(model.global_w),
                        "grad_U": _grad_norm(model.U),
                        "grad_V": _grad_norm(model.V),
                        "grad_bias": _grad_norm(model.bias),
                    }

                optimizer.step()

        # --- Early stop metric: train subset NDCG@1000 ---
        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)
        train_ndcg1000, _n_used_train_eval = ndcg_at_k_dense(
            y_train_true_eval, train_scores_eval, k=1000
        )

        # --- Test metrics ---
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
        if print_delta or debug:
            with torch.no_grad():
                # mean |delta_w| per model (M,)
                mean_abs_delta = model.delta_w().detach().abs().mean(dim=1).cpu().numpy()
                uv_l2 = float(model.uv_l2().detach().cpu().item())
                bias_l2 = float(model.bias_l2().detach().cpu().item())
                diag = (
                    " | "
                    f"uv_l2={uv_l2:.6e} "
                    f"mean_abs_delta=[{mean_abs_delta[0]:.3e},{mean_abs_delta[1]:.3e},{mean_abs_delta[2]:.3e}] "
                    f"bias_l2={bias_l2:.6e}"
                )

        extra = ""
        if debug:
            with torch.no_grad():
                # Parameter distributions
                gw_stats = _tensor_stats_1d(model.global_w.detach())
                u_stats = _tensor_stats_all(model.U.detach())
                v_stats = _tensor_stats_all(model.V.detach())
                b_stats = _tensor_stats_1d(model.bias.detach())

                # Output distribution & saturation on the early-stop subset
                subset_probs = train_scores_eval.detach()
                s_stats = _tensor_stats_all(subset_probs)
                sat = _fraction_at_bounds01(subset_probs, eps=1e-8)

                # Track the *pre-clamp* linear output on a tiny batch to catch clamp saturation.
                # We reconstruct: out_lin = (x * w_eff).sum + bias, then clamp.
                xb0 = X_train_eval[: min(8, int(X_train_eval.shape[0]))].to(DEVICE)
                w_eff = model.effective_w()  # (M,L) on DEVICE
                out_lin = (xb0 * w_eff.unsqueeze(0)).sum(dim=1) + model.bias  # (B,L)
                out_lin_stats = _tensor_stats_all(out_lin)
                out_lin_sig = torch.sigmoid(out_lin)
                out_sig_stats = _tensor_stats_all(out_lin_sig)
                out_clamped = torch.clamp(out_lin, min=0.0, max=1.0)
                clamp_sat = _fraction_at_bounds01(out_clamped, eps=1e-8)

            # Include last minibatch grad norms if available.
            grad_line = ""
            if last_batch_debug is not None:
                grad_line = (
                    "\n"
                    f"    grads(last batch): "
                    f"||global_w||={last_batch_debug['grad_global_w']:.3e} "
                    f"||U||={last_batch_debug['grad_U']:.3e} "
                    f"||V||={last_batch_debug['grad_V']:.3e} "
                    f"||bias||={last_batch_debug['grad_bias']:.3e} "
                    f"(loss_main={last_batch_debug['loss_main']:.6f} loss_reg={last_batch_debug['loss_reg']:.6f})"
                )

            extra = (
                "\n"
                "  debug:\n"
                f"    global_w: mean={gw_stats['mean']:.6f} std={gw_stats['std']:.6f} "
                f"min={gw_stats['min']:.6f} p50={gw_stats['p50']:.6f} max={gw_stats['max']:.6f}\n"
                f"    U:        mean={u_stats['mean']:.6e} std={u_stats['std']:.6e} "
                f"min={u_stats['min']:.6e} p50={u_stats['p50']:.6e} max={u_stats['max']:.6e}\n"
                f"    V:        mean={v_stats['mean']:.6e} std={v_stats['std']:.6e} "
                f"min={v_stats['min']:.6e} p50={v_stats['p50']:.6e} max={v_stats['max']:.6e}\n"
                f"    bias:     mean={b_stats['mean']:.6e} std={b_stats['std']:.6e} "
                f"min={b_stats['min']:.6e} p50={b_stats['p50']:.6e} max={b_stats['max']:.6e}\n"
                f"    probs(subset): mean={s_stats['mean']:.6e} std={s_stats['std']:.6e} "
                f"min={s_stats['min']:.6e} p50={s_stats['p50']:.6e} p99={s_stats['p99']:.6e} max={s_stats['max']:.6e} "
                f"sat<=eps={sat['frac_le_eps']:.3f} sat>=1-eps={sat['frac_ge_1m_eps']:.3f}\n"
                f"    out_lin(tiny): mean={out_lin_stats['mean']:.6e} std={out_lin_stats['std']:.6e} "
                f"min={out_lin_stats['min']:.6e} p50={out_lin_stats['p50']:.6e} p99={out_lin_stats['p99']:.6e} max={out_lin_stats['max']:.6e}\n"
                f"    sigmoid(out_lin): mean={out_sig_stats['mean']:.6e} std={out_sig_stats['std']:.6e} "
                f"min={out_sig_stats['min']:.6e} p50={out_sig_stats['p50']:.6e} p99={out_sig_stats['p99']:.6e} max={out_sig_stats['max']:.6e}\n"
                f"    clamp(out_lin): sat<=eps={clamp_sat['frac_le_eps']:.3f} sat>=1-eps={clamp_sat['frac_ge_1m_eps']:.3f}"
                f"{grad_line}"
            )

        epoch_dt = time.perf_counter() - epoch_t0
        print(
            f"[rank={rank} lambda_uv={lambda_uv:g} lambda_bias={lambda_bias:g}] "
            f"Epoch {epoch:02d} | "
            f"loss={loss.item():.6f} (bce={loss_main.item():.6f} reg={loss_reg.item():.6f}) | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"timing train={float(t_train.dt or 0.0):.3f}s "
            f"pred_train={float(t_pred_train.dt or 0.0):.3f}s "
            f"pred_test={float(t_pred_test.dt or 0.0):.3f}s "
            f"total={epoch_dt:.3f}s"
            f"{diag}"
            f"{extra}"
        )

        current = float(train_ndcg1000)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Full train metrics at best epoch
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
