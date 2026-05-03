# STATUS: EXPERIMENTAL
# Purpose: torch_mean_residual variant with softmax-constrained global weights.
#
# Variant name: torch_mean_residual_softmax_global
#
# Change vs baseline (benchmarks/torch_mean_residual.py):
#   - Replace learnable unconstrained global_w with learnable global_logits,
#     and compute global_w = softmax(global_logits) so weights are non-negative
#     and sum to 1 (convex combination).
#
# Everything else (delta residuals, bias, training loop, CLI) is intentionally
# kept identical for controlled benchmarking.
from __future__ import annotations

from pathlib import Path
import sys

# Allow running as a script: `uv run benchmarks/torch_mean_residual_softmax_global.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.datasets import ensemble3_keys, get_dataset_config, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.metrics import (
    load_csr,
    evaluate_model_batched,
    update_markdown_scoreboard,
)
from benchmarks.preprocessing import SparseCSRDataset, log1p_transform


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


class MeanResidualSoftmaxGlobalEnsemble(nn.Module):
    """
    Mean-like ensemble with per-label residual weights and softmax global weights.

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

        # Softmax parameterization:
        #   global_w = softmax(global_logits) is always >=0 and sums to 1.
        # Initialize logits so softmax(global_logits) == w0 as closely as possible.
        # Clamp to avoid -inf from log(0).
        w0 = torch.clamp(w0, min=1e-12)
        self.global_logits = nn.Parameter(torch.log(w0))  # (M,)

        # Per-label residual weights initialized to 0. Learnable.
        self.delta_w = nn.Parameter(torch.zeros((n_models, n_labels), dtype=torch.float32))

        # Per-label bias in logit space (helps match label base rates).
        self.bias = nn.Parameter(torch.zeros((n_labels,), dtype=torch.float32))

    def global_w(self) -> torch.Tensor:
        return torch.softmax(self.global_logits, dim=0)  # (M,)

    def effective_w(self) -> torch.Tensor:
        # (M, L)
        return self.global_w()[:, None] + self.delta_w

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
        return (self.delta_w**2).mean()

    def bias_l2(self) -> torch.Tensor:
        # Mean squared bias for scale-invariant regularization.
        return (self.bias**2).mean()


# (Removed csr_to_log1p_tensor, _Timer, and _predict_in_batches in favor of SparseCSRDataset and evaluate_model_batched)


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
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print extra debugging diagnostics each epoch: "
            "global weights, delta weight distribution, bias distribution, "
            "and output score distribution on the early-stop train subset."
        ),
    )
    args = parser.parse_args()
    dataset = str(args.dataset)
    lambda_delta = float(args.lambda_delta)
    lambda_bias = float(args.lambda_bias)
    print_delta = bool(args.print_delta)
    debug = bool(args.debug)

    # Deterministic-ish
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_mean_residual_softmax_global({','.join(ensemble_keys)})"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")
    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]

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
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in ensemble_keys]
    test_ds = SparseCSRDataset(test_preds, stack_dim=0, transform=log1p_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    cfg = get_dataset_config(dataset)
    init_global: torch.Tensor | None = None
    if cfg.ensemble3_init_weights is not None:
        init_global = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)
        if init_global.shape[0] != n_models:
            raise ValueError(
                f"ensemble3_init_weights has length {init_global.shape[0]}, but ensemble has n_models={n_models}."
            )

    model = MeanResidualSoftmaxGlobalEnsemble(
        n_models=n_models, n_labels=n_labels, init_global=init_global
    ).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )
    criterion = nn.BCEWithLogitsLoss()

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
        train_res_eval = evaluate_model_batched(model, train_eval_loader, y_train_true_eval, k_values=(1000,), device=DEVICE)
        train_ndcg1000 = train_res_eval["ndcg@1000"]

        # --- Test metrics ---
        test_metrics = evaluate_model_batched(model, test_loader, y_test_true, k_values=K_VALUES, f1_k=5, device=DEVICE)

        diag = ""
        if print_delta or debug:
            with torch.no_grad():
                delta_l2 = float(model.delta_l2().detach().cpu().item())
                mean_abs_delta_per_model = model.delta_w.detach().abs().mean(dim=1).cpu().numpy()

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

        extra = ""
        if debug:
            with torch.no_grad():
                w_global = model.global_w().detach()
                delta_w = model.delta_w.detach()
                bias = model.bias.detach()

            # Output distribution on the early-stop subset (to spot saturation / scale issues)
            # Note: model outputs logits; stats help detect exploding/vanishing scores.
            batch = next(iter(train_eval_loader))
            xb_dbg = batch[0].to(DEVICE, non_blocking=True)
            with torch.no_grad():
                subset_logits = model(xb_dbg).detach().cpu()

            w_stats = _tensor_stats_1d(w_global)
            d_stats = _tensor_stats_all(delta_w)
            b_stats = _tensor_stats_1d(bias)
            s_stats = _tensor_stats_all(subset_logits)

            extra = (
                "\n"
                "  debug:\n"
                f"    global_w: mean={w_stats['mean']:.6f} std={w_stats['std']:.6f} "
                f"min={w_stats['min']:.6f} p50={w_stats['p50']:.6f} max={w_stats['max']:.6f}\n"
                f"    delta_w:  mean={d_stats['mean']:.6e} std={d_stats['std']:.6e} "
                f"min={d_stats['min']:.6e} p50={d_stats['p50']:.6e} max={d_stats['max']:.6e}\n"
                f"    bias:     mean={b_stats['mean']:.6e} std={b_stats['std']:.6e} "
                f"min={b_stats['min']:.6e} p50={b_stats['p50']:.6e} max={b_stats['max']:.6e}\n"
                f"    scores:   mean={s_stats['mean']:.6e} std={s_stats['std']:.6e} "
                f"min={s_stats['min']:.6e} p50={s_stats['p50']:.6e} max={s_stats['max']:.6e}"
            )

        print(
            f"[lambda_delta={lambda_delta:g} lambda_bias={lambda_bias:g}] "
            f"Epoch {epoch:02d} | "
            f"loss={loss.item():.6f} (bce={loss_main.item():.6f} reg={loss_reg.item():.6f}) | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"total={time.perf_counter() - epoch_t0:.3f}s"
            f"{diag}"
            f"{extra}"
        )

        current = float(train_ndcg1000)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Compute full train metrics only at best snapshot
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
