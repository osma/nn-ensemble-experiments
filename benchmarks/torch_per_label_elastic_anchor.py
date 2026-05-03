# STATUS: EXPERIMENTAL
# Purpose: Per-label ensemble combining the two top-performing patterns:
#   1. Softmax-constrained global weights + L2 on per-label residuals
#      (from torch_per_label_softmax_global, overall #1)
#   2. Bias decomposition: global scalar + per-label delta
#      (from torch_per_label_bias_global_plus_delta, #1 yso-en, #1 F1@5)
#
# This specific combination has not been tested before. Previous version
# (v1) also included L1 on delta and an L2 anchor to init, which the debug
# output showed were over-regularizing w_delta (keeping it in ~1e-3 range
# throughout training). Removing those lets w_delta contribute meaningfully.
#
# Architecture:
#   w_global = softmax(g_raw)                         # (M,)
#   w_eff[m, l] = w_global[m] + w_delta[m, l]         # (M, L)
#   logits[b, l] = sum_m w_eff[m,l] * x[b,m,l] + bias_global + bias_delta[l]
#
# Loss:
#   BCE + λ_l2 * mean(w_delta²) + λ_bias * mean(bias_delta²)
#
# Debug output: always-on, prints loss decomposition, weight stats, gradient norms.
from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label_elastic_anchor.py`
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


class PerLabelElasticAnchorEnsemble(nn.Module):
    """
    Per-label ensemble with softmax-constrained global weights, elastic net
    regularized per-label residuals, L2-anchored globals, and decomposed bias.

    Parameters:
      - g_raw[m]        (M,)  unconstrained global logits → softmax → w_global
      - w_delta[m, l]   (M, L)  per-label residual weights, init 0
      - bias_global     scalar  shared bias term
      - bias_delta[l]   (L,)    per-label bias residual, init 0

    Effective weights:
      w_eff[m, l] = softmax(g_raw)[m] + w_delta[m, l]

    Forward:
      logits[b, l] = Σ_m w_eff[m, l] * x[b, m, l] + bias_global + bias_delta[l]

    Notes:
    - Inputs are expected to already be log1p-preprocessed (non-negative).
    - Returns raw logits; intended for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        *,
        n_models: int,
        n_labels: int,
        init_global: torch.Tensor | None = None,  # (M,)
    ) -> None:
        super().__init__()
        self.n_models = int(n_models)
        self.n_labels = int(n_labels)

        if init_global is None:
            g = torch.full((self.n_models,), 1.0 / self.n_models, dtype=torch.float32)
        else:
            if init_global.ndim != 1 or init_global.shape[0] != self.n_models:
                raise ValueError(
                    f"init_global must have shape ({self.n_models},), got {tuple(init_global.shape)}"
                )
            g = init_global.to(dtype=torch.float32).clone()
            s = float(g.sum().item())
            if not np.isfinite(s) or s <= 0.0:
                raise ValueError("init_global must sum to a positive finite value")
            g = g / g.sum()

        # Store the init weights for the L2 anchor penalty.
        self.register_buffer("w_init", g.clone())  # (M,)

        # Parameterize global weights via logits; initialize so softmax(g_raw) ≈ g.
        self.g_raw = nn.Parameter(torch.log(torch.clamp(g, min=1e-12)))

        # Per-label residuals around global (init at 0 = "trust the global prior").
        self.w_delta = nn.Parameter(torch.zeros(self.n_models, self.n_labels))

        # Bias decomposition: global scalar + per-label residual.
        self.bias_global = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.bias_delta = nn.Parameter(torch.zeros(self.n_labels))

    def global_w(self) -> torch.Tensor:
        """Softmax-constrained global mixture weights, shape (M,)."""
        return torch.softmax(self.g_raw, dim=0)

    def effective_w(self) -> torch.Tensor:
        """Per-label effective weights, shape (M, L)."""
        return self.global_w()[:, None] + self.w_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input of shape (batch, n_models, n_labels), got {x.shape}"
            )
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected input with n_models={self.n_models}, "
                f"n_labels={self.n_labels}, got {x.shape}"
            )

        w_eff = self.effective_w().unsqueeze(0)  # (1, M, L)
        out = (x * w_eff).sum(dim=1) + self.bias_global + self.bias_delta
        return out

    # --- Regularization terms ---

    def delta_l2(self) -> torch.Tensor:
        """Mean squared per-label residual weight."""
        return (self.w_delta ** 2).mean()

    def delta_l1(self) -> torch.Tensor:
        """Mean absolute per-label residual weight (sparsity)."""
        return self.w_delta.abs().mean()

    def anchor_l2(self) -> torch.Tensor:
        """Squared L2 distance between current global weights and init."""
        return ((self.global_w() - self.w_init) ** 2).sum()

    def bias_delta_l2(self) -> torch.Tensor:
        """Mean squared per-label bias residual."""
        return (self.bias_delta ** 2).mean()


# ============================
# Training / evaluation script
# ============================

DEVICE = get_device()
EPOCHS = 20
K_VALUES = (10, 1000)

PATIENCE = 2
MIN_EPOCHS = 2

EVAL_BATCH_SIZE = 512
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

# Hyperparameters (from proven best values in each source model)
BEST_LR = 0.003
BEST_WEIGHT_DECAY = 0.0
BEST_BATCH_SIZE = 256

# Reproducibility for training shuffles / init
TRAIN_SEED = 0

# Regularization strengths
# v1 lesson: anchor + L1 over-constrained w_delta → kept it near zero throughout.
# v2: only L2 on delta (same as torch_per_label_softmax_global) + bias shrinkage.
LAMBDA_DELTA_L2 = 1e-3   # same as torch_per_label_softmax_global (proven)
LAMBDA_DELTA_L1 = 0.0    # removed: was suppressing w_delta too much
LAMBDA_ANCHOR = 0.0      # removed: was freezing global weights
LAMBDA_BIAS_L2 = 1e-3    # shrinkage on per-label bias delta


def _grad_norm(params: list[torch.Tensor]) -> float:
    """Compute total L2 norm of gradients across a list of parameters."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item() ** 2)
    return float(total ** 0.5)


def train_and_evaluate(
    *,
    dataset: str,
    ensemble_keys: tuple[str, str, str],
    lr: float,
    weight_decay: float,
    batch_size: int,
    lambda_delta_l2: float,
    lambda_bias_l2: float,
    train_loader: torch.utils.data.DataLoader,
    train_eval_loader: torch.utils.data.DataLoader,
    y_train_true_eval: csr_matrix,
    test_loader: torch.utils.data.DataLoader,
    y_test_true: csr_matrix,
    full_train_loader: torch.utils.data.DataLoader,
    y_train_true: csr_matrix,
) -> dict[str, object]:
    """
    Train a model with given hyperparameters and return the best snapshot
    selected by TRAIN subset NDCG@1000 (early stopping metric).

    Prints comprehensive debug diagnostics every epoch.
    v2 changes vs v1: removed L1-on-delta and anchor penalties (were over-regularizing).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    # Deterministic-ish
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    n_models = len(ensemble_keys)
    n_labels = y_train_true.shape[1]

    # Dataset-specific init weights
    init_global: torch.Tensor | None = None
    cfg = get_dataset_config(dataset)
    if cfg.ensemble3 != ensemble_keys:
        raise ValueError(
            "Internal error: ensemble_keys does not match dataset config "
            f"(cfg.ensemble3={cfg.ensemble3}, ensemble_keys={ensemble_keys})"
        )
    if cfg.ensemble3_init_weights is not None:
        init_global = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)
        if init_global.shape[0] != n_models:
            raise ValueError(
                f"ensemble3_init_weights has length {init_global.shape[0]}, "
                f"but ensemble has n_models={n_models}."
            )

    model = PerLabelElasticAnchorEnsemble(
        n_models=n_models,
        n_labels=n_labels,
        init_global=init_global,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
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

    # Print model summary
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[DEBUG] Model parameters: {n_params:,} total")
    print(f"[DEBUG]   g_raw: {tuple(model.g_raw.shape)}")
    print(f"[DEBUG]   w_delta: {tuple(model.w_delta.shape)}")
    print(f"[DEBUG]   bias_global: scalar")
    print(f"[DEBUG]   bias_delta: {tuple(model.bias_delta.shape)}")
    print(f"[DEBUG] Init global weights (w_init): {model.w_init.cpu().numpy().tolist()}")
    print(f"[DEBUG] Regularization: "
          f"λ_delta_l2={lambda_delta_l2:g} λ_bias_l2={lambda_bias_l2:g} "
          f"(L1-delta and anchor removed in v2)")
    print()

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.perf_counter()

        model.train()
        last_loss_bce = 0.0
        last_loss_total = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_bce = criterion(logits, yb)
            loss_l2_delta = lambda_delta_l2 * model.delta_l2()
            loss_bias = lambda_bias_l2 * model.bias_delta_l2()
            loss = loss_bce + loss_l2_delta + loss_bias
            loss.backward()
            optimizer.step()

            last_loss_bce = float(loss_bce.item())
            last_loss_total = float(loss.item())
            n_batches += 1

        # Compute final regularization values (after last step) for debug
        with torch.no_grad():
            dbg_loss_l2_delta = float((lambda_delta_l2 * model.delta_l2()).item())
            dbg_loss_bias = float((lambda_bias_l2 * model.bias_delta_l2()).item())

            # Global weights
            gw = model.global_w().cpu().numpy()

            # Delta stats per model
            wd = model.w_delta.detach()
            delta_mean_abs_per_model = wd.abs().mean(dim=1).cpu().numpy()
            delta_max_abs = float(wd.abs().max().item())
            delta_frac_nonzero = float((wd.abs() > 1e-6).float().mean().item())

            # Bias stats
            bg = float(model.bias_global.item())
            bd_mean_abs = float(model.bias_delta.abs().mean().item())
            bd_max_abs = float(model.bias_delta.abs().max().item())

        # Gradient norms — compute fresh with one batch
        grad_norms_str = ""
        try:
            model.train()
            xb_dbg, yb_dbg = next(iter(train_loader))
            xb_dbg = xb_dbg.to(DEVICE, non_blocking=True)
            yb_dbg = yb_dbg.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits_dbg = model(xb_dbg)
            loss_dbg = (criterion(logits_dbg, yb_dbg)
                        + lambda_delta_l2 * model.delta_l2()
                        + lambda_bias_l2 * model.bias_delta_l2())
            loss_dbg.backward()

            gn_global = _grad_norm([model.g_raw])
            gn_delta = _grad_norm([model.w_delta])
            gn_bias_g = _grad_norm([model.bias_global])
            gn_bias_d = _grad_norm([model.bias_delta])
            grad_norms_str = (
                f"grad_norm: g_raw={gn_global:.4e} w_delta={gn_delta:.4e} "
                f"bias_g={gn_bias_g:.4e} bias_d={gn_bias_d:.4e}"
            )
            optimizer.zero_grad(set_to_none=True)  # clean up
        except Exception as e:
            grad_norms_str = f"grad_norm: ERROR ({e})"

        # --- Train evaluation for early stopping (subset only) ---
        train_res_eval = evaluate_model_batched(
            model, train_eval_loader, y_train_true_eval, k_values=(1000,), device=DEVICE
        )
        train_ndcg1000 = train_res_eval["ndcg@1000"]

        # --- Test evaluation ---
        test_metrics = evaluate_model_batched(
            model, test_loader, y_test_true, k_values=K_VALUES, f1_k=5, device=DEVICE
        )

        epoch_dt = time.perf_counter() - epoch_t0

        # Print comprehensive debug output
        gw_str = ",".join(f"{w:.4f}" for w in gw)
        delta_str = ",".join(f"{d:.4e}" for d in delta_mean_abs_per_model)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={last_loss_total:.6f} "
            f"(bce={last_loss_bce:.6f} "
            f"l2_delta={dbg_loss_l2_delta:.6e} "
            f"bias_l2={dbg_loss_bias:.6e}) | "
            f"train_ndcg@1000(sub)={train_ndcg1000:.6f} | "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"total={epoch_dt:.3f}s"
        )
        print(
            f"  [DEBUG] global_w=[{gw_str}] | "
            f"delta mean|w|/model=[{delta_str}] max|w|={delta_max_abs:.4e} "
            f"frac_active={delta_frac_nonzero:.4f} | "
            f"bias_global={bg:.6f} mean|bias_delta|={bd_mean_abs:.4e} "
            f"max|bias_delta|={bd_max_abs:.4e} | "
            f"{grad_norms_str}"
        )

        current = train_ndcg1000
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Compute full train metrics only for the best epoch snapshot
            best_train_metrics_res = evaluate_model_batched(
                model, full_train_loader, y_train_true, k_values=K_VALUES, device=DEVICE
            )
            best_train_metrics = {
                k: v for k, v in best_train_metrics_res.items() if k.startswith("ndcg")
            }
            best_n_used_train = int(best_train_metrics_res["n_used"])

            best_test_metrics = test_metrics.copy()
            best_n_used_test = int(test_metrics["n_used"])

            epochs_no_improve = 0
            print(f"  [DEBUG] ★ New best at epoch {epoch} (train_ndcg@1000_sub={train_ndcg1000:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  [DEBUG] No improvement ({epochs_no_improve}/{PATIENCE})")

        if epoch >= MIN_EPOCHS and epochs_no_improve >= PATIENCE:
            print(f"  [DEBUG] Early stopping at epoch {epoch}")
            break

    assert best_state is not None
    assert best_epoch is not None
    assert best_train_metrics is not None
    assert best_test_metrics is not None
    assert best_n_used_train is not None
    assert best_n_used_test is not None

    # Load best snapshot before returning
    model.load_state_dict(best_state)

    return {
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "best_train_metrics": best_train_metrics,
        "best_test_metrics": best_test_metrics,
        "best_n_used_train": int(best_n_used_train),
        "best_n_used_test": int(best_n_used_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-label elastic anchor ensemble benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="yso-fi",
        choices=["yso-fi", "yso-en", "koko"],
        help="Dataset to benchmark",
    )
    args = parser.parse_args()
    dataset = str(args.dataset)

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_per_label_elastic_anchor({','.join(ensemble_keys)})"

    scoreboard_path = Path("SCOREBOARD.md")

    print("=" * 70)
    print(f"torch_per_label_elastic_anchor | dataset={dataset}")
    print("=" * 70)
    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]

    n_samples_train = y_train_true.shape[0]
    n_labels = int(y_train_true.shape[1])
    n_models = len(train_preds)

    print(f"[DEBUG] Dataset: {dataset} | n_samples_train={n_samples_train} "
          f"n_labels={n_labels} n_models={n_models}")
    print(f"[DEBUG] Ensemble keys: {ensemble_keys}")

    # Datasets using SparseCSRDataset
    train_ds = SparseCSRDataset(
        train_preds, y_train_true, stack_dim=0,
        transform=lambda xy: (log1p_transform(xy[0]), xy[1])
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BEST_BATCH_SIZE, shuffle=True,
        pin_memory=(DEVICE.type == "cuda")
    )

    full_train_ds = SparseCSRDataset(train_preds, stack_dim=0, transform=log1p_transform)
    full_train_loader = torch.utils.data.DataLoader(
        full_train_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False
    )

    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_samples_train)
    train_eval_idx = rng.choice(n_samples_train, size=n_eval, replace=False)
    train_eval_preds = [p[train_eval_idx] for p in train_preds]
    y_train_true_eval = y_train_true[train_eval_idx]
    train_eval_ds = SparseCSRDataset(
        train_eval_preds, stack_dim=0, transform=log1p_transform
    )
    train_eval_loader = torch.utils.data.DataLoader(
        train_eval_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False
    )

    print("Loading test data...")
    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in ensemble_keys]
    test_ds = SparseCSRDataset(test_preds, stack_dim=0, transform=log1p_transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False
    )

    print(
        f"Training | lr={BEST_LR:g} | wd={BEST_WEIGHT_DECAY:g} | bs={BEST_BATCH_SIZE} | "
        f"λ_delta_l2={LAMBDA_DELTA_L2:g} λ_bias_l2={LAMBDA_BIAS_L2:g} "
        f"(v2: no L1-delta, no anchor)"
    )
    print()

    result = train_and_evaluate(
        dataset=dataset,
        ensemble_keys=ensemble_keys,
        lr=BEST_LR,
        weight_decay=BEST_WEIGHT_DECAY,
        batch_size=BEST_BATCH_SIZE,
        lambda_delta_l2=LAMBDA_DELTA_L2,
        lambda_bias_l2=LAMBDA_BIAS_L2,
        train_loader=train_loader,
        y_train_true=y_train_true,
        train_eval_loader=train_eval_loader,
        y_train_true_eval=y_train_true_eval,
        test_loader=test_loader,
        y_test_true=y_test_true,
        full_train_loader=full_train_loader,
    )

    best_epoch = int(result["best_epoch"])
    best_metric = float(result["best_metric"])
    best_train_metrics = result["best_train_metrics"]
    best_test_metrics = result["best_test_metrics"]
    best_n_used_train = int(result["best_n_used_train"])
    best_n_used_test = int(result["best_n_used_test"])

    print("\n" + "=" * 70)
    print("Training complete")
    print("=" * 70)
    print(
        f"Best epoch={best_epoch} | "
        f"train_ndcg@1000(subset)={best_metric:.6f}"
    )
    print(
        f"Best test metrics | "
        f"ndcg@10={float(best_test_metrics['ndcg@10']):.6f} | "
        f"ndcg@1000={float(best_test_metrics['ndcg@1000']):.6f} | "
        f"f1@5={float(best_test_metrics['f1@5']):.6f}"
    )

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

    print("\nSaved result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
