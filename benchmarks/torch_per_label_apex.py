# STATUS: EXPERIMENTAL
# Purpose: Per-label ensemble (Apex) combining:
#   1. Softmax-constrained global weights (w_global)
#   2. Per-label scale multipliers (scale[l])
#   3. Per-label residual weights (w_delta[m, l])
#   4. Bias decomposition (bias_global + bias_delta[l])
#   5. Active-label masking: residuals/scales only for labels seen in train data.
#
# v1: Initial implementation.
#
# Architecture:
#   w_global = softmax(g_raw)                                # (M,)
#   For active labels:
#     w_eff[m, l] = scale[l] * w_global[m] + w_delta[m, l]   # (M, L_active)
#     logits[b, l] = sum_m(w_eff[m, l] * x[b, m, l]) + bias_global + bias_delta[l]
#   For inactive labels:
#     logits[b, l] = sum_m(w_global[m] * x[b, m, l]) + bias_global
#
# Preprocessing: log1p(clamp(x, 0))
# Loss: BCEWithLogitsLoss
# Debug output: Comprehensive diagnostics every epoch.

from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label_apex.py`
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

DEVICE = get_device()

# Training defaults
EPOCHS = 30
K_VALUES = (10, 1000)
PATIENCE = 3
MIN_EPOCHS = 2

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

# Reproducibility
TRAIN_SEED = 0

# Best hyperparameters
BEST_LR = 0.003
BEST_WEIGHT_DECAY = 0.0

# Per-label scaling bounds
SCALE_MIN = 0.1
SCALE_MAX = 10.0

# Regularization strengths (L2) - Stronger values to prevent collapse
LAMBDA_DELTA_L2 = 1e-2
LAMBDA_BIAS_L2 = 1e-3


def _bounded_scale_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """Map unconstrained raw values to [SCALE_MIN, SCALE_MAX]."""
    lo = float(np.log(SCALE_MIN))
    hi = float(np.log(SCALE_MAX))
    return torch.exp(torch.clamp(raw, min=lo, max=hi))


def _label_active_mask(y_train_true: csr_matrix, train_preds: list[csr_matrix]) -> np.ndarray:
    """Identify labels that appear in truth or any prediction in training set."""
    n_labels = int(y_train_true.shape[1])
    active = np.zeros(n_labels, dtype=bool)
    if y_train_true.nnz:
        active[np.unique(y_train_true.indices)] = True
    for p in train_preds:
        if p.nnz:
            active[np.unique(p.indices)] = True
    return active


def _grad_norm(params: list[torch.Tensor]) -> float:
    """Compute total L2 norm of gradients."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item() ** 2)
    return float(total ** 0.5)


class PerLabelApexEnsemble(nn.Module):
    def __init__(
        self,
        *,
        n_models: int,
        n_labels: int,
        active_idx: torch.Tensor,
        init_global: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_models = int(n_models)
        self.n_labels = int(n_labels)
        self.register_buffer("active_idx", active_idx.long())
        self.n_active = int(active_idx.numel())

        # Global weights mixture prior
        if init_global is None:
            g = torch.full((self.n_models,), 1.0 / self.n_models, dtype=torch.float32)
        else:
            g = init_global.to(dtype=torch.float32).clone()
            g = g / g.sum().clamp(min=1e-12)
        
        # Parameterize global weights via logits; init so softmax(g_raw) ≈ g.
        self.g_raw = nn.Parameter(torch.log(torch.clamp(g, min=1e-12)))

        # Per-label residuals (Full)
        # delta_w: (M, L) initialized at 0
        self.delta_w = nn.Parameter(torch.zeros(self.n_models, self.n_labels))
        # bias: (L) initialized at 0
        self.bias = nn.Parameter(torch.zeros(self.n_labels))

    def global_w(self) -> torch.Tensor:
        return torch.softmax(self.g_raw, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, L)
        w_g = self.global_w()  # (M,)
        
        # w_eff = w_g + delta_w
        w_eff = w_g.view(-1, 1) + self.delta_w  # (M, L)
        
        # logits = sum_m (w_eff[m, l] * x[b, m, l]) + bias[l]
        # (B, M, L) * (1, M, L) -> (B, L)
        logits = (x * w_eff.unsqueeze(0)).sum(dim=1) + self.bias
        
        return logits

    def l2_reg(self, active_only: bool = True) -> torch.Tensor:
        if active_only:
            # Regularize only labels seen in training to avoid crushing priors for unseen labels
            dw_active = self.delta_w.index_select(1, self.active_idx)
            b_active = self.bias.index_select(0, self.active_idx)
            return LAMBDA_DELTA_L2 * (dw_active**2).mean() + LAMBDA_BIAS_L2 * (b_active**2).mean()
        else:
            return LAMBDA_DELTA_L2 * (self.delta_w**2).mean() + LAMBDA_BIAS_L2 * (self.bias**2).mean()


def train_and_evaluate(
    *,
    dataset: str,
    ensemble_keys: tuple[str, str, str],
    train_loader: torch.utils.data.DataLoader,
    train_eval_loader: torch.utils.data.DataLoader,
    y_train_true_eval: csr_matrix,
    test_loader: torch.utils.data.DataLoader,
    y_test_true: csr_matrix,
    full_train_loader: torch.utils.data.DataLoader,
    y_train_true: csr_matrix,
    active_idx: torch.Tensor,
) -> dict[str, object]:
    
    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    n_models = len(ensemble_keys)
    n_labels = y_train_true.shape[1]

    init_global: torch.Tensor | None = None
    cfg = get_dataset_config(dataset)
    if cfg.ensemble3_init_weights is not None:
        init_global = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)

    model = PerLabelApexEnsemble(
        n_models=n_models,
        n_labels=n_labels,
        active_idx=active_idx,
        init_global=init_global,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=BEST_LR,
        weight_decay=BEST_WEIGHT_DECAY,
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

    print(f"[DEBUG] Model initialized with {sum(p.numel() for p in model.parameters()):,} params")
    print(f"[DEBUG] Active labels: {model.n_active}/{model.n_labels} ({100.0*model.n_active/model.n_labels:.2f}%)")
    print(f"[DEBUG] Regularization: λ_bias={LAMBDA_BIAS_L2:g}, λ_delta={LAMBDA_DELTA_L2:g}")
    print()

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        
        last_loss_bce = 0.0
        last_loss_total = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_bce = criterion(logits, yb)
            
            loss_reg = model.l2_reg()
            loss = loss_bce + loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            last_loss_bce = float(loss_bce.item())
            last_loss_total = float(loss.item())

        # Diagnostics
        with torch.no_grad():
            gw = model.global_w().cpu().numpy()
            bg = 0.0 # float(model.bias_global.item())
            
            if model.n_labels > 0:
                wd = model.delta_w.detach().abs().cpu().numpy()
                wd_mean = float(wd.mean())
                bd = model.bias.detach().abs().cpu().numpy()
                bd_mean = float(bd.mean())
            else:
                wd_mean, bd_mean = 0.0, 0.0

        # Grad norms
        model.train()
        optimizer.zero_grad(set_to_none=True)
        xb_dbg, yb_dbg = next(iter(train_loader))
        xb_dbg, yb_dbg = xb_dbg.to(DEVICE), yb_dbg.to(DEVICE)
        logits_dbg = model(xb_dbg)
        loss_dbg = criterion(logits_dbg, yb_dbg)
        loss_dbg.backward()
        gn_gw = _grad_norm([model.g_raw])
        gn_active = _grad_norm([model.delta_w, model.bias])
        optimizer.zero_grad(set_to_none=True)

        train_res_eval = evaluate_model_batched(model, train_eval_loader, y_train_true_eval, k_values=(1000,), device=DEVICE)
        train_ndcg1000 = train_res_eval["ndcg@1000"]
        test_metrics = evaluate_model_batched(model, test_loader, y_test_true, k_values=K_VALUES, f1_k=5, device=DEVICE)

        epoch_dt = time.perf_counter() - epoch_t0
        print(
            f"Epoch {epoch:02d} | loss={last_loss_total:.6f} (bce={last_loss_bce:.6f}) | "
            f"train_ndcg@1000(sub)={train_ndcg1000:.6f} | "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} ndcg@1000={test_metrics['ndcg@1000']:.6f} f1@5={test_metrics['f1@5']:.6f} | "
            f"{epoch_dt:.1f}s"
        )
        print(
            f"  [DEBUG] gw=[{','.join(f'{w:.4f}' for w in gw)}] bias_g={bg:.4f} | "
            f"|w_delta| mean={wd_mean:.2e} | |bias_d| mean={bd_mean:.2e} | "
            f"gn_gw={gn_gw:.2e} gn_active={gn_active:.2e}"
        )

        if train_ndcg1000 > best_metric:
            best_metric = train_ndcg1000
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_test_metrics = test_metrics.copy()
            best_n_used_test = int(test_metrics["n_used"])
            
            best_train_res = evaluate_model_batched(model, full_train_loader, y_train_true, k_values=K_VALUES, device=DEVICE)
            best_train_metrics = {k: v for k, v in best_train_res.items() if k.startswith("ndcg")}
            best_n_used_train = int(best_train_res["n_used"])
            epochs_no_improve = 0
            print(f"  [DEBUG] ★ New best at epoch {epoch}")
        else:
            epochs_no_improve += 1
        
        if epoch >= MIN_EPOCHS and epochs_no_improve >= PATIENCE:
            print(f"  [DEBUG] Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_train_metrics": best_train_metrics,
        "best_test_metrics": best_test_metrics,
        "best_n_used_train": best_n_used_train,
        "best_n_used_test": best_n_used_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yso-fi", choices=["yso-fi", "yso-en", "koko"])
    args = parser.parse_args()
    dataset = str(args.dataset)

    e3 = ensemble3_keys(dataset)
    model_name = f"torch_per_label_apex({','.join(e3)})"
    scoreboard_path = Path("SCOREBOARD.md")

    print(f"--- Training {model_name} on {dataset} ---")
    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in e3]
    
    active_mask = _label_active_mask(y_train_true, train_preds)
    active_idx = torch.from_numpy(np.flatnonzero(active_mask)).long()

    train_ds = SparseCSRDataset(train_preds, y_train_true, stack_dim=0, transform=lambda xy: (log1p_transform(xy[0]), xy[1]))
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=TRAIN_BATCH_SIZE, 
        shuffle=True, 
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=4,
        prefetch_factor=2,
    )

    full_train_ds = SparseCSRDataset(train_preds, stack_dim=0, transform=log1p_transform)
    full_train_loader = torch.utils.data.DataLoader(full_train_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_eval = min(EARLY_STOP_EVAL_ROWS, y_train_true.shape[0])
    idx = rng.choice(y_train_true.shape[0], size=n_eval, replace=False)
    train_eval_loader = torch.utils.data.DataLoader(SparseCSRDataset([p[idx] for p in train_preds], stack_dim=0, transform=log1p_transform), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    y_train_true_eval = y_train_true[idx]

    y_test_true = load_csr(str(truth_path(dataset, "test")))
    test_ds = SparseCSRDataset([load_csr(str(pred_path(dataset, "test", k))) for k in e3], stack_dim=0, transform=log1p_transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=EVAL_BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
    )

    result = train_and_evaluate(
        dataset=dataset, ensemble_keys=e3,
        train_loader=train_loader, train_eval_loader=train_eval_loader, y_train_true_eval=y_train_true_eval,
        test_loader=test_loader, y_test_true=y_test_true,
        full_train_loader=full_train_loader, y_train_true=y_train_true,
        active_idx=active_idx
    )

    update_markdown_scoreboard(scoreboard_path, model_name, dataset, "train", result["best_train_metrics"], result["best_n_used_train"], result["best_epoch"])
    update_markdown_scoreboard(scoreboard_path, model_name, dataset, "test", result["best_test_metrics"], result["best_n_used_test"], result["best_epoch"])
    print(f"\nFinal test metrics: ndcg@10={result['best_test_metrics']['ndcg@10']:.6f} ndcg@1000={result['best_test_metrics']['ndcg@1000']:.6f}")


if __name__ == "__main__":
    main()
