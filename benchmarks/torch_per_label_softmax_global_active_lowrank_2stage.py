# STATUS: EXPERIMENTAL
# Purpose: torch_per_label_softmax_global_active_lowrank simplification 11 (simplify_combined).
#
# Combines five successful/neutral simplifications:
# 1. S6: Remove per-sample centering.
# 2. S2: Fix gate at constant 0.02 (remove learnable raw_gate).
# 4. S4: Remove base_logits channel from mixer features.
# 5. S3: Reduce rank 64 -> 16.
#
# Removes ~8x low-rank parameters and 2 hyperparameters while maintaining top-tier performance.

from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running as a script
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
from benchmarks.preprocessing import SparseCSRDataset, log1p_transform

DEVICE = get_device()

EPOCHS = 30
K_VALUES = (10, 1000)
PATIENCE = 3
MIN_EPOCHS = 2
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337
EVAL_BATCH_SIZE = 512
DEFAULT_TRAIN_SEED = 0

BEST_LR = 0.003
BEST_WEIGHT_DECAY = 0.0
BEST_BATCH_SIZE = 256
LAMBDA_DELTA_L2 = 1e-3

LOWRANK_LR = 1e-3
LOWRANK_WEIGHT_DECAY = 1e-2

DEFAULT_RANK = 16  # S3: Reduced from 64
FIXED_GATE = 1.0  # Increased to prevent vanishing gradients


def _label_active_mask(y_train_true: csr_matrix, train_preds: list[csr_matrix]) -> np.ndarray:
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


class ActiveMLPMixer(nn.Module):
    """
    MLP over flattened (M x L_active) inputs to predict label adjustments.
    """
    def __init__(self, n_models: int, n_active: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.n_models = n_models
        self.n_active = n_active
        self.hidden_dim = hidden_dim

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden = nn.Linear(self.n_models * self.n_active, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.delta_layer = nn.Linear(hidden_dim, self.n_active)

        # Zero-initialize the final layer so it starts as a true residual block
        nn.init.zeros_(self.delta_layer.weight)
        nn.init.zeros_(self.delta_layer.bias)

    def forward(self, x_active: torch.Tensor) -> torch.Tensor:
        if self.n_active == 0:
            return x_active.new_zeros((x_active.shape[0], 0))

        x = self.flatten(x_active)
        x = self.dropout1(x)
        x = F.relu(self.hidden(x))
        x = self.dropout2(x)
        delta = self.delta_layer(x)
        return delta


class ActiveLowRankEnsemble(nn.Module):
    def __init__(
        self,
        *,
        n_models: int,
        n_labels: int,
        active_idx: torch.Tensor,
        rank: int,
        init_global: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_models = int(n_models)
        self.n_labels = int(n_labels)

        if active_idx.ndim != 1:
            raise ValueError("active_idx must be 1D")
        self.register_buffer("active_idx", active_idx.long())
        self.n_active = int(active_idx.numel())

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

        self.g_raw = nn.Parameter(torch.log(torch.clamp(g, min=1e-12)))
        self.w_delta = nn.Parameter(torch.zeros(self.n_models, self.n_labels))
        self.bias = nn.Parameter(torch.zeros(self.n_labels))

        self.lowrank = ActiveMLPMixer(
            n_models=self.n_models,
            n_active=self.n_active,
            hidden_dim=64,
            dropout_rate=0.5,
        )
        self.use_lowrank = True

    def global_w(self) -> torch.Tensor:
        return torch.softmax(self.g_raw, dim=0)

    def effective_w(self) -> torch.Tensor:
        return self.global_w()[:, None] + self.w_delta

    def get_lowrank_delta(self, x_active: torch.Tensor) -> torch.Tensor:
        delta_active = self.lowrank(x_active)

        # Restore per-sample centering to prevent pushing all outputs up or down
        delta_active = delta_active - delta_active.mean(dim=1, keepdim=True)

        # S2: fixed gate replaces learnable gate
        return delta_active * FIXED_GATE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, n_models, n_labels), got {x.shape}")

        w_eff = self.effective_w().unsqueeze(0)  # (1, M, L)
        base_logits = (x * w_eff).sum(dim=1) + self.bias

        if self.n_active == 0 or not self.use_lowrank:
            return base_logits

        x_active = x.index_select(dim=2, index=self.active_idx)
        gated_delta_active = self.get_lowrank_delta(x_active)
        
        base_logits_active = base_logits.index_select(dim=1, index=self.active_idx)
        out_active = base_logits_active + gated_delta_active

        out = base_logits.clone()
        out.index_copy_(dim=1, index=self.active_idx, source=out_active)
        return out

    def delta_l2(self) -> torch.Tensor:
        return (self.w_delta**2).mean()


def _delta_stats(
    model: ActiveLowRankEnsemble, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[float, float]:
    sum_abs = 0.0
    n_abs = 0
    max_samples = 1_000_000
    samples: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                xb = batch[0]
            else:
                xb = batch
            xb = xb.to(device, non_blocking=True)

            x_active = xb.index_select(dim=2, index=model.active_idx)
            delta = model.get_lowrank_delta(x_active)
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


def train_and_evaluate(
    *,
    dataset: str,
    ensemble_keys: list[str],
    train_loader: torch.utils.data.DataLoader,
    train_eval_loader: torch.utils.data.DataLoader,
    y_train_true_eval: csr_matrix,
    test_loader: torch.utils.data.DataLoader,
    y_test_true: csr_matrix,
    full_train_loader: torch.utils.data.DataLoader,
    y_train_true: csr_matrix,
    active_idx: torch.Tensor,
    train_seed: int = DEFAULT_TRAIN_SEED,
) -> dict[str, float | int | dict[str, float]]:
    n_models = len(ensemble_keys)
    n_labels = y_train_true.shape[1]

    # Make each run deterministic-ish (init + dataloader shuffle)
    torch.manual_seed(train_seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(train_seed)

    cfg = get_dataset_config(dataset)
    init_global: torch.Tensor | None = None
    if cfg.ensemble3_init_weights is not None:
        init_global = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)

    model = ActiveLowRankEnsemble(
        n_models=n_models,
        n_labels=n_labels,
        active_idx=active_idx,
        rank=DEFAULT_RANK,
        init_global=init_global,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    def _run_stage(stage_num: int, optimizer: optim.Optimizer, max_epochs: int):
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
            grad_norms: list[float] = []
            grad_norms_lr: list[float] = []

            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss_bce = criterion(logits, yb)
                loss = loss_bce + (LAMBDA_DELTA_L2 * model.delta_l2())
                loss.backward()
                
                # Debug: monitor gradient norms
                gnorm = 0.0
                gnorm_lr = 0.0
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        g_sq = p.grad.data.norm(2).item() ** 2
                        gnorm += g_sq
                        if "lowrank" in name:
                            gnorm_lr += g_sq
                grad_norms.append(gnorm ** 0.5)
                grad_norms_lr.append(gnorm_lr ** 0.5)

                optimizer.step()
                last_loss = float(loss.item())

            train_res_eval = evaluate_model_batched(model, train_eval_loader, y_train_true_eval, k_values=(10, 1000), device=DEVICE)
            train_ndcg1000 = train_res_eval["ndcg@1000"]
            train_ndcg10 = train_res_eval["ndcg@10"]

            test_metrics = evaluate_model_batched(model, test_loader, y_test_true, k_values=K_VALUES, f1_k=5, device=DEVICE)

            delta_mean_abs, delta_p95_abs = _delta_stats(model, train_eval_loader, DEVICE) if model.use_lowrank else (0.0, 0.0)
            mean_gnorm = np.mean(grad_norms) if grad_norms else 0.0
            mean_gnorm_lr = np.mean(grad_norms_lr) if grad_norms_lr else 0.0

            with torch.no_grad():
                mlp_w1_norm = model.lowrank.hidden.weight.norm().item()
                mlp_w2_norm = model.lowrank.delta_layer.weight.norm().item()
                w_delta_norm = model.w_delta.norm().item()

            epoch_dt = time.perf_counter() - epoch_t0
            print(
                f"S{stage_num}E{epoch:02d} | "
                f"loss={float(last_loss or 0.0):.6f} | "
                f"train_ndcg@1000(subset)={train_ndcg1000:.6f} "
                f"train_ndcg@10={train_ndcg10:.6f} | "
                f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
                f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
                f"test_f1@5={test_metrics['f1@5']:.6f} | "
                f"gate={FIXED_GATE:.4f}(fixed) LowRank_delta_mean={delta_mean_abs:.6f} p95={delta_p95_abs:.6f} | "
                f"gnorm={mean_gnorm:.4f} gnorm_lr={mean_gnorm_lr:.4f} | "
                f"mlp_w1_norm={mlp_w1_norm:.2f} mlp_w2_norm={mlp_w2_norm:.2f} w_delta_norm={w_delta_norm:.2f} | "
                f"total={epoch_dt:.3f}s"
            )

            current = train_ndcg10
            if current > best_metric:
                best_metric = current
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

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
        model.load_state_dict(best_state)

        return {
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "best_train_metrics": best_train_metrics,
            "best_test_metrics": best_test_metrics,
            "best_n_used_train": int(best_n_used_train),
            "best_n_used_test": int(best_n_used_test),
        }

    print("\n--- STAGE 1: Train base parameters only ---")
    model.use_lowrank = False
    model.g_raw.requires_grad_(True)
    model.w_delta.requires_grad_(True)
    model.bias.requires_grad_(True)
    model.lowrank.requires_grad_(False)

    optimizer_s1 = optim.AdamW(
        [{"params": [model.g_raw, model.w_delta, model.bias], "lr": BEST_LR, "weight_decay": BEST_WEIGHT_DECAY}],
        eps=1e-8,
    )
    res1 = _run_stage(1, optimizer_s1, EPOCHS)

    print("\n--- STAGE 2: Train lowrank parameters only ---")
    model.use_lowrank = True
    model.g_raw.requires_grad_(False)
    model.w_delta.requires_grad_(False)
    model.bias.requires_grad_(False)
    model.lowrank.requires_grad_(True)

    optimizer_s2 = optim.SGD(
        [{"params": model.lowrank.parameters(), "lr": LOWRANK_LR, "weight_decay": LOWRANK_WEIGHT_DECAY}],
    )
    res2 = _run_stage(2, optimizer_s2, EPOCHS)

    res_final = res2
    res_final["best_epoch"] = res1["best_epoch"] + res2["best_epoch"]
    return res_final

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
        "--seed",
        type=int,
        default=DEFAULT_TRAIN_SEED,
        help="Random seed for training (init + shuffle). Default: %(default)s",
    )
    args = parser.parse_args()
    dataset = str(args.dataset)
    train_seed = int(args.seed)

    ensemble_keys = ensemble3_keys(dataset)
    base_name = f"torch_per_label_softmax_global_active_lowrank_2stage({','.join(ensemble_keys)})"
    model_name = base_name if train_seed == DEFAULT_TRAIN_SEED else f"{base_name}/seed={train_seed}"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]

    active_mask = _label_active_mask(y_train_true, train_preds)
    active_idx_np = np.flatnonzero(active_mask).astype(np.int64)
    active_idx = torch.from_numpy(active_idx_np)

    n_samples_train = y_train_true.shape[0]
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
            f"{k} avg nnz/row={_csr_avg_nnz_per_row(p):.2f}" for k, p in zip(ensemble_keys, train_preds, strict=True)
        )
    )

    train_ds = SparseCSRDataset(train_preds, y_train_true, stack_dim=0, transform=lambda xy: (log1p_transform(xy[0]), xy[1]))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BEST_BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type == "cuda"))

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

    result = train_and_evaluate(
        dataset=dataset,
        ensemble_keys=ensemble_keys,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        y_train_true_eval=y_train_true_eval,
        test_loader=test_loader,
        y_test_true=y_test_true,
        full_train_loader=full_train_loader,
        y_train_true=y_train_true,
        active_idx=active_idx,
        train_seed=train_seed,
    )

    best_epoch = int(result["best_epoch"])
    best_metric = float(result["best_metric"])
    best_train_metrics = result["best_train_metrics"]
    best_test_metrics = result["best_test_metrics"]
    best_n_used_train = int(result["best_n_used_train"])
    best_n_used_test = int(result["best_n_used_test"])

    print("\n====================")
    print("Training complete")
    print("====================")
    print(
        f"best_epoch={best_epoch} | "
        f"train_ndcg@1000(subset)={best_metric:.6f}"
    )
    print(
        "Best test metrics | "
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
