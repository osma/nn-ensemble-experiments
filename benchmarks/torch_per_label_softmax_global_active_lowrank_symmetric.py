# STATUS: EXPERIMENTAL
# Purpose: torch_per_label_softmax_global + active-label Low-Rank cross-label mixer.
# Simplification 8: simplify_symmetric_lowrank
#
# Ties U and V matrices (symmetric factorization) to halve label-side parameters
# and enforce symmetric cross-label similarity.

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

EPOCHS = 20
K_VALUES = (10, 1000)
PATIENCE = 3
MIN_EPOCHS = 2
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337
EVAL_BATCH_SIZE = 512
TRAIN_SEED = 0

BEST_LR = 0.003
BEST_WEIGHT_DECAY = 0.0
BEST_BATCH_SIZE = 256
LAMBDA_DELTA_L2 = 1e-3

LOWRANK_LR = 1e-4
LOWRANK_WEIGHT_DECAY = 1e-2

DEFAULT_RANK = 64


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


class LowRankActiveMixer(nn.Module):
    """
    Project feats (B, C, L_active) -> (B, L_active) via low-rank linear mixer.
    Symmetric version: U and V are tied.
    """
    def __init__(self, n_channels: int, n_active: int, rank: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_active = n_active
        self.rank = rank

        self.U = nn.Parameter(0.01 * torch.randn(self.n_active, self.rank))
        # V is tied to U
        self.W = nn.Parameter(0.01 * torch.randn(self.n_channels, self.rank))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        if self.n_active == 0:
            return feats.new_zeros((B, 0))

        # (B*C, L) @ (L, r) -> (B*C, r)
        x2 = feats.reshape(B * self.n_channels, self.n_active)
        Z = x2 @ self.U
        Z = Z.reshape(B, self.n_channels, self.rank)
        
        # Mix channels
        h = torch.sum(Z * self.W.unsqueeze(0), dim=1)  # (B, r)
        
        # Project back to labels using U.t() (Symmetric)
        delta = h @ self.U.t()  # (B, L_active)
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

        # C = M raw logits + 1 base logit
        self.lowrank = LowRankActiveMixer(n_channels=self.n_models + 1, n_active=self.n_active, rank=rank)

        # Gate on the delta. max=0.2. Init p=0.1.
        raw_init = float(np.log(0.1 / 0.9))
        self.raw_gate = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def global_w(self) -> torch.Tensor:
        return torch.softmax(self.g_raw, dim=0)

    def effective_w(self) -> torch.Tensor:
        return self.global_w()[:, None] + self.w_delta

    def get_lowrank_delta(self, x_active: torch.Tensor, base_logits_active: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([x_active, base_logits_active.unsqueeze(1)], dim=1)  # (B, M+1, L_active)
        delta_active = self.lowrank(feats)
        
        DELTA_CLAMP = 0.5
        delta_active = DELTA_CLAMP * torch.tanh(delta_active / DELTA_CLAMP)
        delta_active = delta_active - delta_active.mean(dim=1, keepdim=True)
        
        DELTA_GATE_MAX = 0.2
        gate = torch.sigmoid(self.raw_gate) * DELTA_GATE_MAX
        return delta_active * gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, n_models, n_labels), got {x.shape}")
        
        w_eff = self.effective_w().unsqueeze(0)  # (1, M, L)
        base_logits = (x * w_eff).sum(dim=1) + self.bias

        if self.n_active == 0:
            return base_logits

        x_active = x.index_select(dim=2, index=self.active_idx)
        base_logits_active = base_logits.index_select(dim=1, index=self.active_idx)
        
        gated_delta_active = self.get_lowrank_delta(x_active, base_logits_active)
        
        out_active = base_logits_active * (1.0 + gated_delta_active)

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
            w_eff = model.effective_w().unsqueeze(0)
            base_logits = (xb * w_eff).sum(dim=1) + model.bias
            base_logits_active = base_logits.index_select(dim=1, index=model.active_idx)
            
            delta = model.get_lowrank_delta(x_active, base_logits_active)
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
) -> dict[str, float | int | dict[str, float]]:
    n_models = len(ensemble_keys)
    n_labels = y_train_true.shape[1]

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

    # Use a two-tiered learning rate
    optimizer = optim.AdamW(
        [
            {"params": [model.g_raw, model.w_delta, model.bias], "lr": BEST_LR, "weight_decay": BEST_WEIGHT_DECAY},
            {"params": model.lowrank.parameters(), "lr": LOWRANK_LR, "weight_decay": LOWRANK_WEIGHT_DECAY},
            {"params": [model.raw_gate], "lr": LOWRANK_LR, "weight_decay": 0.0},
        ],
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
        total_loss = 0.0
        total_batches = 0
        max_grad_norm = 0.0
        
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_bce = criterion(logits, yb)
            loss = loss_bce + (LAMBDA_DELTA_L2 * model.delta_l2())
            loss.backward()
            
            # Debug: monitor grad norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.detach().data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            max_grad_norm = max(max_grad_norm, grad_norm)
            
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)

        train_res_eval = evaluate_model_batched(model, train_eval_loader, y_train_true_eval, k_values=(10, 1000), device=DEVICE)
        train_ndcg1000 = train_res_eval["ndcg@1000"]
        train_ndcg10 = train_res_eval["ndcg@10"]

        test_metrics = evaluate_model_batched(model, test_loader, y_test_true, k_values=K_VALUES, f1_k=5, device=DEVICE)

        delta_mean_abs, delta_p95_abs = _delta_stats(model, train_eval_loader, DEVICE)
        with torch.no_grad():
            gate_val = torch.sigmoid(model.raw_gate).item()

        epoch_dt = time.perf_counter() - epoch_t0

        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.6f} | grad_norm_max={max_grad_norm:.4f} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} train_ndcg@10={train_ndcg10:.6f} | "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"gate={gate_val:.4f} LowRank_delta_mean={delta_mean_abs:.6f} p95={delta_p95_abs:.6f} | "
            f"total={epoch_dt:.3f}s"
        )

        current = train_ndcg1000
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
    assert best_epoch is not None
    assert best_train_metrics is not None
    assert best_test_metrics is not None
    assert best_n_used_train is not None
    assert best_n_used_test is not None

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

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_per_label_softmax_global_active_lowrank_symmetric({','.join(ensemble_keys)})"
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
