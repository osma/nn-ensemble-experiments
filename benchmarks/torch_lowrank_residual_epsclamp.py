# STATUS: EXPERIMENTAL
# Purpose: Low-rank residual ensemble (U@V) like torch_lowrank_residual, but clamps
# to [eps, 1-eps] instead of [0,1] to avoid exact 0/1 with BCELoss.
from __future__ import annotations

from pathlib import Path
import sys
import time

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
from benchmarks.preprocessing import SparseCSRDataset, sqrt_transform

DEVICE = get_device()

EPOCHS = 20
K_VALUES = (10, 1000)
PATIENCE = 2
MIN_EPOCHS = 2

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512

EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

LR = 0.003
WEIGHT_DECAY = 0.0

DEFAULT_RANK = 32
DEFAULT_LAMBDA_UV_L2 = 1e-2
DEFAULT_LAMBDA_BIAS_L2 = 1e-3
DEFAULT_EPS = 1e-6

TRAIN_SEED = 0


# (Removed csr_to_sqrt_tensor and _predict_in_batches in favor of SparseCSRDataset and evaluate_model_batched)


class LowRankResidualEpsClamp(nn.Module):
    def __init__(
        self,
        *,
        n_models: int,
        n_labels: int,
        rank: int,
        eps: float,
        init_global: torch.Tensor | None,
    ):
        super().__init__()
        if n_models < 1:
            raise ValueError("n_models must be positive")
        if n_labels < 1:
            raise ValueError("n_labels must be positive")
        if rank < 1:
            raise ValueError("rank must be positive")
        if not (0.0 < eps < 0.5):
            raise ValueError("eps must be in (0, 0.5)")

        self.n_models = int(n_models)
        self.n_labels = int(n_labels)
        self.rank = int(rank)
        self.eps = float(eps)

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

        self.U = nn.Parameter(torch.empty((self.n_models, self.rank), dtype=torch.float32))
        self.V = nn.Parameter(torch.zeros((self.rank, self.n_labels), dtype=torch.float32))
        nn.init.normal_(self.U, mean=0.0, std=0.1)

        self.bias = nn.Parameter(torch.zeros((self.n_labels,), dtype=torch.float32))

    def delta_w(self) -> torch.Tensor:
        return self.U @ self.V

    def effective_w(self) -> torch.Tensor:
        return self.global_w[:, None] + self.delta_w()

    def uv_l2(self) -> torch.Tensor:
        return (self.U.pow(2).mean() + self.V.pow(2).mean()) / 2.0

    def bias_l2(self) -> torch.Tensor:
        return self.bias.pow(2).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.effective_w()
        out_lin = (x * w_eff.unsqueeze(0)).sum(dim=1) + self.bias
        return torch.clamp(out_lin, self.eps, 1.0 - self.eps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="yso-fi",
        choices=["yso-fi", "yso-en", "koko"],
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--lambda-uv", type=float, default=DEFAULT_LAMBDA_UV_L2)
    parser.add_argument("--lambda-bias", type=float, default=DEFAULT_LAMBDA_BIAS_L2)
    args = parser.parse_args()

    dataset = str(args.dataset)
    rank = int(args.rank)
    eps = float(args.eps)
    lambda_uv = float(args.lambda_uv)
    lambda_bias = float(args.lambda_bias)

    if rank < 1:
        raise ValueError("rank must be positive")
    if not (0.0 < eps < 0.5):
        raise ValueError("eps must be in (0, 0.5)")
    if lambda_uv < 0:
        raise ValueError("lambda_uv must be nonnegative")
    if lambda_bias < 0:
        raise ValueError("lambda_bias must be nonnegative")

    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_lowrank_residual_epsclamp({','.join(ensemble_keys)})"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")
    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]

    n_samples_train = y_train_true.shape[0]
    n_labels = int(y_train_true.shape[1])
    n_models = len(train_preds)

    cfg = get_dataset_config(dataset)
    init_global: torch.Tensor | None = None
    if cfg.ensemble3_init_weights is not None:
        init_global = torch.tensor(cfg.ensemble3_init_weights, dtype=torch.float32)
        if init_global.shape[0] != n_models:
            raise ValueError(
                f"ensemble3_init_weights has length {init_global.shape[0]}, but n_models={n_models}."
            )

    model = LowRankResidualEpsClamp(
        n_models=n_models,
        n_labels=n_labels,
        rank=rank,
        eps=eps,
        init_global=init_global,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-8)
    criterion = nn.BCELoss()

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
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in ensemble_keys]
    test_ds = SparseCSRDataset(test_preds, stack_dim=0, transform=sqrt_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

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
            probs = model(xb)
            loss_main = criterion(probs, yb)
            loss_reg = lambda_uv * model.uv_l2() + lambda_bias * model.bias_l2()
            loss = loss_main + loss_reg
            loss.backward()
            optimizer.step()

        train_res_eval = evaluate_model_batched(model, train_eval_loader, y_train_true_eval, k_values=(1000,), device=DEVICE)
        train_ndcg1000 = train_res_eval["ndcg@1000"]

        test_metrics = evaluate_model_batched(model, test_loader, y_test_true, k_values=K_VALUES, f1_k=5, device=DEVICE)

        epoch_dt = time.perf_counter() - epoch_t0
        print(
            f"[rank={rank} eps={eps:g} lambda_uv={lambda_uv:g} lambda_bias={lambda_bias:g}] "
            f"Epoch {epoch:02d} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"total={epoch_dt:.3f}s"
        )

        current = float(train_ndcg1000)
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
