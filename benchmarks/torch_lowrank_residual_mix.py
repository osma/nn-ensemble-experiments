# STATUS: EXPERIMENTAL
# Purpose: Low-rank residual weights (U@V) + A1 cross-label mixing (P@Q).
#
# Baseline component:
#   out_base[b,l] = sum_m (w_global[m] + delta_w[m,l]) * x[b,m,l]
# Mixing component (A1):
#   s[b,l]   = sum_m w_global[m] * x[b,m,l]
#   mix[b,:] = s[b,:] @ (P@Q)
# Final:
#   out = clamp(out_base + mix + bias, [0,1])
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
from benchmarks.metrics import load_csr, ndcg_at_k_dense, f1_at_k_dense, update_markdown_scoreboard

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
DEFAULT_MIX_RANK = 32
DEFAULT_LAMBDA_UV_L2 = 1e-2
DEFAULT_LAMBDA_MIX_L2 = 1e-3
DEFAULT_LAMBDA_BIAS_L2 = 1e-3

TRAIN_SEED = 0


def csr_to_sqrt_tensor(csr: csr_matrix) -> torch.Tensor:
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
            outs.append(model(xb).detach().cpu())
    return torch.cat(outs, dim=0)


class LowRankResidualMix(nn.Module):
    def __init__(
        self,
        *,
        n_models: int,
        n_labels: int,
        rank: int,
        mix_rank: int,
        init_global: torch.Tensor | None,
    ):
        super().__init__()
        if n_models < 1:
            raise ValueError("n_models must be positive")
        if n_labels < 1:
            raise ValueError("n_labels must be positive")
        if rank < 1:
            raise ValueError("rank must be positive")
        if mix_rank < 1:
            raise ValueError("mix_rank must be positive")

        self.n_models = int(n_models)
        self.n_labels = int(n_labels)
        self.rank = int(rank)
        self.mix_rank = int(mix_rank)

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

        # Per-label residual weights: delta_w = U@V, V=0 init => delta starts at 0.
        self.U = nn.Parameter(torch.empty((self.n_models, self.rank), dtype=torch.float32))
        self.V = nn.Parameter(torch.zeros((self.rank, self.n_labels), dtype=torch.float32))
        nn.init.normal_(self.U, mean=0.0, std=0.1)

        # A1 mixing: mix = s @ (P@Q), Q=0 init => mix starts at 0.
        self.P = nn.Parameter(torch.empty((self.n_labels, self.mix_rank), dtype=torch.float32))
        self.Q = nn.Parameter(torch.zeros((self.mix_rank, self.n_labels), dtype=torch.float32))
        nn.init.normal_(self.P, mean=0.0, std=0.02)

        self.bias = nn.Parameter(torch.zeros((self.n_labels,), dtype=torch.float32))

    def delta_w(self) -> torch.Tensor:
        return self.U @ self.V

    def effective_w(self) -> torch.Tensor:
        return self.global_w[:, None] + self.delta_w()

    def uv_l2(self) -> torch.Tensor:
        return (self.U.pow(2).mean() + self.V.pow(2).mean()) / 2.0

    def mix_l2(self) -> torch.Tensor:
        return (self.P.pow(2).mean() + self.Q.pow(2).mean()) / 2.0

    def bias_l2(self) -> torch.Tensor:
        return self.bias.pow(2).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, M, L), got {tuple(x.shape)}")
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected x with (M={self.n_models}, L={self.n_labels}), got {tuple(x.shape)}"
            )

        w_eff = self.effective_w()
        out_base = (x * w_eff.unsqueeze(0)).sum(dim=1)  # (B, L)

        s = (x * self.global_w[None, :, None]).sum(dim=1)  # (B, L)
        mix = (s @ self.P) @ self.Q  # (B, L)

        out_lin = out_base + mix + self.bias
        return torch.clamp(out_lin, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="yso-fi",
        choices=["yso-fi", "yso-en", "koko"],
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--mix-rank", type=int, default=DEFAULT_MIX_RANK)
    parser.add_argument("--lambda-uv", type=float, default=DEFAULT_LAMBDA_UV_L2)
    parser.add_argument("--lambda-mix", type=float, default=DEFAULT_LAMBDA_MIX_L2)
    parser.add_argument("--lambda-bias", type=float, default=DEFAULT_LAMBDA_BIAS_L2)
    args = parser.parse_args()

    dataset = str(args.dataset)
    rank = int(args.rank)
    mix_rank = int(args.mix_rank)
    lambda_uv = float(args.lambda_uv)
    lambda_mix = float(args.lambda_mix)
    lambda_bias = float(args.lambda_bias)

    if rank < 1:
        raise ValueError("rank must be positive")
    if mix_rank < 1:
        raise ValueError("mix_rank must be positive")
    if lambda_uv < 0:
        raise ValueError("lambda_uv must be nonnegative")
    if lambda_mix < 0:
        raise ValueError("lambda_mix must be nonnegative")
    if lambda_bias < 0:
        raise ValueError("lambda_bias must be nonnegative")

    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    ensemble_keys = ensemble3_keys(dataset)
    model_name = f"torch_lowrank_residual_mix({','.join(ensemble_keys)})"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in ensemble_keys]
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

    model = LowRankResidualMix(
        n_models=n_models,
        n_labels=n_labels,
        rank=rank,
        mix_rank=mix_rank,
        init_global=init_global,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-8)
    criterion = nn.BCELoss()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train),
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
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                probs = model(xb)
                loss_main = criterion(probs, yb)
                loss_reg = (
                    lambda_uv * model.uv_l2()
                    + lambda_mix * model.mix_l2()
                    + lambda_bias * model.bias_l2()
                )
                loss = loss_main + loss_reg
                loss.backward()
                optimizer.step()

        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)
        train_ndcg1000, _ = ndcg_at_k_dense(y_train_true_eval, train_scores_eval, k=1000)

        with _Timer() as t_pred_test:
            test_scores = _predict_in_batches(model, X_test)

        test_metrics: dict[str, float] = {}
        n_used_test: int | None = None
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg
        f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
        test_metrics["f1@5"] = f1

        print(
            f"[rank={rank} mix_rank={mix_rank} lambda_uv={lambda_uv:g} lambda_mix={lambda_mix:g} lambda_bias={lambda_bias:g}] "
            f"Epoch {epoch:02d} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
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
