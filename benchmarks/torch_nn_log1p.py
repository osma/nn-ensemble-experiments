# STATUS: EXPERIMENTAL
# Purpose: NN ensemble with log1p preprocessing instead of sqrt.
from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csr_matrix

from benchmarks.datasets import ensemble3_keys, pred_path, truth_path
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

LR = 1e-3
WEIGHT_DECAY = 0.01
TRAIN_SEED = 0


def csr_to_log1p_tensor(csr: csr_matrix) -> torch.Tensor:
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


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


class Log1pNNModel(nn.Module):
    def __init__(
        self,
        source_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.model_config = {
            "source_dim": int(source_dim),
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "output_dim": int(output_dim),
            "dropout_rate": float(dropout_rate),
        }
        self.conv = nn.Conv1d(source_dim, 1, 1, bias=False)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden = nn.Linear(input_dim * source_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.delta_layer = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / float(self.model_config["source_dim"]))
        nn.init.zeros_(self.delta_layer.weight)
        nn.init.zeros_(self.delta_layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs to have shape (B, M, L), got {tuple(inputs.shape)}")

        mean = self.conv(inputs)[:, 0, :]
        x = self.flatten(inputs)
        x = self.dropout1(x)
        x = F.relu(self.hidden(x))
        x = self.dropout2(x)
        delta = self.delta_layer(x)
        corrected = mean + delta
        return torch.clamp(corrected, min=0.0, max=1.0)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yso-fi", choices=["yso-fi", "yso-en", "koko"])
    args = parser.parse_args()
    dataset = str(args.dataset)

    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    e3 = ensemble3_keys(dataset)
    model_name = f"torch_nn_log1p({','.join(e3)})"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr(str(truth_path(dataset, "train")))
    train_preds = [load_csr(str(pred_path(dataset, "train", k))) for k in e3]

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
    test_preds = [load_csr(str(pred_path(dataset, "test", k))) for k in e3]
    X_test = torch.stack([csr_to_log1p_tensor(p) for p in test_preds], dim=1)

    n_models = int(X_train.shape[1])
    n_labels = int(X_train.shape[2])
    if n_models != 3:
        raise ValueError(f"Expected 3-way ensemble input (M=3), got M={n_models}")

    HIDDEN_DIM = 100
    DROPOUT_RATE = 0.2

    model = Log1pNNModel(
        source_dim=n_models,
        input_dim=n_labels,
        hidden_dim=HIDDEN_DIM,
        output_dim=n_labels,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-8)
    criterion = nn.BCELoss()

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type == "cuda"))

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
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)
        train_ndcg1000, _n_used_train_eval = ndcg_at_k_dense(y_train_true_eval, train_scores_eval, k=1000)

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
            f"Epoch {epoch:02d} | "
            f"loss={loss.item():.6f} | "
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
