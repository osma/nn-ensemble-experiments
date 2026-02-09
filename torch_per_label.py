# STATUS: ACTIVE (recommended base model)
# Purpose: Best-performing per-label linear ensemble trained with BCE on raw logits.
from pathlib import Path
import sys
import time
import itertools

# Allow running as a script: `uv run benchmarks/torch_per_label.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

from benchmarks.device import get_device
from benchmarks.metrics import (
    load_csr,
    ndcg_at_k_dense,
    f1_at_k_dense,
    update_markdown_scoreboard,
)


class PerLabelWeightedEnsemble(nn.Module):
    """
    Per-label weighted ensemble with bias.

    For each label l:
        score[l] = sum_m w[m, l] * x[m, l] + b[l]

    Notes:
    - Applies a fixed sub-linear log1p transform to inputs.

    Input:
        x: (batch, M, L)
            M = number of base models
            L = number of labels

    Output:
        (batch, L) raw logits

    Notes:
    - Returns raw logits (no clamp, no sigmoid).
    - Intended for use with BCEWithLogitsLoss or ranking-aware losses.
    """

    def __init__(self, n_models: int, n_labels: int):
        super().__init__()
        self.n_models = n_models
        self.n_labels = n_labels

        # Per-model, per-label weights
        self.weights = nn.Parameter(torch.full((n_models, n_labels), 1.0 / n_models))

        # Per-label bias
        self.bias = nn.Parameter(torch.zeros(n_labels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, n_models, n_labels)
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input of shape (batch, n_models, n_labels), got {x.shape}"
            )
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected input with n_models={self.n_models}, "
                f"n_labels={self.n_labels}, got {x.shape}"
            )

        # Inputs are already log1p-scaled during preprocessing
        x_scaled = x

        weighted = x_scaled * self.weights.unsqueeze(0)
        out = weighted.sum(dim=1) + self.bias
        return out


# ============================
# Training / evaluation script
# ============================

DEVICE = get_device()
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
K_VALUES = (10, 1000)

PATIENCE = 2
MIN_EPOCHS = 2

EVAL_BATCH_SIZE = 512
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337


def csr_to_dense_tensor(csr):
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())


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


def train_and_evaluate(
    lr: float,
    weight_decay: float,
    batch_size: int,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    y_train_true_eval: csr_matrix,
    X_train_eval: torch.Tensor,
    y_test_true: csr_matrix,
    X_test: torch.Tensor,
    n_models: int,
    n_labels: int,
) -> tuple[float, dict[str, float], int]:
    """Train and evaluate model with given hyperparameters."""
    
    model = PerLabelWeightedEnsemble(
        n_models=n_models,
        n_labels=n_labels,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8,
    )

    # Unweighted BCE performs best for NDCG in this setup
    criterion = nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Early stopping: select best epoch by TRAIN NDCG@1000 (no test leakage)
    best_metric = float("-inf")
    best_state = None
    best_train_metrics = None
    best_test_metrics = None
    best_n_used_train = None
    best_n_used_test = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.perf_counter()

        model.train()
        with _Timer() as t_train_step:
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # --- Train evaluation for early stopping (subset only) ---
        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)

        t_ndcg_train: dict[int, float] = {}
        with _Timer() as t:
            train_ndcg1000, n_used_train = ndcg_at_k_dense(
                y_train_true_eval, train_scores_eval, k=1000
            )
        assert t.dt is not None
        t_ndcg_train[1000] = t.dt

        # --- Test evaluation (batched; no CSR conversion) ---
        with _Timer() as t_pred_test:
            test_scores = _predict_in_batches(model, X_test)

        test_metrics = {}
        t_ndcg_test: dict[int, float] = {}
        for k in K_VALUES:
            with _Timer() as t:
                ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
            assert t.dt is not None
            t_ndcg_test[k] = t.dt
            test_metrics[f"ndcg@{k}"] = ndcg

        with _Timer() as t_f1:
            f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
        test_metrics["f1@5"] = f1

        epoch_dt = time.perf_counter() - epoch_t0

        def _dt(timer: _Timer) -> float:
            return float(timer.dt) if timer.dt is not None else 0.0

        print(
            f"Epoch {epoch:02d} timing | "
            f"train_step={_dt(t_train_step):.3f}s | "
            f"pred_train={_dt(t_pred_train):.3f}s | "
            f"ndcg_train@1000={t_ndcg_train.get(1000, 0.0):.3f}s | "
            f"pred_test={_dt(t_pred_test):.3f}s | "
            f"ndcg_test@10={t_ndcg_test.get(10, 0.0):.3f}s ndcg_test@1000={t_ndcg_test.get(1000, 0.0):.3f}s | "
            f"f1@5={_dt(t_f1):.3f}s | "
            f"total={epoch_dt:.3f}s"
        )

        current = train_ndcg1000
        if current > best_metric:
            best_metric = current
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Compute full train metrics only for the best epoch snapshot
            full_train_scores = _predict_in_batches(model, X_train)
            best_train_metrics = {}
            for k in K_VALUES:
                ndcg, n_used_train_full = ndcg_at_k_dense(
                    y_train_true_eval, full_train_scores, k=k
                )
                best_train_metrics[f"ndcg@{k}"] = ndcg
            best_n_used_train = n_used_train_full

            best_test_metrics = test_metrics.copy()
            best_n_used_test = n_used_test
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch >= MIN_EPOCHS and epochs_no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    
    # Return the best validation metric (train_ndcg1000) and test metrics
    return best_metric, best_test_metrics, best_n_used_test


def main():
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr("data/train-output.npz")
    train_preds = [
        load_csr("data/train-bonsai.npz"),
        load_csr("data/train-fasttext.npz"),
        load_csr("data/train-mllm.npz"),
    ]

    # Keep X_train on CPU; move only minibatches to GPU.
    X_train = torch.stack([csr_to_dense_tensor(p) for p in train_preds], dim=1)

    # Keep Y_train on CPU (requested).
    Y_train = csr_to_dense_tensor(y_train_true)

    # Fixed random subset of train rows for per-epoch early stopping metric
    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_train = X_train.shape[0]
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_train)
    train_eval_idx = rng.choice(n_train, size=n_eval, replace=False)
    X_train_eval = X_train[train_eval_idx]
    y_train_true_eval = y_train_true[train_eval_idx]

    print("Loading test data...")

    # Keep y_test_true / Y_test on CPU (requested).
    y_test_true = load_csr("data/test-output.npz")
    test_preds = [
        load_csr("data/test-bonsai.npz"),
        load_csr("data/test-fasttext.npz"),
        load_csr("data/test-mllm.npz"),
    ]

    # Keep X_test on CPU; move to GPU only for evaluation forward pass.
    X_test = torch.stack([csr_to_dense_tensor(p) for p in test_preds], dim=1)

    n_models = X_train.shape[1]
    n_labels = X_train.shape[2]

    # Define hyperparameter search space
    lr_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    weight_decay_values = [0.0, 1e-4, 1e-3, 1e-2]
    batch_size_values = [8, 16, 32, 64]

    # Generate all combinations
    param_combinations = list(itertools.product(
        lr_values, weight_decay_values, batch_size_values
    ))

    print(f"Running grid search over {len(param_combinations)} combinations...")
    
    best_score = float("-inf")
    best_params = None
    best_test_metrics = None
    best_n_samples = None

    for i, (lr, weight_decay, batch_size) in enumerate(param_combinations):
        print(f"\n=== Testing combination {i+1}/{len(param_combinations)} ===")
        print(f"Parameters: lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}")
        
        try:
            score, test_metrics, n_samples = train_and_evaluate(
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                X_train=X_train,
                Y_train=Y_train,
                y_train_true_eval=y_train_true_eval,
                X_train_eval=X_train_eval,
                y_test_true=y_test_true,
                X_test=X_test,
                n_models=n_models,
                n_labels=n_labels,
            )
            
            if score > best_score:
                best_score = score
                best_params = (lr, weight_decay, batch_size)
                best_test_metrics = test_metrics
                best_n_samples = n_samples
                
            print(f"Score: {score:.6f}")
            
        except Exception as e:
            print(f"Error with parameters lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}: {e}")
            continue

    print(f"\n=== Grid Search Complete ===")
    print(f"Best parameters: lr={best_params[0]}, weight_decay={best_params[1]}, batch_size={best_params[2]}")
    print(f"Best validation score (train NDCG@1000): {best_score:.6f}")
    
    # Update scoreboard with the best result
    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_per_label",
        dataset="test",
        metrics=best_test_metrics,
        n_samples=best_n_samples,
        epoch=None,  # No specific epoch since we're using grid search
    )

    print("\nFinal test metrics | ")
    for metric, value in best_test_metrics.items():
        print(f"  {metric}={value:.6f}")
    print(f"Best parameters: lr={best_params[0]}, weight_decay={best_params[1]}, batch_size={best_params[2]}")
    print("\nSaved best result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
