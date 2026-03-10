import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

from benchmarks.datasets import DatasetConfig, get_dataset_config, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.metrics import (
    K_DEFAULT,
    f1_at_k_dense,
    load_csr,
    ndcg_at_k_dense,
)
from benchmarks.preprocessing import csr_to_log1p_tensor


class _Timer:
    def __init__(self):
        self.t0: float | None = None

    def start(self) -> None:
        self.t0 = torch.cuda.Event(enable_timing=True)
        self.t0.record()
        torch.cuda.synchronize()

    def stop(self) -> float:
        assert self.t0 is not None
        t1 = torch.cuda.Event(enable_timing=True)
        t1.record()
        torch.cuda.synchronize()
        return self.t0.elapsed_time(t1) / 1000.0


class NNSimpleLowRank(nn.Module):
    """
    Simple ensemble model with low-rank cross-label coupling.

    Architecture:
        - Global per-model weight (shared across labels)
        - Per-label residual weight (like torch_nn_simple)
        - Low-rank factorization for label correlations

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
        rank: int = 8,
        init_global: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_models = n_models
        self.n_labels = n_labels

        # Global per-model weight (shared across labels)
        self.global_weight = nn.Parameter(torch.zeros(n_models))

        # Per-label residual weight (like torch_nn_simple)
        self.label_weight = nn.Parameter(torch.zeros(n_labels))

        # Low-rank factorization for label correlations
        # U and V are (n_labels, rank), so U @ V.T gives (n_labels, n_labels)
        self.U = nn.Parameter(torch.randn(n_labels, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n_labels, rank) * 0.01)

        # Bias term
        self.bias = nn.Parameter(torch.zeros(n_labels))

        # Initialize global weight if provided
        if init_global is not None:
            with torch.no_grad():
                self.global_weight.copy_(init_global)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, M, L)
        batch_size = x.size(0)

        # Global weighted mean over models
        global_weights = torch.softmax(self.global_weight, dim=0)  # (M,)
        global_out = torch.einsum("bml,m->bl", x, global_weights)  # (batch, L)

        # Per-label residual weight
        # self.label_weight is per-label (L,), so broadcast over batch and model dims.
        label_out = (x * self.label_weight.view(1, 1, -1)).sum(dim=1)  # (batch, L)

        # Low-rank label coupling: U @ V.T applied to the global output
        lr_matrix = self.U @ self.V.T  # (L, L)
        lr_coupling = global_out @ lr_matrix.T  # (batch, L)

        # Combine everything
        logits = global_out + label_out + lr_coupling + self.bias  # (batch, L)
        return logits

    def effective_w(self) -> torch.Tensor:
        """Return the effective per-model weights (summed over labels)."""
        global_weights = torch.softmax(self.global_weight, dim=0)  # (M,)
        return global_weights

    def l2_reg(self) -> torch.Tensor:
        """L2 regularization on low-rank factors."""
        return (self.U ** 2).sum() + (self.V ** 2).sum()


def train_and_evaluate(
    *,
    dataset: str,
    ensemble_keys: tuple[str, str, str],
    lr: float = 0.01,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    rank: int = 8,
    max_epochs: int = 20,
    patience: int = 3,
    X_train: torch.Tensor | None = None,
    Y_train: csr_matrix | None = None,
    y_train_true: csr_matrix | None = None,
    X_val: torch.Tensor | None = None,
    Y_val: csr_matrix | None = None,
    y_val_true: csr_matrix | None = None,
    X_test: torch.Tensor | None = None,
    y_test_true: csr_matrix | None = None,
) -> dict:
    """Train and evaluate the model."""
    device = get_device()
    print(f"Using device: {device}")

    # Load data if not provided
    if X_train is None or Y_train is None or y_train_true is None:
        dataset_config = get_dataset_config(dataset)
        X_train, Y_train, y_train_true = _load_split(
            dataset_config, "train", ensemble_keys, device
        )
        # benchmarks.datasets.pred_path/truth_path accept only "train" or "test".
        # Use "train" for validation/early-stopping signals.
        X_val, Y_val, y_val_true = _load_split(
            dataset_config, "train", ensemble_keys, device
        )
        X_test, _, y_test_true = _load_split(
            dataset_config, "test", ensemble_keys, device
        )

    n_models, n_labels = X_train.shape[1], X_train.shape[2]
    print(f"Data shape: {X_train.shape} (train), {X_val.shape} (val), {X_test.shape} (test)")
    print(f"Labels: {n_labels}")

    # Initialize model
    # Use mean of training predictions as initial global weights
    with torch.no_grad():
        init_global = X_train.mean(dim=(0, 2))  # (M,)
        init_global = init_global / init_global.sum()  # normalize

    model = NNSimpleLowRank(
        n_models=n_models,
        n_labels=n_labels,
        rank=rank,
        init_global=init_global,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Training loop
    best_val_ndcg = -1.0
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    n_train_samples = X_train.shape[0]
    n_batches = (n_train_samples + batch_size - 1) // batch_size

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Shuffle training data
        perm = torch.randperm(n_train_samples, device=device)
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]

        for i in range(0, n_train_samples, batch_size):
            batch_X = X_shuffled[i : i + batch_size]
            batch_Y = Y_shuffled[i : i + batch_size]

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_Y)
            loss += weight_decay * model.l2_reg()  # Add L2 on low-rank factors
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            # ndcg_at_k_dense expects y_true as CSR; Y_val is a dense torch.Tensor.
            val_ndcg_10 = ndcg_at_k_dense(y_val_true, val_logits, k=10).item()
            val_ndcg_1000 = ndcg_at_k_dense(y_val_true, val_logits, k=1000).item()

        print(
            f"Epoch {epoch:2d}: loss={epoch_loss / n_batches:.4f}, "
            f"val NDCG@10={val_ndcg_10:.4f}, val NDCG@1000={val_ndcg_1000:.4f}"
        )

        # Early stopping check
        if val_ndcg_10 > best_val_ndcg:
            best_val_ndcg = val_ndcg_10
            best_epoch = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_ndcg_10 = ndcg_at_k_dense(y_test_true, test_logits, k=10).item()
        test_ndcg_1000 = ndcg_at_k_dense(y_test_true, test_logits, k=1000).item()
        test_f1_5 = f1_at_k_dense(y_test_true, test_logits, k=5).item()

    print(
        f"Test NDCG@10={test_ndcg_10:.4f}, NDCG@1000={test_ndcg_1000:.4f}, F1@5={test_f1_5:.4f}"
    )

    return {
        "epoch": best_epoch,
        "train_ndcg_10": -1.0,  # Not tracked in this simplified version
        "train_ndcg_1000": -1.0,
        "test_ndcg_10": test_ndcg_10,
        "test_ndcg_1000": test_ndcg_1000,
        "test_f1_5": test_f1_5,
    }


def _load_split(
    dataset_config: DatasetConfig,
    split: str,
    ensemble_keys: tuple[str, str, str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, csr_matrix]:
    """Load a data split and return (X, Y, y_true)."""
    # Load predictions
    preds = []
    for model_key in ensemble_keys:
        pred_path_i = pred_path(dataset_config.name, split, model_key)
        pred_csr = load_csr(str(pred_path_i))
        preds.append(pred_csr)

    # Stack into (n_samples, n_models, n_labels)
    M = len(ensemble_keys)
    n_samples = preds[0].shape[0]
    n_labels = preds[0].shape[1]
    X_np = np.zeros((n_samples, M, n_labels), dtype=np.float32)
    for i, pred_csr in enumerate(preds):
        X_np[:, i, :] = pred_csr.toarray()

    X = torch.from_numpy(X_np).to(device)

    # Preprocess: log1p
    # csr_to_log1p_tensor expects a 2D CSR matrix; our X is 3D (n_samples, M, L).
    # Apply log1p in torch directly to preserve shape.
    X = torch.log1p(torch.clamp_min(X, 0.0))

    # Load ground truth
    truth_path_i = truth_path(dataset_config.name, split)
    y_true = load_csr(str(truth_path_i))

    # Convert to dense tensor for training
    Y = torch.from_numpy(y_true.toarray()).float().to(device)

    return X, Y, y_true


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ensemble", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    # Determine ensemble keys
    if args.ensemble:
        ensemble_keys = tuple(args.ensemble.split(","))
    else:
        from benchmarks.datasets import ensemble3_keys

        ensemble_keys = ensemble3_keys(args.dataset)

    print(f"Dataset: {args.dataset}")
    print(f"Ensemble: {ensemble_keys}")

    results = train_and_evaluate(
        dataset=args.dataset,
        ensemble_keys=ensemble_keys,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        rank=args.rank,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    # Print results in scoreboard format
    print("\nResults:")
    print(f"epoch={results['epoch']}")
    print(f"train_ndcg_10={results['train_ndcg_10']:.6f}")
    print(f"train_ndcg_1000={results['train_ndcg_1000']:.6f}")
    print(f"test_ndcg_10={results['test_ndcg_10']:.6f}")
    print(f"test_ndcg_1000={results['test_ndcg_1000']:.6f}")
    print(f"test_f1_5={results['test_f1_5']:.6f}")


if __name__ == "__main__":
    main()
