# STATUS: ACTIVE (alternative base model)
# Purpose: Conv1d-based summation variant of per-label linear ensemble.
from pathlib import Path
import sys

# Allow running as a script: `uv run benchmarks/torch_per_label_conv.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


class PerLabelWeightedConvEnsemble(nn.Module):
    """
    Per-label weighted ensemble with bias, using Conv1d for summation.

    For each label l:
        score[l] = sum_m w[m, l] * x[m, l] + b[l]

    Notes:
    - Conv1d(kernel_size=1) is used purely to implement the sum over models.
    - Weights are per-model, per-label (same expressiveness as PerLabelWeightedEnsemble).
    - Inputs are assumed to be preprocessed with log1p.
    - Returns raw logits.
    """

    def __init__(self, n_models: int, n_labels: int):
        super().__init__()
        self.n_models = n_models
        self.n_labels = n_labels

        # Per-model, per-label weights (start as identity; scaling handled by conv)
        self.weights = nn.Parameter(torch.ones((n_models, n_labels)))

        # Per-label bias
        self.bias = nn.Parameter(torch.zeros(n_labels))

        # Conv1d used only as a summation operator over models
        self.sum_conv = nn.Conv1d(
            in_channels=n_models,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )

        # Initialize to a mean (not sum) and freeze
        with torch.no_grad():
            self.sum_conv.weight.fill_(1.0 / n_models)
        self.sum_conv.weight.requires_grad_(False)

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

        # Apply per-label weights
        weighted = x * self.weights.unsqueeze(0)

        # Sum over models via Conv1d
        summed = self.sum_conv(weighted).squeeze(1)

        # Add per-label bias
        out = summed + self.bias
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

EVAL_BATCH_SIZE = 128


def csr_to_dense_tensor(csr):
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())


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

    model = PerLabelWeightedConvEnsemble(
        n_models=n_models,
        n_labels=n_labels,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.001,
        eps=1e-8,
    )

    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Early stopping: select best epoch by TRAIN NDCG@1000 (no test leakage)
    best_metric = float("-inf")
    best_epoch = None
    best_state = None
    best_train_metrics = None
    best_test_metrics = None
    best_n_used_train = None
    best_n_used_test = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # --- Train evaluation (batched; no CSR conversion) ---
        train_scores = _predict_in_batches(model, X_train)
        train_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_train = ndcg_at_k_dense(y_train_true, train_scores, k=k)
            train_metrics[f"ndcg@{k}"] = ndcg

        # --- Test evaluation (batched; no CSR conversion) ---
        test_scores = _predict_in_batches(model, X_test)
        test_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg

        f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
        test_metrics["f1@5"] = f1

        current = train_metrics["ndcg@1000"]
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_train_metrics = train_metrics.copy()
            best_test_metrics = test_metrics.copy()
            best_n_used_train = n_used_train
            best_n_used_test = n_used_test
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch >= MIN_EPOCHS and epochs_no_improve >= PATIENCE:
            break

        # Log mean per-model weights (averaged over labels) and conv1d weights
        mean_weights = model.weights.detach().cpu().mean(dim=1).numpy()
        conv_weights = model.sum_conv.weight.detach().cpu().view(-1).numpy()

        print(
            f"Epoch {epoch:02d} | "
            f"Loss {loss.item():.6f} | "
            f"Weights: "
            f"bonsai={mean_weights[0]:.3f}*{conv_weights[0]:.3f}, "
            f"fasttext={mean_weights[1]:.3f}*{conv_weights[1]:.3f}, "
            f"mllm={mean_weights[2]:.3f}*{conv_weights[2]:.3f}"
        )

    model.load_state_dict(best_state)

    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_per_label_conv",
        dataset="train",
        metrics=best_train_metrics,
        n_samples=best_n_used_train,
        epoch=best_epoch,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_per_label_conv",
        dataset="test",
        metrics=best_test_metrics,
        n_samples=best_n_used_test,
        epoch=best_epoch,
    )

    print("\nSaved best result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
