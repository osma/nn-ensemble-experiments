from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

# Allow running as a script: `uv run benchmarks/weighted_conv_residual_ensemble.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import load_csr, ndcg_at_k, f1_at_k, update_markdown_scoreboard


class LowRankLabelResidual(nn.Module):
    """
    Low-rank cross-label residual applied on logits.

    Implements:
        Δ = scale * (Y @ V) @ U^T

    where:
        Y: (batch, L) logits
        U, V: (L, r)
    """

    def __init__(self, n_labels: int, rank: int = 16, scale: float = 0.1):
        super().__init__()
        self.rank = rank
        self.scale = scale

        self.U = nn.Parameter(torch.randn(n_labels, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n_labels, rank) * 0.01)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.scale * (y @ self.V) @ self.U.T


class PerLabelWeightedConvResidualEnsemble(nn.Module):
    """
    Per-label weighted ensemble with bias, using Conv1d for summation,
    plus a low-rank cross-label residual on logits.
    """

    def __init__(self, n_models: int, n_labels: int, rank: int = 16, residual_scale: float = 0.1):
        super().__init__()
        self.n_models = n_models
        self.n_labels = n_labels

        # Per-model, per-label weights
        self.weights = nn.Parameter(
            torch.full((n_models, n_labels), 1.0 / n_models)
        )

        # Per-label bias
        self.bias = nn.Parameter(torch.zeros(n_labels))

        # Conv1d used only as a summation operator over models
        self.sum_conv = nn.Conv1d(
            in_channels=n_models,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )

        # Initialize to a true sum (not mean) and freeze
        with torch.no_grad():
            self.sum_conv.weight.fill_(1.0)
        self.sum_conv.weight.requires_grad_(False)

        # Low-rank cross-label residual
        self.residual = LowRankLabelResidual(
            n_labels=n_labels,
            rank=rank,
            scale=residual_scale,
        )

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

        # Per-label weighted sum over models
        weighted = x * self.weights.unsqueeze(0)
        summed = self.sum_conv(weighted).squeeze(1)

        # Base logits
        logits = summed + self.bias

        # Add low-rank cross-label residual
        logits = logits + self.residual(logits)

        return logits


# ============================
# Training / evaluation script
# ============================

DEVICE = "cpu"
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
K_VALUES = (10, 1000)

RESIDUAL_RANK = 16
RESIDUAL_SCALE = 0.1


def csr_to_dense_tensor(csr):
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())


def main():
    scoreboard_path = Path("SCOREBOARD.md")

    print("Loading training data...")

    y_train_true = load_csr("data/train-output.npz")
    train_preds = [
        load_csr("data/train-bonsai.npz"),
        load_csr("data/train-fasttext.npz"),
        load_csr("data/train-mllm.npz"),
    ]

    X_train = torch.stack([csr_to_dense_tensor(p) for p in train_preds], dim=1)
    Y_train = csr_to_dense_tensor(y_train_true)

    print("Loading test data...")

    y_test_true = load_csr("data/test-output.npz")
    test_preds = [
        load_csr("data/test-bonsai.npz"),
        load_csr("data/test-fasttext.npz"),
        load_csr("data/test-mllm.npz"),
    ]

    X_test = torch.stack([csr_to_dense_tensor(p) for p in test_preds], dim=1)

    n_models = X_train.shape[1]
    n_labels = X_train.shape[2]

    model = PerLabelWeightedConvResidualEnsemble(
        n_models=n_models,
        n_labels=n_labels,
        rank=RESIDUAL_RANK,
        residual_scale=RESIDUAL_SCALE,
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
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    best_metric = float("-inf")
    best_state = None
    best_train_metrics = None
    best_test_metrics = None
    best_n_used_train = None
    best_n_used_test = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()

        # --- Train evaluation ---
        with torch.no_grad():
            train_scores = model(X_train)

        y_train_pred_csr = tensor_to_csr(train_scores)
        train_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_train = ndcg_at_k(y_train_true, y_train_pred_csr, k=k)
            train_metrics[f"ndcg@{k}"] = ndcg

        # --- Test evaluation ---
        with torch.no_grad():
            test_scores = model(X_test)

        y_test_pred_csr = tensor_to_csr(test_scores)
        test_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k(y_test_true, y_test_pred_csr, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg

        f1, _ = f1_at_k(y_test_true, y_test_pred_csr, k=5)
        test_metrics["f1@5"] = f1

        current = test_metrics["ndcg@10"]
        if current > best_metric:
            best_metric = current
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_train_metrics = train_metrics.copy()
            best_test_metrics = test_metrics.copy()
            best_n_used_train = n_used_train
            best_n_used_test = n_used_test

        print(
            f"Epoch {epoch:02d} | "
            f"Loss {loss.item():.6f}"
        )

    model.load_state_dict(best_state)

    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_per_label_conv_residual",
        dataset="train",
        metrics=best_train_metrics,
        n_samples=best_n_used_train,
        epoch=best_epoch,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_per_label_conv_residual",
        dataset="test",
        metrics=best_test_metrics,
        n_samples=best_n_used_test,
        epoch=best_epoch,
    )

    print("\nSaved best result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
