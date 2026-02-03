from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

# Allow running as a script: `uv run benchmarks/per_label_weighted_conv_ensemble.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import load_csr, ndcg_at_k, f1_at_k, update_markdown_scoreboard


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

DEVICE = "cpu"
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
K_VALUES = (10, 1000)


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
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        model_name = f"torch_per_label_conv_epoch{epoch:02d}"

        # --- Train evaluation ---
        with torch.no_grad():
            train_scores = model(X_train)

        y_train_pred_csr = tensor_to_csr(train_scores)
        train_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_train = ndcg_at_k(y_train_true, y_train_pred_csr, k=k)
            train_metrics[f"ndcg@{k}"] = ndcg

        update_markdown_scoreboard(
            path=scoreboard_path,
            model=model_name,
            dataset="train",
            metrics=train_metrics,
            n_samples=n_used_train,
        )

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

        update_markdown_scoreboard(
            path=scoreboard_path,
            model=model_name,
            dataset="test",
            metrics=test_metrics,
            n_samples=n_used_test,
        )

        weights = model.conv.weight.detach().cpu().numpy().reshape(-1)
        print(
            f"Epoch {epoch:02d} | "
            f"Loss {loss.item():.6f} | "
            f"Weights: "
            f"bonsai={weights[0]:.3f}, "
            f"fasttext={weights[1]:.3f}, "
            f"mllm={weights[2]:.3f}"
        )

    print("\nSaved per-epoch results to SCOREBOARD.md")


if __name__ == "__main__":
    main()
