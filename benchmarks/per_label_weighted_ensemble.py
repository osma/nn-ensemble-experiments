from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

# Allow running as a script: `uv run benchmarks/per_label_weighted_ensemble.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.ndcg import load_csr, ndcg_at_k, update_markdown_scoreboard


class PerLabelWeightedEnsemble(nn.Module):
    """
    Per-label weighted ensemble with bias.

    For each label l:
        score[l] = sum_m w[m, l] * x[m, l] + b[l]

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
        self.weights = nn.Parameter(
            torch.full((n_models, n_labels), 1.0 / n_models)
        )

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

        weighted = x * self.weights.unsqueeze(0)
        out = weighted.sum(dim=1) + self.bias
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
    return torch.sqrt(torch.clamp(x, min=0.0))


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

    model = PerLabelWeightedEnsemble(
        n_models=n_models,
        n_labels=n_labels,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.001,
        eps=1e-8,
    )

    # Unweighted BCE performs best for NDCG in this setup
    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    # Early stopping disabled for now

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        model_name = f"torch_per_label_epoch{epoch:02d}"

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

        update_markdown_scoreboard(
            path=scoreboard_path,
            model=model_name,
            dataset="test",
            metrics=test_metrics,
            n_samples=n_used_test,
        )

        # Early stopping disabled

        print(
            f"Epoch {epoch:02d} | "
            f"Loss {loss.item():.6f}"
        )

    print("\nSaved per-epoch results to SCOREBOARD.md")


if __name__ == "__main__":
    main()
