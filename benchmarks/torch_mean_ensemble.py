from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

from .ndcg import load_csr, ndcg_at_k, update_markdown_scoreboard

DEVICE = "cpu"
EPOCHS = 10
LR = 1e-2
K_VALUES = (10, 1000)


class MeanWeightedConv1D(nn.Module):
    """
    Input:  (batch, 3, L)
    Output: (batch, L)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=3,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x).squeeze(1)
        return torch.clamp(out, min=0.0, max=1.0)


def csr_to_dense_tensor(csr):
    return torch.from_numpy(csr.toarray()).float()


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

    model = MeanWeightedConv1D().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    print("Starting training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        output_train = model(X_train)
        loss = criterion(output_train, Y_train)
        loss.backward()
        optimizer.step()

        model.eval()

        model_name = f"torch_mean_epoch{epoch:02d}"

        # --- Train evaluation ---
        y_train_pred_csr = tensor_to_csr(output_train)
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
            output_test = model(X_test)

        y_test_pred_csr = tensor_to_csr(output_test)
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
