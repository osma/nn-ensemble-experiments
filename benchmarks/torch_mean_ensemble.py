from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

from .ndcg import load_csr, ndcg_at_k, update_markdown_scoreboard

DEVICE = "cpu"
EPOCHS = 20
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
        return self.conv(x).squeeze(1)


def csr_to_dense_tensor(csr):
    return torch.from_numpy(csr.toarray()).float()


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())


def main():
    scoreboard_path = Path("SCOREBOARD.md")

    print("Loading training data...")

    y_true = load_csr("data/train-output.npz")
    preds = [
        load_csr("data/train-bonsai.npz"),
        load_csr("data/train-fasttext.npz"),
        load_csr("data/train-mllm.npz"),
    ]

    X = torch.stack([csr_to_dense_tensor(p) for p in preds], dim=1)
    Y = csr_to_dense_tensor(y_true)

    model = MeanWeightedConv1D().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("Starting training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

        model.eval()
        y_pred_csr = tensor_to_csr(output)

        metrics = {}
        for k in K_VALUES:
            ndcg, n_used = ndcg_at_k(y_true, y_pred_csr, k=k)
            metrics[f"ndcg@{k}"] = ndcg

        model_name = f"torch_mean_epoch_{epoch}"

        update_markdown_scoreboard(
            path=scoreboard_path,
            model=model_name,
            dataset="train",
            metrics=metrics,
            n_samples=n_used,
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
