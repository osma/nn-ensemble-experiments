# STATUS: EXPERIMENTAL (documented negative result)
# Purpose: Post-hoc low-rank residual on frozen logits; preserved for reference.
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

# Allow running as a script: `uv run benchmarks/posthoc_lowrank_residual.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import load_csr, ndcg_at_k, f1_at_k, update_markdown_scoreboard
from benchmarks.torch_per_label_conv import PerLabelWeightedConvEnsemble


class LowRankResidual(nn.Module):
    """
    Very small low-rank residual trained post-hoc on frozen logits.

    Δ = scale * (Y @ V) @ Uᵀ

    IMPORTANT:
    - Input logits must be detached
    - This model never sees gradients from the base ensemble
    """

    def __init__(self, n_labels: int, rank: int = 8, scale: float = 0.03):
        super().__init__()
        self.scale = scale
        self.U = nn.Parameter(torch.randn(n_labels, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n_labels, rank) * 0.01)

    def forward(self, y_detached: torch.Tensor) -> torch.Tensor:
        return self.scale * (y_detached @ self.V) @ self.U.T


DEVICE = "cpu"
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
K_VALUES = (10, 1000)

BASE_EPOCH = 3
RESIDUAL_RANK = 8
RESIDUAL_SCALE = 0.03


def csr_to_dense_tensor(csr):
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())


def main():
    scoreboard_path = Path("SCOREBOARD.md")

    print("Loading data...")

    y_train_true = load_csr("data/train-output.npz")
    y_test_true = load_csr("data/test-output.npz")

    train_preds = [
        load_csr("data/train-bonsai.npz"),
        load_csr("data/train-fasttext.npz"),
        load_csr("data/train-mllm.npz"),
    ]
    test_preds = [
        load_csr("data/test-bonsai.npz"),
        load_csr("data/test-fasttext.npz"),
        load_csr("data/test-mllm.npz"),
    ]

    X_train = torch.stack([csr_to_dense_tensor(p) for p in train_preds], dim=1)
    X_test = torch.stack([csr_to_dense_tensor(p) for p in test_preds], dim=1)

    Y_train = csr_to_dense_tensor(y_train_true)

    n_models = X_train.shape[1]
    n_labels = X_train.shape[2]

    print("Training base model (frozen afterwards)...")

    base = PerLabelWeightedConvEnsemble(
        n_models=n_models,
        n_labels=n_labels,
    ).to(DEVICE)

    optimizer = optim.AdamW(base.parameters(), lr=LR, weight_decay=0.001)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in range(1, BASE_EPOCH + 1):
        base.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = base(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        print(f"Base epoch {epoch:02d} | Loss {loss.item():.6f}")

    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        base_train_logits = base(X_train)
        base_test_logits = base(X_test)

    print("Training post-hoc residual...")

    residual = LowRankResidual(
        n_labels=n_labels,
        rank=RESIDUAL_RANK,
        scale=RESIDUAL_SCALE,
    ).to(DEVICE)

    optimizer = optim.AdamW(residual.parameters(), lr=LR, weight_decay=0.01)

    target = Y_train - torch.sigmoid(base_train_logits)

    res_ds = torch.utils.data.TensorDataset(
        base_train_logits.detach(), target
    )
    res_loader = torch.utils.data.DataLoader(
        res_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    mse = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        residual.train()
        for xb, yb in res_loader:
            optimizer.zero_grad()
            delta = residual(xb)
            loss = mse(delta, yb)
            loss.backward()
            optimizer.step()
        print(f"Residual epoch {epoch:02d} | Loss {loss.item():.6f}")

    print("Evaluating combined model...")

    with torch.no_grad():
        train_scores = base_train_logits + residual(base_train_logits.detach())
        test_scores = base_test_logits + residual(base_test_logits.detach())

    for split, scores, y_true in (
        ("train", train_scores, y_train_true),
        ("test", test_scores, y_test_true),
    ):
        y_pred = tensor_to_csr(scores)
        metrics = {}
        for k in K_VALUES:
            ndcg, n_used = ndcg_at_k(y_true, y_pred, k=k)
            metrics[f"ndcg@{k}"] = ndcg
        f1, _ = f1_at_k(y_true, y_pred, k=5)
        metrics["f1@5"] = f1

        update_markdown_scoreboard(
            path=scoreboard_path,
            model="torch_per_label_conv_posthoc_residual",
            dataset=split,
            metrics=metrics,
            n_samples=n_used,
            epoch=BASE_EPOCH,
        )

    print("\nSaved results to SCOREBOARD.md")


if __name__ == "__main__":
    main()
