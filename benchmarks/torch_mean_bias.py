from pathlib import Path
import sys

# Allow running as a script: `uv run benchmarks/torch_mean_bias.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

from benchmarks.device import get_device
from benchmarks.metrics import load_csr, ndcg_at_k, f1_at_k, update_markdown_scoreboard

DEVICE = get_device()
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
K_VALUES = (10, 1000)

PATIENCE = 2
MIN_EPOCHS = 2

EVAL_BATCH_SIZE = 128


class MeanWeightedConv1D(nn.Module):
    """
    Input:  (batch, 3, L)
    Output: (batch, L)
    """

    def __init__(self, n_labels: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=3,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, n_labels))
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.conv(x).squeeze(1)
        delta = self.bias
        out = mean + delta
        return torch.clamp(out, min=0.0, max=1.0)


def csr_to_dense_tensor(csr):
    x = torch.from_numpy(csr.toarray()).float()
    return torch.clamp(x, min=0.0)


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
            out = model(xb)
            outs.append(out.detach().cpu())
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
    X_train = torch.log1p(X_train)

    # Keep Y_train on CPU (requested).
    Y_train = csr_to_dense_tensor(y_train_true)

    print("Loading test data...")

    y_test_true = load_csr("data/test-output.npz")
    test_preds = [
        load_csr("data/test-bonsai.npz"),
        load_csr("data/test-fasttext.npz"),
        load_csr("data/test-mllm.npz"),
    ]

    # Keep X_test on CPU; move to GPU only for evaluation forward pass.
    X_test = torch.stack([csr_to_dense_tensor(p) for p in test_preds], dim=1)
    X_test = torch.log1p(X_test)

    n_labels = X_train.shape[2]
    model = MeanWeightedConv1D(n_labels).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
        eps=1e-8,
    )
    criterion = nn.BCELoss()

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
            output_train = model(xb)
            loss = criterion(output_train, yb)
            loss.backward()
            optimizer.step()

        # --- Train evaluation (batched) ---
        full_train_output = _predict_in_batches(model, X_train)
        y_train_pred_csr = tensor_to_csr(full_train_output)
        train_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_train = ndcg_at_k(y_train_true, y_train_pred_csr, k=k)
            train_metrics[f"ndcg@{k}"] = ndcg

        # --- Test evaluation (batched; reporting only) ---
        output_test = _predict_in_batches(model, X_test)
        y_test_pred_csr = tensor_to_csr(output_test)
        test_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k(y_test_true, y_test_pred_csr, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg

        f1, _ = f1_at_k(y_test_true, y_test_pred_csr, k=5)
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

    model.load_state_dict(best_state)

    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_mean_bias",
        dataset="train",
        metrics=best_train_metrics,
        n_samples=best_n_used_train,
        epoch=best_epoch,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_mean_bias",
        dataset="test",
        metrics=best_test_metrics,
        n_samples=best_n_used_test,
        epoch=best_epoch,
    )

    print("\nSaved best result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
