from pathlib import Path
import sys

# Allow running as a script: `uv run benchmarks/torch_mean.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.device import get_device
from benchmarks.preprocessing import csr_to_log1p_tensor, tensor_to_csr
from benchmarks.models.torch_mean import MeanWeightedConv1D
from benchmarks.metrics import load_csr, ndcg_at_k, f1_at_k, update_markdown_scoreboard

DEVICE = get_device()
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
K_VALUES = (10, 1000)

PATIENCE = 2
MIN_EPOCHS = 2


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

    X_train = torch.stack([csr_to_log1p_tensor(p) for p in train_preds], dim=1)
    Y_train = torch.from_numpy(y_train_true.toarray()).float()

    print("Loading test data...")

    y_test_true = load_csr("data/test-output.npz")
    test_preds = [
        load_csr("data/test-bonsai.npz"),
        load_csr("data/test-fasttext.npz"),
        load_csr("data/test-mllm.npz"),
    ]

    X_test = torch.stack([csr_to_log1p_tensor(p) for p in test_preds], dim=1)

    X_train = X_train.to(DEVICE)
    Y_train = Y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)

    model = MeanWeightedConv1D().to(DEVICE)
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
        train_ds, batch_size=BATCH_SIZE, shuffle=True
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
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            output_train = model(xb)
            loss = criterion(output_train, yb)
            loss.backward()
            optimizer.step()

        model.eval()

        # --- Train evaluation ---
        with torch.no_grad():
            full_train_output = model(X_train)

        y_train_pred_csr = tensor_to_csr(full_train_output)
        train_metrics = {}
        for k in K_VALUES:
            ndcg, n_used_train = ndcg_at_k(y_train_true, y_train_pred_csr, k=k)
            train_metrics[f"ndcg@{k}"] = ndcg

        # --- Test evaluation (computed for reporting only; NOT used for selection) ---
        with torch.no_grad():
            output_test = model(X_test)

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
        model="torch_mean",
        dataset="train",
        metrics=best_train_metrics,
        n_samples=best_n_used_train,
        epoch=best_epoch,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model="torch_mean",
        dataset="test",
        metrics=best_test_metrics,
        n_samples=best_n_used_test,
        epoch=best_epoch,
    )

    print("\nSaved best result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
