from pathlib import Path
import sys

import torch
import numpy as np
from scipy.sparse import csr_matrix

# Allow running as a script: `uv run benchmarks/epoch_ensemble.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import load_csr, ndcg_at_k, f1_at_k, update_markdown_scoreboard


def csr_to_dense_tensor(csr: csr_matrix) -> torch.Tensor:
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())


def main():
    """
    Epoch ensembling for PerLabelWeightedConvEnsemble.

    This script assumes that multiple checkpoints (epochs) of the SAME model
    have already been trained and recorded in SCOREBOARD.md, and that we can
    reproduce their logits by re-running training up to the desired epochs.

    We explicitly ensemble early epochs (e.g. 2 and 3), which are known to have
    complementary ranking properties:
    - earlier epochs: sharper rankings
    - slightly later epochs: better calibration

    The ensemble is a simple convex combination of logits:
        logits = α * logits_epoch_A + (1 - α) * logits_epoch_B
    """

    DEVICE = "cpu"
    EPOCHS = 3
    LR = 1e-3
    BATCH_SIZE = 32

    # Epochs to ensemble (must be <= EPOCHS)
    EPOCH_A = 2
    EPOCH_B = 3

    # Mixing coefficient
    ALPHA = 0.5

    K_VALUES = (10, 1000)

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

    from benchmarks.torch_per_label_conv import PerLabelWeightedConvEnsemble

    n_models = X_train.shape[1]
    n_labels = X_train.shape[2]

    model = PerLabelWeightedConvEnsemble(
        n_models=n_models,
        n_labels=n_labels,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.001,
        eps=1e-8,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    logits_epoch = {}

    print("Training and capturing logits...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if epoch in (EPOCH_A, EPOCH_B):
            model.eval()
            with torch.no_grad():
                logits_epoch[epoch] = {
                    "train": model(X_train).clone(),
                    "test": model(X_test).clone(),
                }
            print(f"Captured logits for epoch {epoch:02d}")

    print("Ensembling epochs...")

    train_logits = (
        ALPHA * logits_epoch[EPOCH_A]["train"]
        + (1.0 - ALPHA) * logits_epoch[EPOCH_B]["train"]
    )
    test_logits = (
        ALPHA * logits_epoch[EPOCH_A]["test"]
        + (1.0 - ALPHA) * logits_epoch[EPOCH_B]["test"]
    )

    for split, scores, y_true in (
        ("train", train_logits, y_train_true),
        ("test", test_logits, y_test_true),
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
            model=f"torch_per_label_conv_epoch{EPOCH_A:02d}_{EPOCH_B:02d}",
            dataset=split,
            metrics=metrics,
            n_samples=n_used,
        )

    print("\nSaved epoch ensemble results to SCOREBOARD.md")


if __name__ == "__main__":
    main()
