from pathlib import Path
import itertools

import numpy as np
from scipy.sparse import csr_matrix

from .ndcg import load_csr, ndcg_at_k, update_markdown_scoreboard

K = 1000


def weighted_mean_ensemble(matrices: dict[str, csr_matrix], weights: dict[str, float]) -> csr_matrix:
    """
    Compute a weighted mean ensemble of CSR prediction matrices.
    """
    items = list(matrices.items())
    name0, mat0 = items[0]

    result = mat0.astype(np.float64).copy()
    result.data *= weights[name0]

    for name, mat in items[1:]:
        result += mat * weights[name]

    return result


def generate_weight_grid(step: float = 0.1):
    """
    Generate weight combinations for three models that sum to 1.
    """
    grid = []
    values = np.arange(0.0, 1.0 + 1e-9, step)

    for w1, w2 in itertools.product(values, repeat=2):
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1:
            continue
        grid.append((w1, w2, w3))

    return grid


def main():
    models = ["bonsai", "fasttext", "mllm"]

    train_preds = {
        "bonsai": load_csr("data/train-bonsai.npz"),
        "fasttext": load_csr("data/train-fasttext.npz"),
        "mllm": load_csr("data/train-mllm.npz"),
    }
    train_true = load_csr("data/train-output.npz")

    print("=== GRID SEARCH (TRAIN, NDCG@1000) ===")
    print("Ground truth shape:", train_true.shape)

    best_score = -1.0
    best_weights: dict[str, float] | None = None

    for w1, w2, w3 in generate_weight_grid(step=0.1):
        weights = {
            "bonsai": w1,
            "fasttext": w2,
            "mllm": w3,
        }

        ensemble = weighted_mean_ensemble(train_preds, weights)
        score, _ = ndcg_at_k(train_true, ensemble, k=K)

        if score > best_score:
            best_score = score
            best_weights = weights

    assert best_weights is not None

    print("Best train NDCG@1000:", f"{best_score:.6f}")
    print("Best weights:", best_weights)

    scoreboard_path = Path("SCOREBOARD.md")

    # Evaluate with fixed weights on both train and test
    for split in ("train", "test"):
        print(f"\n=== {split.upper()} (WEIGHTED MEAN) ===")

        y_true = load_csr(f"data/{split}-output.npz")
        preds = {
            "bonsai": load_csr(f"data/{split}-bonsai.npz"),
            "fasttext": load_csr(f"data/{split}-fasttext.npz"),
            "mllm": load_csr(f"data/{split}-mllm.npz"),
        }

        ensemble = weighted_mean_ensemble(preds, best_weights)

        metrics = {}
        for k in (10, 1000):
            ndcg, n_used = ndcg_at_k(y_true, ensemble, k=k)
            print(f"NDCG@{k} = {ndcg:.6f} (computed over {n_used} samples)")
            metrics[f"ndcg@{k}"] = ndcg

        update_markdown_scoreboard(
            path=scoreboard_path,
            model="mean_weighted",
            dataset=split,
            metrics=metrics,
            n_samples=n_used,
        )

    print(f"\nSaved results to {scoreboard_path}")
    print("Final weights:", best_weights)


if __name__ == "__main__":
    main()
