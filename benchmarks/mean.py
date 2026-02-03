# STATUS: BASELINE
# Purpose: Simple non-torch mean ensemble baseline.
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

# Allow running as a script: `uv run benchmarks/mean_ensemble.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import load_csr, ndcg_at_k, f1_at_k, update_markdown_scoreboard

K = 1000


def mean_ensemble(matrices: list[csr_matrix]) -> csr_matrix:
    """
    Compute the elementwise mean of multiple CSR prediction matrices.
    """
    if not matrices:
        raise ValueError("No matrices provided for mean ensemble")

    summed = matrices[0].astype(np.float64).copy()
    for m in matrices[1:]:
        summed += m

    summed /= len(matrices)
    return summed


def main():
    splits = {
        "train": {
            "truth": "data/train-output.npz",
            "models": {
                "bonsai": "data/train-bonsai.npz",
                "fasttext": "data/train-fasttext.npz",
                "mllm": "data/train-mllm.npz",
            },
        },
        "test": {
            "truth": "data/test-output.npz",
            "models": {
                "bonsai": "data/test-bonsai.npz",
                "fasttext": "data/test-fasttext.npz",
                "mllm": "data/test-mllm.npz",
            },
        },
    }

    scoreboard_path = Path("SCOREBOARD.md")

    for split, cfg in splits.items():
        print(f"\n=== {split.upper()} (MEAN ENSEMBLE) ===")

        y_true = load_csr(cfg["truth"])
        preds = [load_csr(p) for p in cfg["models"].values()]

        print("Ground truth shape:", y_true.shape)

        y_mean = mean_ensemble(preds)

        metrics = {}
        for k in (10, 1000):
            ndcg, n_used = ndcg_at_k(y_true, y_mean, k=k)
            print(f"NDCG@{k} = {ndcg:.6f} (computed over {n_used} samples)")
            metrics[f"ndcg@{k}"] = ndcg

        f1, _ = f1_at_k(y_true, y_mean, k=5)
        metrics["f1@5"] = f1

        update_markdown_scoreboard(
            path=scoreboard_path,
            model="mean",
            dataset=split,
            metrics=metrics,
            n_samples=n_used,
        )

    print(f"\nSaved results to {scoreboard_path}")


if __name__ == "__main__":
    main()
