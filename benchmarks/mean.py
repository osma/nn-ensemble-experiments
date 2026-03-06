# STATUS: BASELINE
# Purpose: Simple non-torch mean ensemble baseline.
from pathlib import Path
import sys

# Allow running as a script: `uv run benchmarks/mean.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
from scipy.sparse import csr_matrix

from benchmarks.datasets import ensemble3_keys, pred_path, truth_path
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="yso-fi",
        choices=["yso-fi", "yso-en", "koko"],
        help="Dataset to benchmark",
    )
    args = parser.parse_args()
    dataset = str(args.dataset)

    e3 = ensemble3_keys(dataset)
    model_name = f"mean({','.join(e3)})"

    scoreboard_path = Path("SCOREBOARD.md")

    for split in ("train", "test"):
        print(f"\n=== {dataset} | {split.upper()} (MEAN ENSEMBLE) ===")

        y_true = load_csr(str(truth_path(dataset, split)))
        preds = [load_csr(str(pred_path(dataset, split, k))) for k in e3]

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
            model=model_name,
            dataset=dataset,
            split=split,
            metrics=metrics,
            n_samples=n_used,
        )

    print(f"\nSaved results to {scoreboard_path}")


if __name__ == "__main__":
    main()
