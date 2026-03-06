# STATUS: BASELINE
# Purpose: Individual base model benchmarks (no learning).
from pathlib import Path

import argparse

from .datasets import all_pred_keys, pred_path, truth_path
from .metrics import load_csr, ndcg_at_k, f1_at_k, update_markdown_scoreboard

K = 1000


def _maybe_load(path: Path):
    if not path.exists():
        return None
    return load_csr(str(path))


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
    scoreboard_path = Path("SCOREBOARD.md")

    for split in ("train", "test"):
        y_true = load_csr(str(truth_path(dataset, split)))
        print(f"\n=== {dataset} | {split.upper()} ===")
        print("Ground truth shape:", y_true.shape)

        for model_key in all_pred_keys(dataset, split):
            p = pred_path(dataset, split, model_key)
            y_pred = _maybe_load(p)
            if y_pred is None:
                continue

            print(f"\nModel: {model_key}")
            print("Predictions shape:", y_pred.shape)

            metrics = {}
            for k in (10, 1000):
                ndcg, n_used = ndcg_at_k(y_true, y_pred, k=k)
                print(f"NDCG@{k} = {ndcg:.6f} (computed over {n_used} samples)")
                metrics[f"ndcg@{k}"] = ndcg

            f1, _ = f1_at_k(y_true, y_pred, k=5)
            metrics["f1@5"] = f1

            update_markdown_scoreboard(
                path=scoreboard_path,
                model=model_key,
                dataset=dataset,
                split=split,
                metrics=metrics,
                n_samples=n_used,
            )

    print(f"\nSaved results to {scoreboard_path}")


if __name__ == "__main__":
    main()
