from pathlib import Path

from .ndcg import load_csr, ndcg_at_k, update_markdown_scoreboard

K = 1000


def main():
    splits = {
        "train": {
            "truth": "data/train-output.npz",
            "models": {
                "mllm": "data/train-mllm.npz",
                "bonsai": "data/train-bonsai.npz",
                "fasttext": "data/train-fasttext.npz",
                "nn": "data/train-nn.npz",
            },
        },
        "test": {
            "truth": "data/test-output.npz",
            "models": {
                "mllm": "data/test-mllm.npz",
                "bonsai": "data/test-bonsai.npz",
                "fasttext": "data/test-fasttext.npz",
                "nn": "data/test-nn.npz",
            },
        },
    }

    scoreboard_path = Path("SCOREBOARD.md")

    for split, cfg in splits.items():
        y_true = load_csr(cfg["truth"])
        print(f"\n=== {split.upper()} ===")
        print("Ground truth shape:", y_true.shape)

        for model, path in cfg["models"].items():
            y_pred = load_csr(path)

            print(f"\nModel: {model}")
            print("Predictions shape:", y_pred.shape)

            metrics = {}
            for k in (10, 1000):
                ndcg, n_used = ndcg_at_k(y_true, y_pred, k=k)
                print(f"NDCG@{k} = {ndcg:.6f} (computed over {n_used} samples)")
                metrics[f"ndcg@{k}"] = ndcg

            update_markdown_scoreboard(
                path=scoreboard_path,
                model=model,
                dataset=split,
                metrics=metrics,
                n_samples=n_used,
            )

    print(f"\nSaved results to {scoreboard_path}")


if __name__ == "__main__":
    main()
