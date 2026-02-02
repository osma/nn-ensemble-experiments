from pathlib import Path

from benchmarks.ndcg import load_csr, ndcg_at_k, update_markdown_scoreboard

K = 1000


def main():
    y_true = load_csr("data/test-output.npz")
    print("Ground truth shape:", y_true.shape)

    models = {
        "mllm": "data/test-mllm.npz",
        "bonsai": "data/test-bonsai.npz",
        "fasttext": "data/test-fasttext.npz",
        "nn": "data/test-nn.npz",
    }

    scoreboard_path = Path("SCOREBOARD.md")

    for model, path in models.items():
        y_pred = load_csr(path)

        print(f"\nModel: {model}")
        print("Predictions shape:", y_pred.shape)

        ndcg, n_used = ndcg_at_k(y_true, y_pred, k=K)
        print(f"NDCG@{K} = {ndcg:.6f} (computed over {n_used} samples)")

        update_markdown_scoreboard(
            path=scoreboard_path,
            model=model,
            dataset="test",
            ndcg=ndcg,
            n_samples=n_used,
        )

    print(f"\nSaved results to {scoreboard_path}")


if __name__ == "__main__":
    main()
