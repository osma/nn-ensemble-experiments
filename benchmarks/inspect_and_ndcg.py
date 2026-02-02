import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path

K = 1000


def load_csr(path: str) -> csr_matrix:
    """Load a CSR matrix stored in NPZ format."""
    data = np.load(path)
    return csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"]),
    )


def ndcg_at_k(y_true: csr_matrix, y_pred: csr_matrix, k: int = K):
    """
    Compute mean NDCG@k for multilabel classification using sparse matrices.

    Rows with zero true labels are skipped.
    """
    ndcgs = []

    n_samples = y_true.shape[0]
    for i in range(n_samples):
        true_row = y_true.getrow(i)
        if true_row.nnz == 0:
            continue

        pred_row = y_pred.getrow(i)

        # Select top-k predicted labels efficiently from sparse data
        if pred_row.nnz > k:
            topk_pos = np.argpartition(pred_row.data, -k)[-k:]
            top_indices = pred_row.indices[topk_pos]
            top_scores = pred_row.data[topk_pos]
            order = np.argsort(top_scores)[::-1]
            ranked = top_indices[order]
        else:
            order = np.argsort(pred_row.data)[::-1]
            ranked = pred_row.indices[order]

        # DCG
        rel = np.isin(ranked, true_row.indices).astype(float)
        discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
        dcg = np.sum(rel * discounts)

        # IDCG
        ideal_len = min(true_row.nnz, k)
        ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2))
        idcg = np.sum(ideal_discounts)

        ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)), len(ndcgs)


def update_markdown_scoreboard(
    path: Path,
    model: str,
    dataset: str,
    ndcg: float,
    n_samples: int,
):
    header = [
        "# Benchmark Scoreboard\n\n",
        "## NDCG@1000\n\n",
        "| Model | Dataset | NDCG@1000 | Samples |\n",
        "|-------|---------|-----------|---------|\n",
    ]

    rows = {}

    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("|") and "Model" not in line:
                cols = [c.strip() for c in line.strip("|").split("|")]
                if len(cols) == 4:
                    rows[(cols[0], cols[1])] = cols

    rows[(model, dataset)] = [
        model,
        dataset,
        f"{ndcg:.6f}",
        str(n_samples),
    ]

    body = [
        f"| {v[0]} | {v[1]} | {v[2]} | {v[3]} |\n"
        for v in sorted(rows.values(), key=lambda x: (x[0], x[1]))
    ]

    path.write_text("".join(header + body))


def main():
    y_true = load_csr("data/test-output.npz")
    print("Ground truth shape:", y_true.shape)

    models = {
        "mllm": "data/test-mllm.npz",
        "bonsai": "data/test-bonsai.npz",
        "fasttext": "data/test-fasttext.npz",
        "nn": "data/test-nn.npz",
    }

    scoreboard_path = Path("benchmarks/SCOREBOARD.md")

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
