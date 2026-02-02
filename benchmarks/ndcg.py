import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path

K_DEFAULT = 1000


def load_csr(path: str) -> csr_matrix:
    """Load a CSR matrix stored in NPZ format."""
    data = np.load(path)
    return csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"]),
    )


def ndcg_at_k(y_true: csr_matrix, y_pred: csr_matrix, k: int = K_DEFAULT):
    """
    Compute mean NDCG@k for multilabel classification using sparse matrices.

    Rows with zero true labels are skipped.
    """
    ndcgs = []

    for i in range(y_true.shape[0]):
        true_row = y_true.getrow(i)
        if true_row.nnz == 0:
            continue

        pred_row = y_pred.getrow(i)

        if pred_row.nnz > k:
            topk_pos = np.argpartition(pred_row.data, -k)[-k:]
            top_indices = pred_row.indices[topk_pos]
            top_scores = pred_row.data[topk_pos]
            order = np.argsort(top_scores)[::-1]
            ranked = top_indices[order]
        else:
            order = np.argsort(pred_row.data)[::-1]
            ranked = pred_row.indices[order]

        rel = np.isin(ranked, true_row.indices).astype(float)
        discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
        dcg = float(np.sum(rel * discounts))

        ideal_len = min(true_row.nnz, k)
        ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2))
        idcg = float(np.sum(ideal_discounts))

        ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)), len(ndcgs)


def update_markdown_scoreboard(
    path: Path,
    model: str,
    dataset: str,
    metrics: dict[str, float],
    n_samples: int,
):
    header = [
        "# Benchmark Scoreboard\n\n",
        "| Model | Train NDCG@10 | Train NDCG@1000 | Test NDCG@10 | Test NDCG@1000 |\n",
        "|-------|---------------|----------------|-------------|----------------|\n",
    ]

    rows: dict[str, dict[str, str]] = {}

    if path.exists():
        for line in path.read_text().splitlines():
            if not line.startswith("|"):
                continue

            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) != 5:
                continue

            if cols[0] == "Model" or all(set(c) <= {"-"} for c in cols):
                continue

            rows[cols[0]] = {
                "model": cols[0],
                "train ndcg@10": cols[1],
                "train ndcg@1000": cols[2],
                "test ndcg@10": cols[3],
                "test ndcg@1000": cols[4],
            }

    row = rows.get(
        model,
        {
            "model": model,
            "train ndcg@10": "",
            "train ndcg@1000": "",
            "test ndcg@10": "",
            "test ndcg@1000": "",
        },
    )

    prefix = "train" if dataset == "train" else "test"
    for k, v in metrics.items():
        row[f"{prefix} {k}"] = f"{v:.6f}"

    rows[model] = row

    body = [
        f"| {r['model']} | {r['train ndcg@10']} | {r['train ndcg@1000']} | "
        f"{r['test ndcg@10']} | {r['test ndcg@1000']} |\n"
        for r in sorted(rows.values(), key=lambda x: x["model"])
    ]

    path.write_text("".join(header + body))
