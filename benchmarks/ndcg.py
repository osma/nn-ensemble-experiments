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
        "| Model | Dataset | NDCG@10 | NDCG@1000 | Samples |\n",
        "|-------|---------|---------|-----------|---------|\n",
    ]

    rows: dict[tuple[str, str], dict[str, str]] = {}

    if path.exists():
        for line in path.read_text().splitlines():
            if not line.startswith("|"):
                continue

            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) != 5:
                continue

            if cols[0] == "Model" or all(set(c) <= {"-"} for c in cols):
                continue

            rows[(cols[0], cols[1])] = {
                "model": cols[0],
                "dataset": cols[1],
                "ndcg@10": cols[2],
                "ndcg@1000": cols[3],
                "samples": cols[4],
            }

    key = (model, dataset)
    row = rows.get(
        key,
        {
            "model": model,
            "dataset": dataset,
            "ndcg@10": "",
            "ndcg@1000": "",
            "samples": str(n_samples),
        },
    )

    for k, v in metrics.items():
        row[k] = f"{v:.6f}"

    row["samples"] = str(n_samples)
    rows[key] = row

    body = [
        f"| {r['model']} | {r['dataset']} | {r['ndcg@10']} | {r['ndcg@1000']} | {r['samples']} |\n"
        for r in sorted(rows.values(), key=lambda x: (x["model"], x["dataset"]))
    ]

    path.write_text("".join(header + body))
