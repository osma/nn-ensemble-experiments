import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import torch

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


def f1_at_k(y_true: csr_matrix, y_pred: csr_matrix, k: int):
    """
    Compute example-based mean F1@k using top-k predictions.

    Rows with zero true labels are skipped.
    """
    f1s = []

    for i in range(y_true.shape[0]):
        true_row = y_true.getrow(i)
        if true_row.nnz == 0:
            continue

        pred_row = y_pred.getrow(i)

        if pred_row.nnz > k:
            topk_pos = np.argpartition(pred_row.data, -k)[-k:]
            top_indices = pred_row.indices[topk_pos]
        else:
            top_indices = pred_row.indices

        pred_set = set(top_indices.tolist())
        true_set = set(true_row.indices.tolist())

        tp = len(pred_set & true_set)
        if tp == 0:
            f1s.append(0.0)
            continue

        fp = len(pred_set) - tp
        fn = len(true_set) - tp

        f1 = (2 * tp) / (2 * tp + fp + fn)
        f1s.append(f1)

    return float(np.mean(f1s)), len(f1s)


def _csr_row_indices_list(y_true: csr_matrix) -> list[np.ndarray]:
    """
    Precompute true label indices per row for a CSR matrix.

    Returns a list of length n_rows, where each element is a 1D np.ndarray of
    label indices (possibly empty).
    """
    indptr = y_true.indptr
    indices = y_true.indices
    out: list[np.ndarray] = []
    for i in range(y_true.shape[0]):
        out.append(indices[indptr[i] : indptr[i + 1]])
    return out


def ndcg_at_k_dense(
    y_true: csr_matrix,
    y_score: torch.Tensor,
    k: int = K_DEFAULT,
):
    """
    Compute mean NDCG@k from:
      - y_true: CSR ground truth (binary relevance; values ignored)
      - y_score: dense torch tensor of shape (n_samples, n_labels)

    This avoids converting dense predictions to CSR (a major bottleneck).

    Notes:
    - Rows with zero true labels are skipped.
    - y_score can be logits or probabilities; only ranking matters.
    - Computation uses torch.topk for top-k retrieval, then per-row set
      membership checks against sparse true indices.
    """
    if y_score.ndim != 2:
        raise ValueError(f"Expected y_score to have shape (N, L), got {y_score.shape}")
    if y_score.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Row mismatch: y_true has {y_true.shape[0]} rows, y_score has {y_score.shape[0]}"
        )
    if k <= 0:
        raise ValueError("k must be positive")

    # Work on CPU for membership checks; topk can be done on GPU then moved.
    # Keeping it simple and robust: do topk on the current device, then move indices to CPU.
    k_eff = min(k, y_score.shape[1])
    topk_idx = torch.topk(y_score, k=k_eff, dim=1, largest=True, sorted=True).indices
    topk_idx_np = topk_idx.detach().cpu().numpy()

    true_idx_list = _csr_row_indices_list(y_true)

    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
    ndcgs: list[float] = []

    for i in range(y_true.shape[0]):
        true_idx = true_idx_list[i]
        if true_idx.size == 0:
            continue

        ranked = topk_idx_np[i]
        rel = np.isin(ranked, true_idx).astype(np.float64)

        dcg = float(np.sum(rel * discounts))

        ideal_len = min(true_idx.size, k_eff)
        idcg = float(np.sum(discounts[:ideal_len]))

        ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)), len(ndcgs)


def f1_at_k_dense(
    y_true: csr_matrix,
    y_score: torch.Tensor,
    k: int,
):
    """
    Compute example-based mean F1@k from:
      - y_true: CSR ground truth
      - y_score: dense torch tensor of shape (n_samples, n_labels)

    Uses torch.topk to get predicted indices; avoids CSR conversion.
    Rows with zero true labels are skipped.
    """
    if y_score.ndim != 2:
        raise ValueError(f"Expected y_score to have shape (N, L), got {y_score.shape}")
    if y_score.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Row mismatch: y_true has {y_true.shape[0]} rows, y_score has {y_score.shape[0]}"
        )
    if k <= 0:
        raise ValueError("k must be positive")

    k_eff = min(k, y_score.shape[1])
    topk_idx = torch.topk(y_score, k=k_eff, dim=1, largest=True, sorted=False).indices
    topk_idx_np = topk_idx.detach().cpu().numpy()

    true_idx_list = _csr_row_indices_list(y_true)

    f1s: list[float] = []
    for i in range(y_true.shape[0]):
        true_idx = true_idx_list[i]
        if true_idx.size == 0:
            continue

        pred = topk_idx_np[i]
        # Use numpy set ops (fast enough at k=5)
        tp = int(np.intersect1d(pred, true_idx, assume_unique=False).size)
        if tp == 0:
            f1s.append(0.0)
            continue

        fp = int(k_eff - tp)
        fn = int(true_idx.size - tp)
        f1s.append((2.0 * tp) / (2.0 * tp + fp + fn))

    return float(np.mean(f1s)), len(f1s)


def _parse_float(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("-inf")


def _render_top10_table(
    rows: list[dict[str, str]],
    sort_key: str,
    title: str,
):
    ranked = sorted(
        rows,
        key=lambda r: _parse_float(r[sort_key]),
        reverse=True,
    )[:10]

    if sort_key == "test ndcg@10":
        header = "| Rank | Model | Test NDCG@10 | Test NDCG@1000 | Test F1@5 |\n"
        sep = "|------|-------|--------------|----------------|-----------|\n"
        line = "| {rank} | {model} | {ndcg10:.6f} | {ndcg1000:.6f} | {f1:.6f} |\n"
    elif sort_key == "test ndcg@1000":
        header = "| Rank | Model | Test NDCG@1000 | Test NDCG@10 | Test F1@5 |\n"
        sep = "|------|-------|----------------|--------------|-----------|\n"
        line = "| {rank} | {model} | {ndcg1000:.6f} | {ndcg10:.6f} | {f1:.6f} |\n"
    else:  # test f1@5
        header = "| Rank | Model | Test F1@5 | Test NDCG@10 | Test NDCG@1000 |\n"
        sep = "|------|-------|-----------|--------------|----------------|\n"
        line = "| {rank} | {model} | {f1:.6f} | {ndcg10:.6f} | {ndcg1000:.6f} |\n"

    body = []
    for i, r in enumerate(ranked, start=1):
        body.append(
            line.format(
                rank=i,
                model=r["model"],
                ndcg10=_parse_float(r.get("test ndcg@10", "")),
                ndcg1000=_parse_float(r.get("test ndcg@1000", "")),
                f1=_parse_float(r.get("test f1@5", "")),
            )
        )

    return [
        f"\n## {title}\n\n",
        header,
        sep,
        *body,
    ]


def update_markdown_scoreboard(
    path: Path,
    model: str,
    dataset: str,
    metrics: dict[str, float],
    n_samples: int,
    epoch: int | None = None,
):
    header = [
        "# Benchmark Scoreboard\n\n",
        "| Model | Epoch | Train NDCG@10 | Train NDCG@1000 | Test NDCG@10 | Test NDCG@1000 | Test F1@5 |\n",
        "|-------|-------|---------------|----------------|-------------|----------------|-----------|\n",
    ]

    rows: dict[str, dict[str, str]] = {}

    if path.exists():
        for line in path.read_text().splitlines():
            if not line.startswith("|"):
                continue

            cols = [c.strip() for c in line.strip("|").split("|")]
            if len(cols) != 7:
                continue

            if cols[0] == "Model" or all(set(c) <= {"-"} for c in cols):
                continue

            rows[cols[0]] = {
                "model": cols[0],
                "epoch": cols[1],
                "train ndcg@10": cols[2],
                "train ndcg@1000": cols[3],
                "test ndcg@10": cols[4],
                "test ndcg@1000": cols[5],
                "test f1@5": cols[6],
            }

    row = rows.get(
        model,
        {
            "model": model,
            "epoch": "",
            "train ndcg@10": "",
            "train ndcg@1000": "",
            "test ndcg@10": "",
            "test ndcg@1000": "",
            "test f1@5": "",
        },
    )

    prefix = "train" if dataset == "train" else "test"
    for k, v in metrics.items():
        row[f"{prefix} {k}"] = f"{v:.6f}"

    if epoch is not None:
        row["epoch"] = str(epoch)

    rows[model] = row

    ordered_rows = sorted(rows.values(), key=lambda x: x["model"])

    main_table = [
        f"| {r['model']} | {r.get('epoch','')} | {r['train ndcg@10']} | {r['train ndcg@1000']} | "
        f"{r['test ndcg@10']} | {r['test ndcg@1000']} | {r['test f1@5']} |\n"
        for r in ordered_rows
    ]

    top10_ndcg10 = _render_top10_table(
        ordered_rows,
        sort_key="test ndcg@10",
        title="Top 10 Models by Test NDCG@10",
    )

    top10_ndcg1000 = _render_top10_table(
        ordered_rows,
        sort_key="test ndcg@1000",
        title="Top 10 Models by Test NDCG@1000",
    )

    top10_f1 = _render_top10_table(
        ordered_rows,
        sort_key="test f1@5",
        title="Top 10 Models by Test F1@5",
    )

    path.write_text(
        "".join(header + main_table + top10_ndcg10 + top10_ndcg1000 + top10_f1)
    )
