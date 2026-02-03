"""
This module has been removed.

Use `benchmarks.metrics` directly instead:
    from benchmarks.metrics import ndcg_at_k, load_csr, f1_at_k, update_markdown_scoreboard
"""
raise ImportError(
    "benchmarks.ndcg has been removed; import from benchmarks.metrics instead"
)


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
        other = "test ndcg@1000"
        header = "| Rank | Model | Test NDCG@10 | Test NDCG@1000 |\n"
        sep = "|------|-------|--------------|----------------|\n"
        line = (
            "| {rank} | {model} | {v1:.6f} | {v2:.6f} |\n"
        )
    else:
        other = "test ndcg@10"
        header = "| Rank | Model | Test NDCG@1000 | Test NDCG@10 |\n"
        sep = "|------|-------|----------------|--------------|\n"
        line = (
            "| {rank} | {model} | {v1:.6f} | {v2:.6f} |\n"
        )

    body = []
    for i, r in enumerate(ranked, start=1):
        body.append(
            line.format(
                rank=i,
                model=r["model"],
                v1=_parse_float(r[sort_key]),
                v2=_parse_float(r[other]),
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

    ordered_rows = sorted(rows.values(), key=lambda x: x["model"])

    main_table = [
        f"| {r['model']} | {r['train ndcg@10']} | {r['train ndcg@1000']} | "
        f"{r['test ndcg@10']} | {r['test ndcg@1000']} |\n"
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

    path.write_text(
        "".join(header + main_table + top10_ndcg10 + top10_ndcg1000)
    )
