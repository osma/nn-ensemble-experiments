import numpy as np
from scipy.sparse import csr_matrix

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


def main():
    y_pred = load_csr("data/test-mllm.npz")
    y_true = load_csr("data/test-output.npz")

    print("Predictions shape:", y_pred.shape)
    print("Ground truth shape:", y_true.shape)

    ndcg, n_used = ndcg_at_k(y_true, y_pred, k=K)
    print(f"\nNDCG@{K} = {ndcg:.6f} (computed over {n_used} samples)")


if __name__ == "__main__":
    main()
