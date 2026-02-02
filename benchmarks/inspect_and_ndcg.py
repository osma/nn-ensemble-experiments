import numpy as np

K = 1000


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = K):
    """
    Compute mean NDCG@k for multilabel classification.

    Parameters
    ----------
    y_true : np.ndarray
        Binary relevance matrix of shape (n_samples, n_labels).
    y_score : np.ndarray
        Prediction score matrix of shape (n_samples, n_labels).
    k : int
        Cutoff for NDCG.

    Returns
    -------
    mean_ndcg : float
        Mean NDCG@k over samples with at least one positive label.
    n_used : int
        Number of samples used in the computation.
    """
    ndcgs = []

    for true, score in zip(y_true, y_score):
        if true.sum() == 0:
            continue

        order = np.argsort(score)[::-1][:k]
        rel = true[order]

        discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
        dcg = np.sum(rel * discounts)

        ideal_rel = np.sort(true)[::-1][:k]
        ideal_dcg = np.sum(ideal_rel * discounts[: ideal_rel.size])

        ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    return float(np.mean(ndcgs)), len(ndcgs)


def main():
    data = np.load("data/test-mllm.npz")

    print("Keys in data/test-mllm.npz:")
    for key in data.files:
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    # Adjust these names if necessary after inspecting the keys above
    y_true = data["y_true"]
    y_score = data["y_pred"]

    ndcg, n_used = ndcg_at_k(y_true, y_score, k=K)
    print(f"\nNDCG@{K} = {ndcg:.6f} (computed over {n_used} samples)")


if __name__ == "__main__":
    main()
