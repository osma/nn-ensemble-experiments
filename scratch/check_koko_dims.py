import numpy as np
from scipy.sparse import load_npz
from benchmarks.datasets import truth_path, pred_path, ensemble3_keys

dataset = "koko"
y_train_true = load_npz(truth_path(dataset, "train"))
print(f"Dataset: {dataset}")
print(f"y_train_true shape: {y_train_true.shape}")
print(f"y_train_true nnz: {y_train_true.nnz}")

ensemble_keys = ensemble3_keys(dataset)
for k in ensemble_keys:
    p = load_npz(pred_path(dataset, "train", k))
    print(f"Pred {k} shape: {p.shape}, nnz: {p.nnz}")
