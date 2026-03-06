# STATUS: EXPERIMENTAL
# Purpose: Candidate-set-only training/evaluation for a mean-like ensemble.
#
# Motivation:
# - Base predictions are extremely sparse (koko: <=100 nonzeros/model/doc).
# - Dense BCE over all labels encourages learning per-label bias/frequency rather
#   than improving ranking among plausible candidates.
# - This model trains (and evaluates) on per-document candidate sets:
#     candidates = union of nonzero predicted labels across the 3 base models
#   plus optional sampled negatives from outside candidates.
#
# Scoring:
#   logit[d, l] = mean_m x[d, m, l] + bias[l]      (logits; no clamp)
#
# Training loss:
#   BCEWithLogits over (candidates ∪ sampled negatives) for each doc
#
# Evaluation (ranking metric):
#   For NDCG/F1, we build a dense (N, L) score matrix where non-candidates get a
#   very negative logit, so they never enter top-k.
from __future__ import annotations

from pathlib import Path
import sys

# Allow running as a script: `uv run benchmarks/torch_mean_candidate.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

from benchmarks.datasets import ensemble3_keys, pred_path, truth_path
from benchmarks.device import get_device
from benchmarks.metrics import load_csr, ndcg_at_k_dense, f1_at_k_dense, update_markdown_scoreboard
from benchmarks.preprocessing import csr_to_log1p_tensor


DEVICE = get_device()

# Training defaults (kept similar to other torch benchmarks)
EPOCHS = 20
LR = 0.003
WEIGHT_DECAY = 0.0
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512

K_VALUES = (10, 1000)

PATIENCE = 2
MIN_EPOCHS = 1  # candidate training tends to peak early; allow selecting epoch 1

EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337
TRAIN_SEED = 0

# Candidate/negative sampling
NEG_PER_DOC = 256  # sampled negatives outside candidate set (0 disables)
NEG_SEED = 4242

# For evaluation: logits for non-candidates
NON_CANDIDATE_LOGIT = -1e9


class TorchMeanCandidate(nn.Module):
    """
    Mean ensemble in logit space with learnable per-label bias.

    Input:
        x: (batch, M=3, L) log1p-preprocessed non-negative scores
    Output:
        logits: (batch, L)
    """

    def __init__(self, n_models: int, n_labels: int):
        super().__init__()
        if n_models != 3:
            raise ValueError("This experimental model is intended for 3-way ensembles only")
        self.n_models = int(n_models)
        self.n_labels = int(n_labels)
        self.bias = nn.Parameter(torch.zeros((n_labels,), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, M, L), got {tuple(x.shape)}")
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected x with (M={self.n_models}, L={self.n_labels}), got {tuple(x.shape)}"
            )
        mean = x.mean(dim=1)  # (B, L)
        return mean + self.bias


def _sync_if_cuda() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class _Timer:
    def __init__(self):
        self.t0: float | None = None
        self.dt: float | None = None

    def __enter__(self):
        _sync_if_cuda()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _sync_if_cuda()
        assert self.t0 is not None
        self.dt = time.perf_counter() - self.t0


def _row_union_indices(a: csr_matrix, b: csr_matrix, c: csr_matrix) -> list[np.ndarray]:
    """
    Return per-row union of nonzero column indices across three CSR matrices.
    """
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("Prediction matrices must have identical shape")

    out: list[np.ndarray] = []
    for i in range(a.shape[0]):
        ia = a.indices[a.indptr[i] : a.indptr[i + 1]]
        ib = b.indices[b.indptr[i] : b.indptr[i + 1]]
        ic = c.indices[c.indptr[i] : c.indptr[i + 1]]
        if ia.size == 0 and ib.size == 0 and ic.size == 0:
            out.append(np.empty((0,), dtype=np.int64))
        else:
            out.append(np.unique(np.concatenate([ia, ib, ic]).astype(np.int64, copy=False)))
    return out


def _sample_negatives(
    *,
    n_labels: int,
    candidate_idx: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample k unique labels from outside candidate_idx.
    If k=0, returns empty.
    """
    if k <= 0:
        return np.empty((0,), dtype=np.int64)

    cand = candidate_idx
    cand_set = set(cand.tolist()) if cand.size else set()
    # Fast path when candidate set is tiny vs n_labels (typical)
    negs: list[int] = []
    while len(negs) < k:
        # Oversample a bit to reduce loops
        draw = rng.integers(0, n_labels, size=(k * 2,), endpoint=False, dtype=np.int64)
        for x in draw.tolist():
            if x in cand_set:
                continue
            negs.append(x)
            if len(negs) >= k:
                break
    # unique while preserving order
    negs_arr = np.fromiter(dict.fromkeys(negs), dtype=np.int64)
    if negs_arr.size >= k:
        return negs_arr[:k]
    # In extremely pathological cases, retry
    return _sample_negatives(n_labels=n_labels, candidate_idx=candidate_idx, k=k, rng=rng)


def _gather_logits_and_targets(
    *,
    logits_full: torch.Tensor,   # (B, L) on DEVICE
    y_true_full: torch.Tensor,   # (B, L) on DEVICE
    cand_list: list[np.ndarray], # length B, CPU arrays of indices
    neg_per_doc: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a packed (B, Kmax) tensor for logits and targets by selecting per-row
    candidate columns + sampled negatives, padded with a sentinel.

    Returns:
        logits_packed: (B, Kmax) float
        targets_packed: (B, Kmax) float
    """
    device = logits_full.device
    bsz, n_labels = logits_full.shape
    assert bsz == len(cand_list)

    selected_per_row: list[np.ndarray] = []
    max_k = 0
    for i in range(bsz):
        cand = cand_list[i]
        neg = _sample_negatives(n_labels=n_labels, candidate_idx=cand, k=neg_per_doc, rng=rng)
        if cand.size == 0 and neg.size == 0:
            idx = np.empty((0,), dtype=np.int64)
        elif cand.size == 0:
            idx = neg
        elif neg.size == 0:
            idx = cand
        else:
            idx = np.unique(np.concatenate([cand, neg]).astype(np.int64, copy=False))
        selected_per_row.append(idx)
        max_k = max(max_k, int(idx.size))

    if max_k == 0:
        # No candidates anywhere in this batch -> return empty; caller should skip
        empty = torch.empty((bsz, 0), device=device, dtype=logits_full.dtype)
        return empty, empty

    # Pad with -1 indices (masked later)
    idx_padded = torch.full((bsz, max_k), -1, device=device, dtype=torch.int64)
    for i, idx in enumerate(selected_per_row):
        if idx.size == 0:
            continue
        idx_t = torch.from_numpy(idx).to(device=device, dtype=torch.int64)
        idx_padded[i, : idx_t.numel()] = idx_t

    # Gather with masking
    valid = idx_padded >= 0
    safe_idx = torch.clamp(idx_padded, min=0)

    logits = torch.gather(logits_full, dim=1, index=safe_idx)
    targets = torch.gather(y_true_full, dim=1, index=safe_idx)

    # Mask padded positions: logits arbitrary; targets 0 and will be masked out of loss
    targets = torch.where(valid, targets, torch.zeros_like(targets))
    logits = torch.where(valid, logits, torch.zeros_like(logits))

    return logits, targets


def _masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute mean BCEWithLogits over masked positions.
    """
    # elementwise bce
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    loss = loss * mask.to(dtype=loss.dtype)
    denom = torch.clamp(mask.sum(), min=1.0)
    return loss.sum() / denom


def _predict_candidates_only(
    *,
    model: nn.Module,
    x_cpu: torch.Tensor,         # (N, M, L) on CPU
    cand_rows: list[np.ndarray], # length N, CPU arrays
    n_labels: int,
) -> torch.Tensor:
    """
    Predict logits for candidate labels only; non-candidates get very negative logits.
    Returns dense (N, L) CPU tensor for use with ndcg_at_k_dense / f1_at_k_dense.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_cpu),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
    )

    outs: list[torch.Tensor] = []
    offset = 0
    with torch.no_grad():
        for (xb,) in loader:
            bsz = xb.shape[0]
            xb = xb.to(DEVICE, non_blocking=True)
            logits_full = model(xb)  # (B, L)

            # Start with all very negative
            out = torch.full((bsz, n_labels), NON_CANDIDATE_LOGIT, device=logits_full.device)
            for i in range(bsz):
                idx = cand_rows[offset + i]
                if idx.size == 0:
                    continue
                idx_t = torch.from_numpy(idx).to(device=logits_full.device, dtype=torch.int64)
                out[i].index_copy_(0, idx_t, logits_full[i].index_select(0, idx_t))
            outs.append(out.detach().cpu())
            offset += bsz

    return torch.cat(outs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="yso-fi",
        choices=["yso-fi", "yso-en", "koko"],
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--neg-per-doc",
        type=int,
        default=NEG_PER_DOC,
        help="Number of negatives sampled outside candidate set per document (0 disables).",
    )
    args = parser.parse_args()
    dataset = str(args.dataset)
    neg_per_doc = int(args.neg_per_doc)
    if neg_per_doc < 0:
        raise ValueError("--neg-per-doc must be >= 0")

    torch.manual_seed(TRAIN_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)

    rng_neg = np.random.default_rng(NEG_SEED)

    e3 = ensemble3_keys(dataset)
    model_name = f"torch_mean_candidate({','.join(e3)})"
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading data...")

    y_train_true_csr = load_csr(str(truth_path(dataset, "train")))
    y_test_true_csr = load_csr(str(truth_path(dataset, "test")))

    train_preds_csr = [load_csr(str(pred_path(dataset, "train", k))) for k in e3]
    test_preds_csr = [load_csr(str(pred_path(dataset, "test", k))) for k in e3]

    # Candidate sets derived from sparse structures (fast; no densification needed)
    train_cand_rows = _row_union_indices(train_preds_csr[0], train_preds_csr[1], train_preds_csr[2])
    test_cand_rows = _row_union_indices(test_preds_csr[0], test_preds_csr[1], test_preds_csr[2])

    # Dense input features for the model (still OK; model is simple)
    X_train = torch.stack([csr_to_log1p_tensor(p) for p in train_preds_csr], dim=1)
    X_test = torch.stack([csr_to_log1p_tensor(p) for p in test_preds_csr], dim=1)

    # Dense targets for loss (we will gather only candidate/negative columns)
    Y_train = torch.from_numpy(y_train_true_csr.toarray()).float()
    Y_test = torch.from_numpy(y_test_true_csr.toarray()).float()

    n_models = int(X_train.shape[1])
    n_labels = int(X_train.shape[2])

    # Early stop subset
    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_train = int(X_train.shape[0])
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_train)
    train_eval_idx = rng.choice(n_train, size=n_eval, replace=False)

    X_train_eval = X_train[train_eval_idx]
    y_train_true_eval_csr = y_train_true_csr[train_eval_idx]
    train_eval_cand_rows = [train_cand_rows[i] for i in train_eval_idx.tolist()]

    model = TorchMeanCandidate(n_models=n_models, n_labels=n_labels).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )

    print("Starting training...")

    # We'll train with indices so we can map batch rows -> candidate lists
    train_idx = np.arange(n_train, dtype=np.int64)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(train_idx), X_train, Y_train
        ),
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    best_metric = float("-inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_train_metrics: dict[str, float] | None = None
    best_test_metrics: dict[str, float] | None = None
    best_n_used_train: int | None = None
    best_n_used_test: int | None = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        with _Timer() as t_train:
            for idx_b, xb, yb in train_loader:
                # idx_b is on CPU by default; use it to fetch candidate rows
                idx_np = idx_b.numpy().astype(np.int64, copy=False)
                cand_list = [train_cand_rows[i] for i in idx_np.tolist()]

                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits_full = model(xb)  # (B, L)
                logits_sel, targets_sel = _gather_logits_and_targets(
                    logits_full=logits_full,
                    y_true_full=yb,
                    cand_list=cand_list,
                    neg_per_doc=neg_per_doc,
                    rng=rng_neg,
                )
                if logits_sel.numel() == 0:
                    continue

                mask = (logits_sel != 0.0) | (targets_sel != 0.0)
                loss = _masked_bce_with_logits(logits_sel, targets_sel, mask=mask)
                loss.backward()
                optimizer.step()

        # --- Early stop metric: train subset NDCG@1000 (candidates only) ---
        with _Timer() as t_pred_train:
            train_eval_scores = _predict_candidates_only(
                model=model,
                x_cpu=X_train_eval,
                cand_rows=train_eval_cand_rows,
                n_labels=n_labels,
            )
        train_ndcg1000, n_used_train_eval = ndcg_at_k_dense(
            y_train_true_eval_csr, train_eval_scores, k=1000
        )

        # --- Test metrics (candidates only) ---
        with _Timer() as t_pred_test:
            test_scores = _predict_candidates_only(
                model=model,
                x_cpu=X_test,
                cand_rows=test_cand_rows,
                n_labels=n_labels,
            )

        test_metrics: dict[str, float] = {}
        n_used_test: int | None = None
        for k in K_VALUES:
            ndcg, n_used_test = ndcg_at_k_dense(y_test_true_csr, test_scores, k=k)
            test_metrics[f"ndcg@{k}"] = ndcg
        f1, _ = f1_at_k_dense(y_test_true_csr, test_scores, k=5)
        test_metrics["f1@5"] = f1

        print(
            f"[neg_per_doc={neg_per_doc}] "
            f"Epoch {epoch:02d} | "
            f"train_ndcg@1000(subset)={train_ndcg1000:.6f} | "
            f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
            f"test_f1@5={test_metrics['f1@5']:.6f} | "
            f"timing train={float(t_train.dt or 0.0):.3f}s "
            f"pred_train={float(t_pred_train.dt or 0.0):.3f}s "
            f"pred_test={float(t_pred_test.dt or 0.0):.3f}s"
        )

        current = float(train_ndcg1000)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Full train/test metrics at best epoch
            full_train_scores = _predict_candidates_only(
                model=model,
                x_cpu=X_train,
                cand_rows=train_cand_rows,
                n_labels=n_labels,
            )
            best_train_metrics = {}
            n_used_train_full: int | None = None
            for k in K_VALUES:
                ndcg, n_used_train_full = ndcg_at_k_dense(y_train_true_csr, full_train_scores, k=k)
                best_train_metrics[f"ndcg@{k}"] = ndcg
            best_n_used_train = int(n_used_train_full or 0)

            best_test_metrics = test_metrics.copy()
            best_n_used_test = int(n_used_test or 0)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch >= MIN_EPOCHS and epochs_no_improve >= PATIENCE:
            break

    assert best_state is not None
    assert best_epoch is not None
    assert best_train_metrics is not None
    assert best_test_metrics is not None
    assert best_n_used_train is not None
    assert best_n_used_test is not None

    model.load_state_dict(best_state)

    update_markdown_scoreboard(
        path=scoreboard_path,
        model=model_name,
        dataset=dataset,
        split="train",
        metrics=best_train_metrics,
        n_samples=best_n_used_train,
        epoch=best_epoch,
    )
    update_markdown_scoreboard(
        path=scoreboard_path,
        model=model_name,
        dataset=dataset,
        split="test",
        metrics=best_test_metrics,
        n_samples=best_n_used_test,
        epoch=best_epoch,
    )

    print(
        "\nFinal test metrics | "
        f"ndcg@10={best_test_metrics['ndcg@10']:.6f} | "
        f"ndcg@1000={best_test_metrics['ndcg@1000']:.6f} | "
        f"f1@5={best_test_metrics['f1@5']:.6f} | "
        f"epoch={best_epoch}"
    )
    print("\nSaved result to SCOREBOARD.md")


if __name__ == "__main__":
    main()
