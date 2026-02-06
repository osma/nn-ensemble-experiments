# STATUS: EXPERIMENTAL
# Purpose: Per-label ensemble where per-model trust is adjusted using label frequency features.
from pathlib import Path
import sys
import time

# Allow running as a script: `uv run benchmarks/torch_per_label_freq_gate.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.device import get_device
from benchmarks.metrics import (
    load_csr,
    ndcg_at_k_dense,
    f1_at_k_dense,
    update_markdown_scoreboard,
)

DEVICE = get_device()
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 32
K_VALUES = (10, 1000)

PATIENCE = 2
MIN_EPOCHS = 2

EVAL_BATCH_SIZE = 512
EARLY_STOP_EVAL_ROWS = 512
EARLY_STOP_SEED = 1337

# Early stopping selection metric for this model.
# - "ndcg10": select by train NDCG@10 on the fixed train subset
# - "ndcg1000": select by train NDCG@1000 on the fixed train subset
# - "mix": select by a weighted mix of both (recommended default)
EARLY_STOP_METRIC = "mix"
EARLY_STOP_MIX_W10 = 0.7
EARLY_STOP_MIX_W1000 = 0.3

# Default regularization strength for the frequency-based residual weight correction.
# Higher -> pushes the model closer to plain torch_per_label behavior.
LAMBDA_DELTA_DEFAULT = 1e-3

# Default alpha_max for bounded alpha.
ALPHA_MAX_DEFAULT = 0.5


def csr_to_dense_tensor(csr):
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


def _predict_in_batches(model: torch.nn.Module, x_cpu: torch.Tensor) -> torch.Tensor:
    model.eval()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_cpu),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
    )

    outs: list[torch.Tensor] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            outs.append(logits.detach().cpu())
    return torch.cat(outs, dim=0)


def _label_freq_features(y_train_true_csr) -> torch.Tensor:
    """
    Build per-label frequency features from y_train_true (CSR).

    Returns:
        Tensor of shape (L, F) on CPU.
    """
    # counts per label (column sums)
    counts = np.asarray(y_train_true_csr.sum(axis=0)).reshape(-1).astype(np.float32)  # (L,)

    logf = np.log1p(counts)
    inv_logf = 1.0 / (1.0 + logf)
    is_zero = (counts == 0.0).astype(np.float32)

    # Rare-label indicator (helps the model learn a distinct regime vs just "small logf")
    is_rare_1 = (counts <= 1.0).astype(np.float32)
    is_rare_5 = (counts <= 5.0).astype(np.float32)

    feats = np.stack([logf, inv_logf, is_zero, is_rare_1, is_rare_5], axis=1)  # (L, 5)

    # Normalize logf for stability; keep indicator features as-is.
    if feats.shape[0] > 0:
        mu = feats[:, 0].mean()
        sd = feats[:, 0].std() + 1e-6
        feats[:, 0] = (feats[:, 0] - mu) / sd

    return torch.from_numpy(feats).float()  # CPU


def _label_counts(y_train_true_csr) -> np.ndarray:
    """
    Returns:
        counts: np.ndarray of shape (L,) float32
    """
    return np.asarray(y_train_true_csr.sum(axis=0)).reshape(-1).astype(np.float32)


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


class PerLabelFreqGatedEnsemble(nn.Module):
    """
    Per-label weighted ensemble with bias, where per-model trust is adjusted by
    label-frequency features.

    This version uses a *residual* correction to the per-label weights:

        w_eff[m,l] = base_w[m,l] + alpha * delta_w[m,l](freq[l])

    Changes vs earlier versions:
    - delta_w is constrained to be zero-mean across models per label, so it can
      only redistribute trust between models (not scale all models together).
    - alpha is constrained to be nonnegative and bounded: alpha in (0, alpha_max).

    Input:
        x: (batch, M, L) log1p-scaled base predictions
    Output:
        (batch, L) raw logits
    """

    def __init__(
        self,
        n_models: int,
        n_labels: int,
        label_feats: torch.Tensor,
        hidden: int = 16,
        alpha_init: float = 0.1,
        alpha_max: float = 0.5,
    ):
        super().__init__()
        if label_feats.ndim != 2:
            raise ValueError(f"label_feats must be 2D (L,F), got {label_feats.shape}")
        if label_feats.shape[0] != n_labels:
            raise ValueError(
                f"label_feats has {label_feats.shape[0]} labels, expected {n_labels}"
            )
        if alpha_max <= 0:
            raise ValueError("alpha_max must be positive")

        self.n_models = n_models
        self.n_labels = n_labels
        self.alpha_max = float(alpha_max)

        # Base per-model, per-label weights (like torch_per_label)
        self.base_w = nn.Parameter(torch.full((n_models, n_labels), 1.0 / n_models))

        # Per-label bias
        self.bias = nn.Parameter(torch.zeros(n_labels))

        # Non-trainable label features (moves with .to(device))
        self.register_buffer("label_feats", label_feats)  # (L, F)
        fdim = int(label_feats.shape[1])

        # Per-label residual generator: (F) -> (M)
        self.delta = nn.Sequential(
            nn.Linear(fdim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_models),
        )

        # Bounded alpha: alpha = alpha_max * sigmoid(alpha_raw)
        # Initialize alpha_raw so that alpha ~= alpha_init.
        alpha_init = float(alpha_init)
        alpha_init = max(1e-6, min(alpha_init, self.alpha_max - 1e-6))
        p = alpha_init / self.alpha_max  # in (0,1)
        alpha_raw_init = float(np.log(p / (1.0 - p)))
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_raw_init, dtype=torch.float32))

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw) * self.alpha_max

    def delta_w(self) -> torch.Tensor:
        """
        Returns:
            delta_w: (M, L) residual weight correction derived from label_feats.

        Constraint:
            For each label l, mean_m delta_w[m,l] == 0.
            This forces the frequency module to redistribute trust between models
            rather than scaling all models together.
        """
        # (L, M)
        dw_lm = self.delta(self.label_feats)

        # Zero-mean across models per label: (L, M)
        dw_lm = dw_lm - dw_lm.mean(dim=1, keepdim=True)

        # (L, M) -> (M, L)
        return dw_lm.transpose(0, 1).contiguous()

    def effective_w(self) -> torch.Tensor:
        """
        Returns:
            w_eff: (M, L)
        """
        dw = self.delta_w()
        return self.base_w + self.alpha() * dw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, n_models, n_labels), got {x.shape}")
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected n_models={self.n_models}, n_labels={self.n_labels}, got {x.shape}"
            )

        w_eff = self.effective_w()  # (M, L)
        out = (x * w_eff.unsqueeze(0)).sum(dim=1) + self.bias
        return out  # raw logits


def _print_diagnostics(
    model: PerLabelFreqGatedEnsemble,
    label_counts: np.ndarray,
    model_names: list[str] | None = None,
) -> None:
    """
    Print diagnostics to understand what the frequency module is doing.

    Args:
        model: trained model (already on DEVICE)
        label_counts: (L,) counts from training ground truth
        model_names: optional list of base model names in order
    """
    if model_names is None:
        model_names = [f"m{i}" for i in range(model.n_models)]
    if len(model_names) != model.n_models:
        model_names = [f"m{i}" for i in range(model.n_models)]

    with torch.no_grad():
        base_w = model.base_w.detach().cpu()  # (M, L)
        dw = model.delta_w().detach().cpu()  # (M, L)
        alpha = float(model.alpha().detach().cpu())
        delta_scaled = alpha * dw
        w_eff = base_w + delta_scaled

    def _rms(t: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(t * t)).item())

    def _mean_per_model(t: torch.Tensor) -> np.ndarray:
        # t: (M, L) -> (M,)
        return t.mean(dim=1).numpy()

    def _mean_abs_per_model(t: torch.Tensor) -> np.ndarray:
        return t.abs().mean(dim=1).numpy()

    print("\n=== Diagnostics: frequency-gated per-label weights ===")
    print(f"alpha = {alpha:.6f} (alpha_max={model.alpha_max:.6f})")
    print(
        "RMS | "
        f"base_w={_rms(base_w):.6f} | "
        f"delta_w={_rms(dw):.6f} | "
        f"alpha*delta_w={_rms(delta_scaled):.6f} | "
        f"w_eff={_rms(w_eff):.6f}"
    )

    base_mean = _mean_per_model(base_w)
    eff_mean = _mean_per_model(w_eff)
    delta_abs_mean = _mean_abs_per_model(delta_scaled)

    print("\nPer-model mean weights (averaged over labels):")
    for i, name in enumerate(model_names):
        print(
            f"  {name:8s} | mean(base_w)={base_mean[i]: .6f} | "
            f"mean(w_eff)={eff_mean[i]: .6f} | "
            f"mean(|alpha*delta_w|)={delta_abs_mean[i]: .6f}"
        )

    # Frequency bins
    counts = label_counts.astype(np.float32)
    bins: list[tuple[str, np.ndarray]] = [
        ("count==0", np.where(counts == 0)[0]),
        ("count==1", np.where(counts == 1)[0]),
        ("count 2-5", np.where((counts >= 2) & (counts <= 5))[0]),
        ("count 6-20", np.where((counts >= 6) & (counts <= 20))[0]),
        ("count 21-100", np.where((counts >= 21) & (counts <= 100))[0]),
        ("count>=101", np.where(counts >= 101)[0]),
    ]

    print("\nPer-bin mean weights by label frequency:")
    print("  (values are means over labels in the bin; weights are per-model)")
    for label, idx in bins:
        if idx.size == 0:
            print(f"  {label:11s} | n_labels=0")
            continue

        idx_t = torch.from_numpy(idx.astype(np.int64))
        base_bin = base_w.index_select(dim=1, index=idx_t).mean(dim=1).numpy()
        eff_bin = w_eff.index_select(dim=1, index=idx_t).mean(dim=1).numpy()
        delta_bin = delta_scaled.index_select(dim=1, index=idx_t).mean(dim=1).numpy()

        parts = []
        for i, name in enumerate(model_names):
            parts.append(
                f"{name}: eff={eff_bin[i]: .4f} (base={base_bin[i]: .4f}, d={delta_bin[i]: .4f})"
            )
        print(f"  {label:11s} | n_labels={idx.size:5d} | " + " | ".join(parts))

    print("=== End diagnostics ===\n")


def _early_stop_score(train_ndcg10: float, train_ndcg1000: float) -> float:
    if EARLY_STOP_METRIC == "ndcg10":
        return float(train_ndcg10)
    if EARLY_STOP_METRIC == "ndcg1000":
        return float(train_ndcg1000)
    if EARLY_STOP_METRIC == "mix":
        return float(
            EARLY_STOP_MIX_W10 * train_ndcg10 + EARLY_STOP_MIX_W1000 * train_ndcg1000
        )
    raise ValueError(f"Unknown EARLY_STOP_METRIC={EARLY_STOP_METRIC!r}")


def _rms(t: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(t * t)).item())


def _train_one(
    *,
    alpha_max: float,
    lambda_delta: float,
    update_scoreboard: bool,
) -> dict[str, object]:
    scoreboard_path = Path("SCOREBOARD.md")

    print("Using device:", DEVICE)
    print("Loading training data...")

    y_train_true = load_csr("data/train-output.npz")
    train_preds = [
        load_csr("data/train-bonsai.npz"),
        load_csr("data/train-fasttext.npz"),
        load_csr("data/train-mllm.npz"),
    ]

    # Keep X_train on CPU; move only minibatches to GPU.
    X_train = torch.stack([csr_to_dense_tensor(p) for p in train_preds], dim=1)

    # Keep Y_train on CPU (requested).
    Y_train = csr_to_dense_tensor(y_train_true)

    # Label frequency features (CPU)
    label_feats = _label_freq_features(y_train_true)
    label_counts = _label_counts(y_train_true)

    # Fixed random subset of train rows for per-epoch early stopping metric
    rng = np.random.default_rng(EARLY_STOP_SEED)
    n_train = X_train.shape[0]
    n_eval = min(EARLY_STOP_EVAL_ROWS, n_train)
    train_eval_idx = rng.choice(n_train, size=n_eval, replace=False)
    X_train_eval = X_train[train_eval_idx]
    y_train_true_eval = y_train_true[train_eval_idx]

    print("Loading test data...")

    y_test_true = load_csr("data/test-output.npz")
    test_preds = [
        load_csr("data/test-bonsai.npz"),
        load_csr("data/test-fasttext.npz"),
        load_csr("data/test-mllm.npz"),
    ]

    # Keep X_test on CPU; move to GPU only for evaluation forward pass.
    X_test = torch.stack([csr_to_dense_tensor(p) for p in test_preds], dim=1)

    n_models = X_train.shape[1]
    n_labels = X_train.shape[2]

    model = PerLabelFreqGatedEnsemble(
        n_models=n_models,
        n_labels=n_labels,
        label_feats=label_feats,
        hidden=16,
        alpha_init=0.1,
        alpha_max=alpha_max,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.001,
        eps=1e-8,
    )

    criterion = nn.BCEWithLogitsLoss()

    print("Starting training...")
    print(
        f"Config | alpha_max={alpha_max:.6f} | lambda_delta={lambda_delta:.6g} | early_stop={EARLY_STOP_METRIC}"
    )

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # Early stopping: select best epoch by train subset metric (no test leakage)
    best_metric = float("-inf")
    best_epoch = None
    best_state = None
    best_train_metrics = None
    best_test_metrics = None
    best_n_used_train = None
    best_n_used_test = None
    best_alpha = None
    best_rms_alpha_delta = None
    best_loss = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.perf_counter()

        model.train()
        with _Timer() as t_train_step:
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                logits = model(xb)

                # Regularize the *applied* frequency-based residual so it stays small unless useful.
                dw = model.delta_w()
                delta_scaled = model.alpha() * dw
                reg = (delta_scaled * delta_scaled).mean()

                loss = criterion(logits, yb) + (lambda_delta * reg)
                loss.backward()
                optimizer.step()

        # --- Train evaluation for early stopping (subset only) ---
        with _Timer() as t_pred_train:
            train_scores_eval = _predict_in_batches(model, X_train_eval)

        with _Timer() as t_ndcg_train:
            train_ndcg10, _ = ndcg_at_k_dense(y_train_true_eval, train_scores_eval, k=10)
            train_ndcg1000, _ = ndcg_at_k_dense(
                y_train_true_eval, train_scores_eval, k=1000
            )

        # --- Test evaluation (batched; no CSR conversion) ---
        with _Timer() as t_pred_test:
            test_scores = _predict_in_batches(model, X_test)

        test_metrics = {}
        t_ndcg_test: dict[int, float] = {}
        for k in K_VALUES:
            with _Timer() as t:
                ndcg, n_used_test = ndcg_at_k_dense(y_test_true, test_scores, k=k)
            assert t.dt is not None
            t_ndcg_test[k] = t.dt
            test_metrics[f"ndcg@{k}"] = ndcg

        with _Timer() as t_f1:
            f1, _ = f1_at_k_dense(y_test_true, test_scores, k=5)
        test_metrics["f1@5"] = f1

        epoch_dt = time.perf_counter() - epoch_t0

        def _dt(timer: _Timer) -> float:
            return float(timer.dt) if timer.dt is not None else 0.0

        current = _early_stop_score(train_ndcg10, train_ndcg1000)

        # Diagnostics scalars for this epoch (cheap)
        with torch.no_grad():
            alpha_now = float(model.alpha().detach().cpu())
            rms_alpha_delta_now = _rms((model.alpha() * model.delta_w()).detach().cpu())

        print(
            f"Epoch {epoch:02d} timing | "
            f"train_step={_dt(t_train_step):.3f}s | "
            f"pred_train={_dt(t_pred_train):.3f}s | "
            f"ndcg_train@10={train_ndcg10:.6f} ndcg_train@1000={train_ndcg1000:.6f} sel={current:.6f} | "
            f"pred_test={_dt(t_pred_test):.3f}s | "
            f"ndcg_test@10={t_ndcg_test.get(10, 0.0):.3f}s ndcg_test@1000={t_ndcg_test.get(1000, 0.0):.3f}s | "
            f"f1@5={_dt(t_f1):.3f}s | "
            f"loss={loss.item():.6f} | "
            f"alpha={alpha_now:.4f} | "
            f"rms(alpha*delta_w)={rms_alpha_delta_now:.6f} | "
            f"total={epoch_dt:.3f}s"
        )

        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_alpha = alpha_now
            best_rms_alpha_delta = rms_alpha_delta_now
            best_loss = float(loss.item())

            # Compute full train metrics only for the best epoch snapshot
            full_train_scores = _predict_in_batches(model, X_train)
            best_train_metrics = {}
            for k in K_VALUES:
                ndcg, n_used_train_full = ndcg_at_k_dense(
                    y_train_true, full_train_scores, k=k
                )
                best_train_metrics[f"ndcg@{k}"] = ndcg
            best_n_used_train = n_used_train_full

            best_test_metrics = test_metrics.copy()
            best_n_used_test = n_used_test
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
    assert best_alpha is not None
    assert best_rms_alpha_delta is not None
    assert best_loss is not None

    model.load_state_dict(best_state)

    if update_scoreboard:
        update_markdown_scoreboard(
            path=scoreboard_path,
            model="torch_per_label_freq_gate",
            dataset="train",
            metrics=best_train_metrics,
            n_samples=best_n_used_train,
            epoch=best_epoch,
        )
        update_markdown_scoreboard(
            path=scoreboard_path,
            model="torch_per_label_freq_gate",
            dataset="test",
            metrics=best_test_metrics,
            n_samples=best_n_used_test,
            epoch=best_epoch,
        )

    print(
        "\nFinal test metrics | "
        f"ndcg@10={best_test_metrics['ndcg@10']:.6f} | "
        f"ndcg@1000={best_test_metrics['ndcg@1000']:.6f} | "
        f"f1@5={best_test_metrics['f1@5']:.6f} | "
        f"epoch={best_epoch} | "
        f"alpha_max={alpha_max:.6f} | "
        f"lambda_delta={lambda_delta:.6g} | "
        f"alpha(best)={best_alpha:.6f} | "
        f"rms(alpha*delta_w)(best)={best_rms_alpha_delta:.6f} | "
        f"loss(best)={best_loss:.6f}"
    )

    _print_diagnostics(
        model=model,
        label_counts=label_counts,
        model_names=["bonsai", "fasttext", "mllm"],
    )

    if update_scoreboard:
        print("\nSaved best result to SCOREBOARD.md")

    return {
        "alpha_max": float(alpha_max),
        "lambda_delta": float(lambda_delta),
        "best_epoch": int(best_epoch),
        "best_sel_metric": float(best_metric),
        "best_loss": float(best_loss),
        "alpha_best": float(best_alpha),
        "rms_alpha_delta_best": float(best_rms_alpha_delta),
        "test_ndcg10": float(best_test_metrics["ndcg@10"]),
        "test_ndcg1000": float(best_test_metrics["ndcg@1000"]),
        "test_f1_5": float(best_test_metrics["f1@5"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a small grid over (alpha_max, lambda_delta) and keep the best by the early-stop metric.",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=ALPHA_MAX_DEFAULT,
        help="alpha_max for bounded alpha (single-run mode).",
    )
    parser.add_argument(
        "--lambda-delta",
        type=float,
        default=LAMBDA_DELTA_DEFAULT,
        help="Regularization strength for applied residual (single-run mode).",
    )
    args = parser.parse_args()

    if not args.sweep:
        _train_one(
            alpha_max=float(args.alpha_max),
            lambda_delta=float(args.lambda_delta),
            update_scoreboard=True,
        )
        return

    # Sweep grid tuned for the *applied-correction* regularizer:
    #   loss = BCE + lambda_delta * mean((alpha * delta_w)^2)
    #
    # Empirically, useful lambda_delta is typically ~0.03–0.1 (sometimes up to ~0.3)
    # when alpha is ~0.1 and alpha_max ~0.5.
    alpha_max_grid = [0.25, 0.5, 0.75]
    lambda_grid = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]

    results: list[dict[str, object]] = []
    best = None

    for amax in alpha_max_grid:
        for lam in lambda_grid:
            print("\n" + "=" * 80)
            print(f"SWEEP RUN | alpha_max={amax} | lambda_delta={lam}")
            print("=" * 80 + "\n")

            r = _train_one(alpha_max=amax, lambda_delta=lam, update_scoreboard=False)
            results.append(r)

            if best is None or float(r["best_sel_metric"]) > float(best["best_sel_metric"]):
                best = r

    assert best is not None

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY (sorted by selection metric desc)")
    print("=" * 80)
    results_sorted = sorted(results, key=lambda x: float(x["best_sel_metric"]), reverse=True)
    for r in results_sorted:
        print(
            f"sel={float(r['best_sel_metric']):.6f} | "
            f"epoch={int(r['best_epoch']):02d} | "
            f"alpha_max={float(r['alpha_max']):.3f} | "
            f"lambda_delta={float(r['lambda_delta']):.3f} | "
            f"alpha(best)={float(r['alpha_best']):.4f} | "
            f"rms(alpha*delta_w)(best)={float(r['rms_alpha_delta_best']):.6f} | "
            f"loss(best)={float(r['best_loss']):.6f} | "
            f"test ndcg@10={float(r['test_ndcg10']):.6f} | "
            f"test ndcg@1000={float(r['test_ndcg1000']):.6f} | "
            f"test f1@5={float(r['test_f1_5']):.6f}"
        )

    print("\nBest config by selection metric:")
    print(
        f"alpha_max={float(best['alpha_max']):.6f} | "
        f"lambda_delta={float(best['lambda_delta']):.6g} | "
        f"epoch={int(best['best_epoch'])} | "
        f"sel={float(best['best_sel_metric']):.6f} | "
        f"alpha(best)={float(best['alpha_best']):.6f} | "
        f"rms(alpha*delta_w)(best)={float(best['rms_alpha_delta_best']):.6f} | "
        f"loss(best)={float(best['best_loss']):.6f}"
    )

    # Re-run best config and update scoreboard
    print("\nRe-running best config to update SCOREBOARD.md...\n")
    _train_one(
        alpha_max=float(best["alpha_max"]),
        lambda_delta=float(best["lambda_delta"]),
        update_scoreboard=True,
    )


if __name__ == "__main__":
    main()
