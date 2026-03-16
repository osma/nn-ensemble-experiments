from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix


def csr_to_log1p_tensor(csr: csr_matrix) -> torch.Tensor:
    """
    Convert a CSR matrix to a dense torch tensor with fixed log1p preprocessing.

    This transform is intentionally applied OUTSIDE the model to avoid
    optimization undoing the calibration benefits.
    """
    x = torch.from_numpy(csr.toarray()).float()
    return torch.log1p(torch.clamp(x, min=0.0))


def csr_to_raw_tensor(csr: csr_matrix) -> torch.Tensor:
    """
    Convert a CSR matrix to a dense torch tensor with NO preprocessing.

    Assumes CSR values are already valid raw scores (guaranteed non-negative and
    in [0, 1] per repo policy for torch_3stage stage-1).
    """
    return torch.from_numpy(csr.toarray()).float()


def csr_to_logit_tensor(csr: csr_matrix, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert a CSR matrix of probabilities in [0, 1] to a dense torch tensor of logits.

    Values are clamped to [eps, 1-eps] to avoid +/-inf logits.

    Notes:
    - This is intended for probability-space base predictors (e.g. Annif outputs).
    - For ranking metrics, logits are valid scores (only ordering matters).
    """
    x = torch.from_numpy(csr.toarray()).float()
    x = torch.clamp(x, min=float(eps), max=1.0 - float(eps))
    return torch.logit(x)


def fit_source_gamma_from_csr(
    matrices: list[csr_matrix],
    *,
    quantile: float = 0.99,
    target: float = 0.3,
    eps: float = 1e-12,
    sample_n: int = 2_000_000,
    seed: int = 1337,
    clip_gamma: tuple[float, float] = (0.5, 2.0),
) -> list[float]:
    """
    Fit per-source gamma values from training prediction distributions (no labels).

    For each source m we estimate q = quantile(p, quantile) over its scores p in [0, 1],
    then choose gamma such that q**gamma == target:

        gamma = log(target) / log(q)

    Intuition:
      - If a source is usually very small (q << 1), then gamma < 1 expands low values.
      - If a source is usually large (q ~ 1), then gamma > 1 compresses values.

    Implementation notes:
      - Works on CSR values only (typically the non-zeros). This is intentional: including
        implicit zeros would make almost all quantiles equal to 0 for sparse matrices.
      - Uses random sampling when nnz is large to keep runtime/memory bounded.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0,1)")
    if not (0.0 < target < 1.0):
        raise ValueError("target must be in (0,1)")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if sample_n <= 0:
        raise ValueError("sample_n must be positive")
    g_lo, g_hi = clip_gamma
    if not (g_lo > 0 and g_hi > 0 and g_lo <= g_hi):
        raise ValueError("clip_gamma must be positive and (lo <= hi)")

    rng = np.random.default_rng(int(seed))
    gammas: list[float] = []

    for m, csr in enumerate(matrices):
        data = np.asarray(csr.data, dtype=np.float64)
        if data.size == 0:
            # Degenerate source: all zeros. Gamma doesn't matter; pick identity.
            gammas.append(1.0)
            continue

        if data.size > sample_n:
            idx = rng.choice(data.size, size=sample_n, replace=False)
            sample = data[idx]
        else:
            sample = data

        # Base model scores are guaranteed to be in [0, 1]; clip defensively anyway.
        sample = np.clip(sample, 0.0, 1.0)

        q = float(np.quantile(sample, quantile))
        q = float(np.clip(q, eps, 1.0 - eps))

        # If q is extremely close to 1, log(q) is ~0 and gamma can explode.
        # Treat that as "already high-scale": use gamma=1 (identity).
        denom = float(np.log(q))
        if abs(denom) < 1e-12:
            gamma = 1.0
        else:
            gamma = float(np.log(float(target)) / denom)

        gamma = float(np.clip(gamma, g_lo, g_hi))
        gammas.append(gamma)

    return gammas


def csr_to_gamma_tensor(csr: csr_matrix, *, gamma: float) -> torch.Tensor:
    """
    Convert a CSR matrix of probabilities in [0,1] to a dense torch tensor after
    per-source gamma correction: p -> p**gamma.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    x = torch.from_numpy(csr.toarray()).float()
    x = torch.clamp(x, min=0.0, max=1.0)
    return torch.pow(x, float(gamma))


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())
