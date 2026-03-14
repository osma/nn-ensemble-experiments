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


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())
