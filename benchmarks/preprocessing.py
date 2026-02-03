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


def tensor_to_csr(t: torch.Tensor) -> csr_matrix:
    return csr_matrix(t.detach().cpu().numpy())
