import torch


def get_device() -> torch.device:
    """
    Return CUDA device if available, otherwise CPU.

    Centralized here so all benchmark scripts behave consistently.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
