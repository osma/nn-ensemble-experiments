import torch
import torch.nn as nn


class PerLabelWeightedEnsemble(nn.Module):
    """
    Per-label weighted ensemble with bias.

    For each label l:
        score[l] = sum_m w[m, l] * x[m, l] + b[l]

    Input:
        x: (batch, M, L)
            M = number of base models
            L = number of labels

    Output:
        (batch, L) raw logits

    Notes:
    - Returns raw logits (no clamp, no sigmoid).
    - Intended for use with BCEWithLogitsLoss or ranking-aware losses.
    """

    def __init__(self, n_models: int, n_labels: int):
        super().__init__()
        self.n_models = n_models
        self.n_labels = n_labels

        # Per-model, per-label weights
        self.weights = nn.Parameter(
            torch.full((n_models, n_labels), 1.0 / n_models)
        )

        # Per-label bias
        self.bias = nn.Parameter(torch.zeros(n_labels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, n_models, n_labels)
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input of shape (batch, n_models, n_labels), got {x.shape}"
            )
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected input with n_models={self.n_models}, "
                f"n_labels={self.n_labels}, got {x.shape}"
            )

        # Broadcast weights to (batch, n_models, n_labels)
        weighted = x * self.weights.unsqueeze(0)
        out = weighted.sum(dim=1) + self.bias
        return out
