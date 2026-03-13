import torch
import torch.nn as nn


class Torch3Stage(nn.Module):
    """
    Stage 1 (initial implementation): per-label learned weighted mean over base model scores.

    Input:  (batch, M, L) raw base model scores in [0, 1]
    Output: (batch, L) probabilities in [0, 1]

    Notes:
    - Mirrors MeanWeightedConv1D (torch_mean), but uses raw inputs (no log1p).
    - Kept as its own module to allow future Stage 2/3 expansions.
    """

    output_type = "probabilities"

    def __init__(self, n_models: int = 3):
        super().__init__()
        if n_models < 1:
            raise ValueError("n_models must be positive")
        self.n_models = int(n_models)
        self.conv = nn.Conv1d(
            in_channels=self.n_models,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / float(self.n_models))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x).squeeze(1)
        return torch.clamp(out, min=0.0, max=1.0)
