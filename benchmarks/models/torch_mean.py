import torch
import torch.nn as nn


class MeanWeightedConv1D(nn.Module):
    """
    Input:  (batch, M, L)
    Output: (batch, L)
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
