import torch
import torch.nn as nn


class MeanWeightedConv1D(nn.Module):
    """
    Input:  (batch, 3, L)
    Output: (batch, L)
    """

    output_type = "probabilities"

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=3,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x).squeeze(1)
        return torch.clamp(out, min=0.0, max=1.0)
