import torch
import torch.nn as nn


class Torch3Stage(nn.Module):
    """
    Stage 1 (probability-like space): softmax-weighted mean over base model scores.

    Input:  (batch, M, L) base model scores in [0, 1] (optionally preprocessed)
    Output: (batch, L) unbounded logits-like scores

    Notes:
    - Mixing weights are constrained by softmax (positive, sum to 1).
    - No per-source bias term (by design) to avoid shifting probability-like
      inputs into an inconsistent space.
    """

    output_type = "logits"

    def __init__(self, n_models: int = 3):
        super().__init__()
        if n_models < 1:
            raise ValueError("n_models must be positive")
        self.n_models = int(n_models)

        # Trainable logits for mixture weights (softmaxed in forward).
        # Initialize to equal weights.
        self.alpha = nn.Parameter(torch.zeros(self.n_models, dtype=torch.float32))

    def effective_w(self) -> torch.Tensor:
        """Return softmax-normalized weights of shape (M,)."""
        return torch.softmax(self.alpha, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, L) scores
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, M, L), got {tuple(x.shape)}")
        if x.shape[1] != self.n_models:
            raise ValueError(
                f"Expected x.shape[1] (M) == n_models={self.n_models}, got {x.shape[1]}"
            )

        w = self.effective_w().to(dtype=x.dtype, device=x.device)  # (M,)
        out = (x * w.view(1, -1, 1)).sum(dim=1)  # (B, L)
        return out
