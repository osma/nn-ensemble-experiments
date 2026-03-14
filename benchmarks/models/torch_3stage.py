import torch
import torch.nn as nn


class Torch3Stage(nn.Module):
    """
    Stage 1 (probability space): learned linear combination over base model scores.

    Input:  (batch, M, L) base model scores in [0, 1] (optionally preprocessed)
    Output: (batch, L) probabilities in [0, 1]

    Notes:
    - Weights are unconstrained (no softmax): they do NOT have to sum to 1 and may be negative.
    - Output is clamped to [0, 1] for use with BCELoss (requested).
    - No per-source bias term (by design) to avoid shifting probability-like
      inputs into an inconsistent space.
    """

    output_type = "prob"

    def __init__(self, n_models: int = 3):
        super().__init__()
        if n_models < 1:
            raise ValueError("n_models must be positive")
        self.n_models = int(n_models)

        # Trainable unconstrained weights (no normalization).
        # Initialize to equal weights.
        self.w = nn.Parameter(
            torch.full((self.n_models,), 1.0 / float(self.n_models), dtype=torch.float32)
        )

    def effective_w(self) -> torch.Tensor:
        """Return the learned weights of shape (M,)."""
        return self.w

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
        return torch.clamp(out, min=0.0, max=1.0)
