import torch
import torch.nn as nn


class Torch3Stage(nn.Module):
    """
    Stage 1 (probability space): learned linear combination over base model scores.

    This version keeps the original probability-space intent (inputs in [0,1],
    typically gamma-corrected externally), but adds torch_mean_residual-style
    capacity and stabilization:
      - global per-model weights (shared across labels)
      - per-label residual weights (strongly regularized toward 0)
      - per-label bias (logit/score space)
      - explicit regularization helpers (global L2, delta L2, bias L2)

    Form:
        score[b, l] = sum_m (w_global[m] + delta_w[m, l]) * x[b, m, l] + bias[l]

    Notes:
    - Outputs are *scores* (not necessarily calibrated probabilities). For ranking
      metrics, only ordering matters.
    - No softmax: weights are unconstrained (can be negative, need not sum to 1).
    """

    output_type = "score"

    def __init__(
        self,
        *,
        n_models: int = 3,
        n_labels: int,
        init_global: torch.Tensor | None = None,
    ):
        super().__init__()
        if n_models < 1:
            raise ValueError("n_models must be positive")
        if n_labels < 1:
            raise ValueError("n_labels must be positive")
        self.n_models = int(n_models)
        self.n_labels = int(n_labels)

        if init_global is None:
            w0 = torch.full((self.n_models,), 1.0 / float(self.n_models), dtype=torch.float32)
        else:
            if init_global.ndim != 1 or init_global.shape[0] != self.n_models:
                raise ValueError(
                    f"init_global must have shape ({self.n_models},), got {tuple(init_global.shape)}"
                )
            w0 = init_global.to(dtype=torch.float32).clone()
            s = float(w0.sum().item())
            if not torch.isfinite(w0).all() or not (s > 0.0):
                raise ValueError("init_global must be finite and sum to a positive value")
            w0 = w0 / w0.sum()

        # Keep a non-trainable copy for anchor regularization.
        self.register_buffer("init_global_w", w0.clone(), persistent=False)

        self.global_w = nn.Parameter(w0)  # (M,)
        self.delta_w = nn.Parameter(torch.zeros((self.n_models, self.n_labels), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((self.n_labels,), dtype=torch.float32))

    def effective_w(self) -> torch.Tensor:
        """Return effective weights of shape (M, L)."""
        return self.global_w[:, None] + self.delta_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, M, L) scores
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, M, L), got {tuple(x.shape)}")
        if x.shape[1] != self.n_models or x.shape[2] != self.n_labels:
            raise ValueError(
                f"Expected x with (M={self.n_models}, L={self.n_labels}), got {tuple(x.shape)}"
            )

        w_eff = self.effective_w().to(dtype=x.dtype, device=x.device)  # (M, L)
        bias = self.bias.to(dtype=x.dtype, device=x.device)  # (L,)
        out = (x * w_eff.unsqueeze(0)).sum(dim=1) + bias  # (B, L)
        return out

    def global_anchor_l2(self) -> torch.Tensor:
        """
        Penalize deviation of global weights from their initialization.

        This tends to preserve "mean-like" behavior and prevents global_w from drifting
        into degenerate regimes (e.g. negative weights) early in training.
        """
        return ((self.global_w - self.init_global_w) ** 2).mean()

    def delta_l2(self) -> torch.Tensor:
        return (self.delta_w ** 2).mean()

    def bias_l2(self) -> torch.Tensor:
        return (self.bias ** 2).mean()
