# Simplification Experiments for `torch_per_label_softmax_global_active_lowrank`

This document specifies 10 simplifications of the current #1 model
(`torch_per_label_softmax_global_active_lowrank`, weighted avg = 0.535017).
Each experiment changes 1–2 aspects of the original architecture or hyperparameters.
The goal is to discover whether any complexity can be removed without sacrificing
performance.

---

## Reference: Original Champion Architecture

Before describing the simplifications, here is a complete summary of the original
model's architecture, parameters, and hyperparameters.

### Model class: `ActiveLowRankEnsemble`

**Learned parameters:**

| Parameter   | Shape              | Init                                    | Description                              |
|-------------|--------------------|-----------------------------------------|------------------------------------------|
| `g_raw`     | (M,)               | `log(clamp(init_global, 1e-12))`        | Unconstrained logits → `softmax(g_raw)` gives global mixture weights |
| `w_delta`   | (M, L)             | zeros                                   | Per-label residual weights               |
| `bias`      | (L,)               | zeros                                   | Per-label bias                           |
| `U`         | (L_active, rank)   | `0.01 * randn`                          | Low-rank: label → latent projection      |
| `V`         | (L_active, rank)   | `0.01 * randn`                          | Low-rank: latent → label projection      |
| `W`         | (n_channels, rank) | `0.01 * randn`                          | Low-rank: channel mixing (n_channels = M+1) |
| `raw_gate`  | scalar             | `log(0.1 / 0.9)` (so sigmoid = 0.1)    | Learnable gate amplitude                 |

**Forward pass:**

```
global_w = softmax(g_raw)                          # (M,)
w_eff = global_w[:, None] + w_delta                # (M, L)
base_logits = sum_m(x * w_eff, dim=m) + bias       # (B, L)

# Low-rank cross-label mixer (active labels only):
x_active = x[:, :, active_idx]                     # (B, M, L_active)
base_logits_active = base_logits[:, active_idx]     # (B, L_active)
feats = cat([x_active, base_logits_active.unsqueeze(1)], dim=1)  # (B, M+1, L_active)

# LowRankActiveMixer forward:
Z = feats.reshape(B*C, L_active) @ U               # (B*C, rank)
Z = Z.reshape(B, C, rank)
h = sum(Z * W.unsqueeze(0), dim=1)                 # (B, rank)
delta = h @ V.T                                     # (B, L_active)

# Safety rails:
delta = 0.5 * tanh(delta / 0.5)                    # tanh clamp to [-0.5, 0.5]
delta = delta - delta.mean(dim=1, keepdim=True)     # per-sample centering
gate = sigmoid(raw_gate) * 0.2                      # gate in [0, 0.2]
gated_delta = delta * gate

# Multiplicative application:
out_active = base_logits_active * (1 + gated_delta)
out = base_logits; out[:, active_idx] = out_active
```

**Loss:**

```
loss = BCEWithLogitsLoss(logits, targets) + LAMBDA_DELTA_L2 * mean(w_delta²)
```

**Hyperparameters:**

| Hyperparameter       | Value  | Scope                          |
|----------------------|--------|--------------------------------|
| `BEST_LR`            | 0.003  | g_raw, w_delta, bias           |
| `BEST_WEIGHT_DECAY`  | 0.0    | g_raw, w_delta, bias           |
| `LOWRANK_LR`         | 1e-4   | U, V, W, raw_gate              |
| `LOWRANK_WEIGHT_DECAY`| 1e-2  | U, V, W (not raw_gate)         |
| `LAMBDA_DELTA_L2`    | 1e-3   | Explicit L2 on w_delta         |
| `DEFAULT_RANK`       | 64     | Low-rank bottleneck dimension  |
| `DELTA_CLAMP`        | 0.5    | tanh saturation bound          |
| `DELTA_GATE_MAX`     | 0.2    | Maximum gate amplitude         |
| `BEST_BATCH_SIZE`    | 256    | Training batch size            |
| `EPOCHS`             | 20     | Maximum epochs                 |
| `PATIENCE`           | 3      | Early stopping patience        |
| `MIN_EPOCHS`         | 2      | Minimum epochs before stopping |
| `EARLY_STOP_EVAL_ROWS`| 512   | Train subset for early stopping |

**Active label mask:** Union of labels with nonzero truth OR nonzero predictions in training data.

**Dataset-specific init:** `init_global` from `DatasetConfig.ensemble3_init_weights` (normalized to sum=1).

---

## Simplification 1: `simplify_no_delta`

**Name:** `torch_per_label_softmax_global_active_lowrank_no_delta`

**Hypothesis:** The per-label weight residuals (`w_delta`) contribute M×L parameters
and an explicit L2 penalty hyperparameter (`LAMBDA_DELTA_L2`). If the low-rank mixer
already captures label-specific adjustments, `w_delta` may be redundant.

**Changes vs. original (1 change):**

1. **Remove `w_delta` entirely.** Effective weights become simply `softmax(g_raw)[:, None]`
   broadcast to all labels. Remove the `LAMBDA_DELTA_L2` hyperparameter and the
   `delta_l2()` regularization term from the loss.

**Concrete spec:**

- Model `__init__`: Remove `self.w_delta = nn.Parameter(...)`. Remove `LAMBDA_DELTA_L2`.
- `effective_w()`: Returns `self.global_w()[:, None].expand(self.n_models, self.n_labels)`.
- `forward()`: `base_logits = (x * w_eff).sum(dim=1) + self.bias` (w_eff has no delta).
- `delta_l2()`: Remove entirely.
- Loss: `loss = BCEWithLogitsLoss(logits, targets)` — no explicit L2 term.
- Optimizer: Two-tiered as before, but base group has only `[model.g_raw, model.bias]`.

**All other architecture, hyperparameters, and training logic remain identical.**

---

## Simplification 2: `simplify_fixed_gate`

**Name:** `torch_per_label_softmax_global_active_lowrank_fixed_gate`

**Hypothesis:** The learnable gate (`raw_gate`) introduces a parameter and
`DELTA_GATE_MAX` hyperparameter. The gate starts at `sigmoid(log(0.1/0.9)) * 0.2 = 0.02`
and may not move far from there during training. Fixing it at a constant may be equivalent.

**Changes vs. original (1 change):**

1. **Replace learnable gate with a fixed constant `gate = 0.02`.** Remove `raw_gate`
   parameter, `DELTA_GATE_MAX` hyperparameter.

**Concrete spec:**

- Model `__init__`: Remove `self.raw_gate = nn.Parameter(...)`.
- `get_lowrank_delta()`: Replace the gate computation:
  ```python
  # Before:
  # DELTA_GATE_MAX = 0.2
  # gate = torch.sigmoid(self.raw_gate) * DELTA_GATE_MAX
  # After:
  FIXED_GATE = 0.02
  return delta_active * FIXED_GATE
  ```
- Optimizer: Remove `raw_gate` from the lowrank param group. The lowrank group
  becomes just `model.lowrank.parameters()`.

**All other architecture, hyperparameters, and training logic remain identical.**

---

## Simplification 3: `simplify_rank16`

**Name:** `torch_per_label_softmax_global_active_lowrank_rank16`

**Hypothesis:** Rank 64 may be excessive for a 3-model ensemble. Reducing to rank 16
cuts low-rank parameters by 4× (U, V, W all shrink) while potentially retaining the
most important cross-label correlations.

**Changes vs. original (1 change):**

1. **Set `DEFAULT_RANK = 16`** instead of 64.

**Concrete spec:**

- Change only the constant: `DEFAULT_RANK = 16`.
- All parameter shapes that depend on rank (U, V, W) adjust automatically.
- Parameter reduction: U goes from (L_active, 64) to (L_active, 16), same for V.
  W goes from (M+1, 64) to (M+1, 16).

**All other architecture, hyperparameters, and training logic remain identical.**

---

## Simplification 4: `simplify_no_base_channel`

**Name:** `torch_per_label_softmax_global_active_lowrank_no_base_ch`

**Hypothesis:** The low-rank mixer receives M+1 channels: the M raw model inputs plus
the base logits. The base logits are a linear combination of the raw inputs, so they
may be redundant. Removing this channel simplifies the mixer and eliminates a circular
dependency between base and mixer.

**Changes vs. original (1 change):**

1. **Use only the M raw model inputs as channels** (remove base_logits from features).
   `n_channels = M` instead of `M + 1`.

**Concrete spec:**

- Model `__init__`: Change `LowRankActiveMixer` construction:
  ```python
  self.lowrank = LowRankActiveMixer(n_channels=self.n_models, n_active=self.n_active, rank=rank)
  ```
  W shape becomes (M, rank) instead of (M+1, rank).
- `get_lowrank_delta()`: Change feature construction:
  ```python
  # Before:
  # feats = torch.cat([x_active, base_logits_active.unsqueeze(1)], dim=1)  # (B, M+1, L_active)
  # After:
  feats = x_active  # (B, M, L_active) — raw model scores only
  ```
- Method signature: `get_lowrank_delta(self, x_active: torch.Tensor) -> torch.Tensor`
  (remove `base_logits_active` argument).
- `forward()`: Call as `self.get_lowrank_delta(x_active)`.
- `_delta_stats()`: Update accordingly to not pass base_logits_active.

**All other architecture, hyperparameters, and training logic remain identical.**

---

## Simplification 5: `simplify_single_lr`

**Name:** `torch_per_label_softmax_global_active_lowrank_single_lr`

**Hypothesis:** The two-tiered optimizer (base LR=0.003, lowrank LR=1e-4) adds
2 hyperparameters (`LOWRANK_LR`, `LOWRANK_WEIGHT_DECAY`). A single optimizer
group with an intermediate learning rate may work equally well.

**Changes vs. original (2 changes):**

1. **Use a single optimizer with one LR for all parameters.** Remove `LOWRANK_LR`
   and `LOWRANK_WEIGHT_DECAY` hyperparameters.
2. **Set a single learning rate of `1e-3`** (geometric mean of 0.003 and 1e-4,
   rounded) with `weight_decay=0.0`.

**Concrete spec:**

- Remove constants `LOWRANK_LR = 1e-4` and `LOWRANK_WEIGHT_DECAY = 1e-2`.
- Change `BEST_LR = 1e-3`.
- Optimizer construction:
  ```python
  optimizer = optim.AdamW(
      model.parameters(),
      lr=1e-3,
      weight_decay=0.0,
      eps=1e-8,
  )
  ```
- Keep `LAMBDA_DELTA_L2 = 1e-3` as the explicit regularizer (unchanged).

**All other architecture and training logic remain identical.**

---

## Simplification 6: `simplify_no_centering`

**Name:** `torch_per_label_softmax_global_active_lowrank_no_center`

**Hypothesis:** Per-sample centering (`delta = delta - delta.mean(dim=1)`) was
identified as critical in the MLP variants (removing it caused "severe regression"
per EXPERIMENTS.md). However, the low-rank structure with small rank may be
inherently more constrained and not need this safety rail.

**Changes vs. original (1 change):**

1. **Remove per-sample centering** from `get_lowrank_delta()`.

**Concrete spec:**

- In `get_lowrank_delta()`, remove the line:
  ```python
  delta_active = delta_active - delta_active.mean(dim=1, keepdim=True)
  ```
  The tanh clamp and gate still apply.

**All other architecture, hyperparameters, and training logic remain identical.**

> **Note:** EXPERIMENTS.md documents that removing centering caused "severe regression"
> for the MLP variant. This tests whether the same holds for the low-rank variant.

---

## Simplification 7: `simplify_additive`

**Name:** `torch_per_label_softmax_global_active_lowrank_additive`

**Hypothesis:** The multiplicative application `base * (1 + gated_delta)` is more
complex than additive `base + gated_delta`. Multiplicative scaling biases adjustments
toward labels with large base logits. Additive application treats all labels equally,
which may be simpler and more robust.

**Changes vs. original (1 change):**

1. **Replace multiplicative with additive delta application.**

**Concrete spec:**

- In `forward()`, change:
  ```python
  # Before:
  # out_active = base_logits_active * (1.0 + gated_delta_active)
  # After:
  out_active = base_logits_active + gated_delta_active
  ```
- Since the delta is now added directly to logits rather than multiplied,
  increase `DELTA_GATE_MAX` from 0.2 to 1.0 to give it comparable effective
  scale. The tanh clamp ([-0.5, 0.5]) combined with gate max 1.0 means the
  additive correction is bounded to [-0.5, 0.5] per label per sample.

**All other architecture, hyperparameters, and training logic remain identical.**

---

## Simplification 8: `simplify_symmetric_lowrank`

**Name:** `torch_per_label_softmax_global_active_lowrank_symmetric`

**Hypothesis:** The low-rank mixer uses separate U and V matrices for projecting
labels into and out of the latent space. Tying them (U = V, symmetric factorization)
halves the label-side parameters and acts as a structural regularizer, enforcing
that the cross-label similarity used for mixing is symmetric.

**Changes vs. original (1 change):**

1. **Tie U and V:** remove V as a separate parameter; use U for both projections.

**Concrete spec:**

- `LowRankActiveMixer.__init__()`: Remove `self.V`. Keep `self.U` and `self.W`.
  ```python
  self.U = nn.Parameter(0.01 * torch.randn(self.n_active, self.rank))
  # self.V removed
  self.W = nn.Parameter(0.01 * torch.randn(self.n_channels, self.rank))
  ```
- `LowRankActiveMixer.forward()`: Use U for both projection steps:
  ```python
  Z = x2 @ self.U                                    # (B*C, rank)
  Z = Z.reshape(B, self.n_channels, self.rank)
  h = torch.sum(Z * self.W.unsqueeze(0), dim=1)      # (B, rank)
  delta = h @ self.U.t()                              # (B, L_active) — use U.t() instead of V.t()
  ```

**All other architecture, hyperparameters, and training logic remain identical.**

---

## Simplification 9: `simplify_no_clamp`

**Name:** `torch_per_label_softmax_global_active_lowrank_no_clamp`

**Hypothesis:** The tanh clamp (`0.5 * tanh(delta / 0.5)`) bounds the raw delta
to [-0.5, 0.5]. Combined with the gate (max 0.2), the effective delta range is
already very small (max ≈ 0.1). The gate alone, plus weight decay on the low-rank
parameters, may be sufficient to prevent instability. Removing the tanh allows
slightly larger gradients and may speed convergence.

**Changes vs. original (1 change):**

1. **Remove the tanh clamp** from `get_lowrank_delta()`.

**Concrete spec:**

- In `get_lowrank_delta()`, remove:
  ```python
  DELTA_CLAMP = 0.5
  delta_active = DELTA_CLAMP * torch.tanh(delta_active / DELTA_CLAMP)
  ```
  Keep centering and gating as-is. The delta passes directly from the low-rank
  output through centering and gating.

**All other architecture, hyperparameters, and training logic remain identical.**

---

## Simplification 10: `simplify_minimal_lowrank`

**Name:** `torch_per_label_softmax_global_active_lowrank_minimal`

**Hypothesis:** Combines two simplifications: removing `w_delta` (Simplification 1)
and using a single learning rate (Simplification 5). This is the most aggressive
simplification — it reduces the model to just softmax global weights + bias + low-rank
mixer, trained with a single optimizer, removing 3 hyperparameters
(`LAMBDA_DELTA_L2`, `LOWRANK_LR`, `LOWRANK_WEIGHT_DECAY`).

**Changes vs. original (2 changes):**

1. **Remove `w_delta` entirely** (same as Simplification 1).
2. **Use a single learning rate of `1e-3` with `weight_decay=1e-3`** for all
   parameters (same as Simplification 5, but with light weight decay to compensate
   for the removed explicit L2 term).

**Concrete spec:**

- Model `__init__`: Remove `self.w_delta = nn.Parameter(...)`.
- `effective_w()`: Returns `self.global_w()[:, None].expand(self.n_models, self.n_labels)`.
- `delta_l2()`: Remove entirely.
- Remove constants `LAMBDA_DELTA_L2`, `LOWRANK_LR`, `LOWRANK_WEIGHT_DECAY`.
- Optimizer:
  ```python
  optimizer = optim.AdamW(
      model.parameters(),
      lr=1e-3,
      weight_decay=1e-3,
      eps=1e-8,
  )
  ```
- Loss: `loss = BCEWithLogitsLoss(logits, targets)` — no explicit L2 term.

**All other architecture and training logic remain identical.**

**Hyperparameter summary after simplification:**

| Hyperparameter       | Value  |
|----------------------|--------|
| `LR`                 | 1e-3   |
| `WEIGHT_DECAY`       | 1e-3   |
| `DEFAULT_RANK`       | 64     |
| `DELTA_CLAMP`        | 0.5    |
| `DELTA_GATE_MAX`     | 0.2    |
| `BATCH_SIZE`         | 256    |
| `EPOCHS`             | 20     |
| `PATIENCE`           | 3      |

Total unique hyperparameters: **8** (vs. original's **13**).

---

## Summary Table

| # | Name | What changes | Params removed | Hyperparams removed |
|---|------|-------------|----------------|---------------------|
| 1 | `simplify_no_delta` | Remove w_delta | M×L | LAMBDA_DELTA_L2 |
| 2 | `simplify_fixed_gate` | Fix gate=0.02 | raw_gate (1) | DELTA_GATE_MAX |
| 3 | `simplify_rank16` | Rank 64→16 | 75% of U,V,W | — |
| 4 | `simplify_no_base_ch` | Remove base_logits channel | rank params in W | — |
| 5 | `simplify_single_lr` | One optimizer group | — | LOWRANK_LR, LOWRANK_WEIGHT_DECAY |
| 6 | `simplify_no_center` | Remove per-sample centering | — | — |
| 7 | `simplify_additive` | Additive delta application | — | — (DELTA_GATE_MAX changed) |
| 8 | `simplify_symmetric` | Tie U=V | L_active×rank | — |
| 9 | `simplify_no_clamp` | Remove tanh clamp | — | DELTA_CLAMP |
| 10 | `simplify_minimal` | Remove w_delta + single LR | M×L + raw_gate | LAMBDA_DELTA_L2, LOWRANK_LR, LOWRANK_WEIGHT_DECAY |

---

## Implementation Notes

Each simplification should be implemented as a new `.py` file in `benchmarks/`
following the naming pattern `torch_per_label_softmax_global_active_lowrank_<suffix>.py`.
The model name on the scoreboard should follow the same pattern
`torch_per_label_softmax_global_active_lowrank_<suffix>(model1,model2,model3)`.

The implementation should copy the original `torch_per_label_softmax_global_active_lowrank.py`
and apply only the changes described above. All boilerplate (data loading, evaluation,
scoreboard update, early stopping) should remain identical.

Benchmark each simplification on all 3 datasets (yso-fi, yso-en, koko) using
`regenerate_scoreboard.sh` and compare weighted average to the original's **0.535017**.

---

## Simplification 11: `simplify_combined`

**Name:** `torch_per_label_softmax_global_active_lowrank_combined`

**Hypothesis:** Simplifications S2, S3, S4, S6, and S8 each individually retained
≥99.9% of the champion's performance while reducing complexity. If their effects
are largely orthogonal, combining all five should yield a dramatically simpler model
with comparable performance. The combined model removes ~8× low-rank parameters,
2 hyperparameters, and 3 architectural mechanisms (learnable gate, per-sample
centering, base-logits channel) while keeping the critical components (`w_delta`,
two-tiered LR, tanh clamp, multiplicative application) that failed simplifications
proved are essential.

**Included simplifications:**

| Source | Change | Individual result |
|--------|--------|-------------------|
| S6 | Remove per-sample centering | **Improvement** (+0.000273, new #1) |
| S2 | Fix gate at constant 0.02 | Negligible regression (−0.000216) |
| S8 | Symmetric factorization (tie U=V) | Minor regression (−0.000189) |
| S4 | Remove base_logits channel | Minimal regression (−0.000526) |
| S3 | Reduce rank 64 → 16 | Minimal regression (−0.000495) |

**Excluded simplifications (performance too fragile):**

| Source | Why excluded |
|--------|--------------|
| S1 (`no_delta`) | Major regression — `w_delta` is critical |
| S5 (`single_lr`) | Regression — two-tiered LR is important |
| S7 (`additive`) | Slight regression — multiplicative is superior |
| S9 (`no_clamp`) | Minor regression — tanh clamp is a useful safety rail |
| S10 (`minimal`) | Combines S1+S5 which both failed |

**Changes vs. original (5 changes):**

1. **Remove per-sample centering** from `get_lowrank_delta()` (S6).
2. **Replace learnable gate with fixed constant `gate = 0.02`**. Remove `raw_gate`
   parameter and `DELTA_GATE_MAX` hyperparameter (S2).
3. **Tie U and V** in `LowRankActiveMixer`: remove V, use U for both projections (S8).
4. **Remove the base_logits channel** from the mixer features. `n_channels = M`
   instead of `M + 1` (S4).
5. **Set `DEFAULT_RANK = 16`** instead of 64 (S3).

---

### Concrete spec

#### Constants (changes from original highlighted)

```python
EPOCHS = 20                      # unchanged
K_VALUES = (10, 1000)            # unchanged
PATIENCE = 3                     # unchanged
MIN_EPOCHS = 2                   # unchanged
EARLY_STOP_EVAL_ROWS = 512       # unchanged
EARLY_STOP_SEED = 1337           # unchanged
EVAL_BATCH_SIZE = 512            # unchanged
TRAIN_SEED = 0                   # unchanged

BEST_LR = 0.003                  # unchanged
BEST_WEIGHT_DECAY = 0.0          # unchanged
BEST_BATCH_SIZE = 256            # unchanged
LAMBDA_DELTA_L2 = 1e-3           # unchanged

LOWRANK_LR = 1e-4                # unchanged
LOWRANK_WEIGHT_DECAY = 1e-2      # unchanged

DEFAULT_RANK = 16                # CHANGED: was 64 (S3)
FIXED_GATE = 0.02                # NEW: replaces learnable raw_gate (S2)

# REMOVED: DELTA_GATE_MAX = 0.2  (S2)
```

#### `LowRankActiveMixer` class

```python
class LowRankActiveMixer(nn.Module):
    """
    Symmetric low-rank mixer: feats (B, C, L_active) -> delta (B, L_active).
    Uses U for both label->latent and latent->label projections (S8).
    """
    def __init__(self, n_channels: int, n_active: int, rank: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_active = n_active
        self.rank = rank

        self.U = nn.Parameter(0.01 * torch.randn(self.n_active, self.rank))
        # self.V REMOVED (S8: symmetric — use U for both projections)
        self.W = nn.Parameter(0.01 * torch.randn(self.n_channels, self.rank))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        if self.n_active == 0:
            return feats.new_zeros((B, 0))

        # (B*C, L_active) @ (L_active, rank) -> (B*C, rank)
        x2 = feats.reshape(B * self.n_channels, self.n_active)
        Z = x2 @ self.U
        Z = Z.reshape(B, self.n_channels, self.rank)

        # Mix channels
        h = torch.sum(Z * self.W.unsqueeze(0), dim=1)  # (B, rank)

        # Project back to labels using U.t() instead of V.t() (S8)
        delta = h @ self.U.t()  # (B, L_active)
        return delta
```

#### `ActiveLowRankEnsemble` class

**`__init__` changes:**

```python
def __init__(
    self,
    *,
    n_models: int,
    n_labels: int,
    active_idx: torch.Tensor,
    rank: int,
    init_global: torch.Tensor | None = None,
) -> None:
    super().__init__()
    self.n_models = int(n_models)
    self.n_labels = int(n_labels)

    if active_idx.ndim != 1:
        raise ValueError("active_idx must be 1D")
    self.register_buffer("active_idx", active_idx.long())
    self.n_active = int(active_idx.numel())

    # init_global handling — identical to original
    if init_global is None:
        g = torch.full((self.n_models,), 1.0 / self.n_models, dtype=torch.float32)
    else:
        if init_global.ndim != 1 or init_global.shape[0] != self.n_models:
            raise ValueError(
                f"init_global must have shape ({self.n_models},), got {tuple(init_global.shape)}"
            )
        g = init_global.to(dtype=torch.float32).clone()
        s = float(g.sum().item())
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("init_global must sum to a positive finite value")
        g = g / g.sum()

    self.g_raw = nn.Parameter(torch.log(torch.clamp(g, min=1e-12)))
    self.w_delta = nn.Parameter(torch.zeros(self.n_models, self.n_labels))  # KEPT (S1 failed)
    self.bias = nn.Parameter(torch.zeros(self.n_labels))

    # S4: n_channels = M (no base_logits channel). S3: rank=16 via DEFAULT_RANK.
    self.lowrank = LowRankActiveMixer(
        n_channels=self.n_models,  # CHANGED: was self.n_models + 1 (S4)
        n_active=self.n_active,
        rank=rank,
    )

    # REMOVED: self.raw_gate (S2 — replaced by FIXED_GATE constant)
```

**`global_w` and `effective_w` — unchanged:**

```python
def global_w(self) -> torch.Tensor:
    return torch.softmax(self.g_raw, dim=0)

def effective_w(self) -> torch.Tensor:
    return self.global_w()[:, None] + self.w_delta
```

**`get_lowrank_delta` — 3 changes (S2, S4, S6):**

```python
def get_lowrank_delta(self, x_active: torch.Tensor) -> torch.Tensor:
    # S4: use raw model scores only (no base_logits_active argument)
    feats = x_active  # (B, M, L_active)

    delta_active = self.lowrank(feats)

    # Tanh clamp KEPT (S9 was unsuccessful)
    DELTA_CLAMP = 0.5
    delta_active = DELTA_CLAMP * torch.tanh(delta_active / DELTA_CLAMP)

    # S6: per-sample centering REMOVED
    # (was: delta_active = delta_active - delta_active.mean(dim=1, keepdim=True))

    # S2: fixed gate replaces learnable gate
    return delta_active * FIXED_GATE
```

**`forward` — adjusted call site (S4):**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected input of shape (batch, n_models, n_labels), got {x.shape}")

    w_eff = self.effective_w().unsqueeze(0)  # (1, M, L)
    base_logits = (x * w_eff).sum(dim=1) + self.bias

    if self.n_active == 0:
        return base_logits

    x_active = x.index_select(dim=2, index=self.active_idx)
    # S4: base_logits_active no longer needed
    # (was: base_logits_active = base_logits.index_select(dim=1, index=self.active_idx))

    gated_delta_active = self.get_lowrank_delta(x_active)  # S4: no base_logits_active

    # Multiplicative application KEPT (S7 additive was inferior)
    base_logits_active = base_logits.index_select(dim=1, index=self.active_idx)
    out_active = base_logits_active * (1.0 + gated_delta_active)

    out = base_logits.clone()
    out.index_copy_(dim=1, index=self.active_idx, source=out_active)
    return out

def delta_l2(self) -> torch.Tensor:
    return (self.w_delta**2).mean()  # KEPT (S1 failed without w_delta)
```

#### `_delta_stats` helper — adjusted (S4)

```python
def _delta_stats(
    model: ActiveLowRankEnsemble, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[float, float]:
    # Same as original but call model.get_lowrank_delta(x_active)
    # without base_logits_active argument.
    sum_abs = 0.0
    n_abs = 0
    max_samples = 1_000_000
    samples: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                xb = batch[0]
            else:
                xb = batch
            xb = xb.to(device, non_blocking=True)

            x_active = xb.index_select(dim=2, index=model.active_idx)
            delta = model.get_lowrank_delta(x_active)  # S4: no base_logits_active
            a = delta.abs().detach().cpu().reshape(-1)

            sum_abs += float(a.sum().item())
            n_abs += int(a.numel())

            if max_samples > 0:
                remaining = max_samples - sum(int(s.numel()) for s in samples)
                if remaining <= 0:
                    max_samples = 0
                else:
                    if a.numel() <= remaining:
                        samples.append(a)
                    else:
                        idx = torch.randperm(a.numel())[:remaining]
                        samples.append(a.index_select(0, idx))

    if n_abs == 0:
        return 0.0, 0.0

    mean_abs = sum_abs / float(n_abs)
    if not samples:
        return float(mean_abs), 0.0

    v = torch.cat(samples, dim=0)
    v, _ = torch.sort(v)
    q_idx = min(int(round(0.95 * (v.numel() - 1))), v.numel() - 1)
    p95_abs = float(v[q_idx].item())
    return float(mean_abs), float(p95_abs)
```

#### Optimizer — adjusted (S2: remove `raw_gate` group)

```python
# Two-tiered LR KEPT (S5 single_lr was unsuccessful)
optimizer = optim.AdamW(
    [
        {"params": [model.g_raw, model.w_delta, model.bias], "lr": BEST_LR, "weight_decay": BEST_WEIGHT_DECAY},
        {"params": model.lowrank.parameters(), "lr": LOWRANK_LR, "weight_decay": LOWRANK_WEIGHT_DECAY},
        # REMOVED: raw_gate group (S2 — gate is now a constant)
    ],
    eps=1e-8,
)
```

#### Training loop — adjusted logging (S2: no `gate_val`)

```python
# In the epoch logging, replace gate_val with the constant:
delta_mean_abs, delta_p95_abs = _delta_stats(model, train_eval_loader, DEVICE)

print(
    f"Epoch {epoch:02d} | loss={float(last_loss or 0.0):.6f} | "
    f"train_ndcg@1000(subset)={train_ndcg1000:.6f} train_ndcg@10={train_ndcg10:.6f} | "
    f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
    f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
    f"test_f1@5={test_metrics['f1@5']:.6f} | "
    f"gate={FIXED_GATE:.4f}(fixed) LowRank_delta_mean={delta_mean_abs:.6f} p95={delta_p95_abs:.6f} | "
    f"total={epoch_dt:.3f}s"
)

# Remove the line:
# with torch.no_grad():
#     gate_val = torch.sigmoid(model.raw_gate).item()
```

#### `main()` function — only `model_name` changes

```python
model_name = f"torch_per_label_softmax_global_active_lowrank_combined({','.join(ensemble_keys)})"
```

**All other code (data loading, early stopping, scoreboard update, evaluation,
preprocessing) remains identical to the original.**

---

### Parameter comparison (M=3, L_active ≈ 10000)

| Component | Original | Combined | Reduction |
|-----------|----------|----------|-----------|
| `g_raw` | 3 | 3 | — |
| `w_delta` | 3 × L | 3 × L | — |
| `bias` | L | L | — |
| `U` | L_active × 64 | L_active × 16 | 4× |
| `V` | L_active × 64 | **removed** | ∞ |
| `W` | 4 × 64 = 256 | 3 × 16 = 48 | 5.3× |
| `raw_gate` | 1 | **removed** | ∞ |
| **Low-rank total** | **~1,280,257** | **~160,048** | **~8×** |

### Hyperparameter comparison

| Hyperparameter | Original | Combined | Status |
|----------------|----------|----------|--------|
| `BEST_LR` | 0.003 | 0.003 | unchanged |
| `BEST_WEIGHT_DECAY` | 0.0 | 0.0 | unchanged |
| `BEST_BATCH_SIZE` | 256 | 256 | unchanged |
| `LAMBDA_DELTA_L2` | 1e-3 | 1e-3 | unchanged |
| `LOWRANK_LR` | 1e-4 | 1e-4 | unchanged |
| `LOWRANK_WEIGHT_DECAY` | 1e-2 | 1e-2 | unchanged |
| `DEFAULT_RANK` | 64 | **16** | changed |
| `DELTA_CLAMP` | 0.5 | 0.5 | unchanged |
| `DELTA_GATE_MAX` | 0.2 | **removed** | replaced by `FIXED_GATE=0.02` |
| `EPOCHS` | 20 | 20 | unchanged |
| `PATIENCE` | 3 | 3 | unchanged |
| `MIN_EPOCHS` | 2 | 2 | unchanged |

**Total unique hyperparameters: 11** (vs. original's **13**).

### Risks

- **Compounding capacity reduction:** S3 (rank 16) and S8 (symmetric) both reduce
  low-rank capacity. Individually each lost ~0.05%, but combined with S4 (fewer
  channels) the effective capacity drops ~8×. If the regressions are additive,
  the worst-case combined loss is ~0.15% — still within the top 10.
- **Loss of centering without clamping headroom:** S6 removes centering while the
  tanh clamp remains. The clamp bounds the output to [-0.5, 0.5] before the gate,
  so even without centering the effective delta is bounded to [-0.01, 0.01].
  This should remain stable.

---

## Simplification 12: `simplify_combined_v2`

**Name:** `torch_per_label_softmax_global_active_lowrank_combined_v2`

**Hypothesis:** S11 combined five simplifications (S2+S3+S4+S6+S8) and achieved
only 99.88% of champion performance (0.5344 vs 0.5350). Analysis of per-dataset
results reveals the regression was concentrated on yso-en (NDCG@10 dropped from
0.6556 to 0.5527), indicating that the three capacity-reducing changes to the mixer
(S3: rank 64→16, S4: remove base channel, S8: tie U=V) compounded and
over-constrained the cross-label coupling.

This new combination takes a conservative approach: include **only the two
simplifications that don't reduce mixer capacity**:

- **S6 (no_centering):** The only simplification that *improved* performance
  (+0.000273, new #1 at 0.535290). Removing per-sample centering benefits the
  low-rank variant because the low-rank structure is inherently constrained and
  centering was actually removing useful signal.
- **S2 (fixed_gate):** Negligible regression (−0.000216). The learned gate barely
  moves from its initialization of 0.02 during training, so fixing it removes a
  parameter without meaningful loss. Also removes the `DELTA_GATE_MAX` hyperparameter.

**Why not include others:**

| Source | Why excluded |
|--------|--------------|
| S3 (`rank16`) | Reduces mixer capacity — was part of the compounding problem in S11. Rank 64 preserves the richer cross-label correlations that benefit yso-en. |
| S4 (`no_base_ch`) | Reduces mixer input — compounds with rank reduction. Base logits provide useful signal on yso-en. |
| S8 (`symmetric`) | Halves label-projection parameters — compounds with rank/channel reduction. Individually safe, but the S11 experiment showed it contributes to over-constraining when combined. |
| S1, S5, S7, S9, S10 | All showed clear regressions individually. |

**Predicted performance:** S6 gained +0.000273 and S2 lost −0.000216. If
the effects are additive, the expected weighted average is ~0.535074, which would
match or slightly exceed the champion (0.535017). Because both changes affect only
the delta computation (not the mixer architecture), they are structurally
independent and unlikely to interact negatively.

**Changes vs. original (2 changes):**

1. **Remove per-sample centering** from `get_lowrank_delta()` (S6).
2. **Replace learnable gate with a fixed constant `gate = 0.02`.** Remove
   `raw_gate` parameter and `DELTA_GATE_MAX` hyperparameter (S2).

---

### Concrete spec

#### Constants (changes from original highlighted)

```python
EPOCHS = 20                      # unchanged
K_VALUES = (10, 1000)            # unchanged
PATIENCE = 3                     # unchanged
MIN_EPOCHS = 2                   # unchanged
EARLY_STOP_EVAL_ROWS = 512       # unchanged
EARLY_STOP_SEED = 1337           # unchanged
EVAL_BATCH_SIZE = 512            # unchanged
TRAIN_SEED = 0                   # unchanged

BEST_LR = 0.003                  # unchanged
BEST_WEIGHT_DECAY = 0.0          # unchanged
BEST_BATCH_SIZE = 256            # unchanged
LAMBDA_DELTA_L2 = 1e-3           # unchanged

LOWRANK_LR = 1e-4                # unchanged
LOWRANK_WEIGHT_DECAY = 1e-2      # unchanged

DEFAULT_RANK = 64                # unchanged (NOT reduced — keep full capacity)
FIXED_GATE = 0.02                # NEW: replaces learnable raw_gate (S2)

# REMOVED: DELTA_GATE_MAX = 0.2  (S2)
```

#### `LowRankActiveMixer` class — UNCHANGED from original

```python
class LowRankActiveMixer(nn.Module):
    """
    Project feats (B, C, L_active) -> (B, L_active) via low-rank linear mixer.
    Separate U and V matrices (NOT tied — S8 excluded).
    """
    def __init__(self, n_channels: int, n_active: int, rank: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_active = n_active
        self.rank = rank

        self.U = nn.Parameter(0.01 * torch.randn(self.n_active, self.rank))
        self.V = nn.Parameter(0.01 * torch.randn(self.n_active, self.rank))
        self.W = nn.Parameter(0.01 * torch.randn(self.n_channels, self.rank))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        if self.n_active == 0:
            return feats.new_zeros((B, 0))

        # (B*C, L) @ (L, r) -> (B*C, r)
        x2 = feats.reshape(B * self.n_channels, self.n_active)
        Z = x2 @ self.U
        Z = Z.reshape(B, self.n_channels, self.rank)

        # Mix channels
        h = torch.sum(Z * self.W.unsqueeze(0), dim=1)  # (B, r)

        # Project back to labels — uses V.t() (separate V, NOT tied to U)
        delta = h @ self.V.t()  # (B, L_active)
        return delta
```

#### `ActiveLowRankEnsemble` class

**`__init__` — 1 change (S2: remove `raw_gate`):**

```python
def __init__(
    self,
    *,
    n_models: int,
    n_labels: int,
    active_idx: torch.Tensor,
    rank: int,
    init_global: torch.Tensor | None = None,
) -> None:
    super().__init__()
    self.n_models = int(n_models)
    self.n_labels = int(n_labels)

    if active_idx.ndim != 1:
        raise ValueError("active_idx must be 1D")
    self.register_buffer("active_idx", active_idx.long())
    self.n_active = int(active_idx.numel())

    # init_global handling — identical to original
    if init_global is None:
        g = torch.full((self.n_models,), 1.0 / self.n_models, dtype=torch.float32)
    else:
        if init_global.ndim != 1 or init_global.shape[0] != self.n_models:
            raise ValueError(
                f"init_global must have shape ({self.n_models},), got {tuple(init_global.shape)}"
            )
        g = init_global.to(dtype=torch.float32).clone()
        s = float(g.sum().item())
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("init_global must sum to a positive finite value")
        g = g / g.sum()

    self.g_raw = nn.Parameter(torch.log(torch.clamp(g, min=1e-12)))
    self.w_delta = nn.Parameter(torch.zeros(self.n_models, self.n_labels))  # KEPT
    self.bias = nn.Parameter(torch.zeros(self.n_labels))

    # C = M raw logits + 1 base logit (base channel KEPT — S4 excluded)
    self.lowrank = LowRankActiveMixer(
        n_channels=self.n_models + 1,  # unchanged
        n_active=self.n_active,
        rank=rank,                      # uses DEFAULT_RANK=64, unchanged
    )

    # REMOVED: self.raw_gate = nn.Parameter(...) (S2 — replaced by FIXED_GATE constant)
```

**`global_w` and `effective_w` — unchanged:**

```python
def global_w(self) -> torch.Tensor:
    return torch.softmax(self.g_raw, dim=0)

def effective_w(self) -> torch.Tensor:
    return self.global_w()[:, None] + self.w_delta
```

**`get_lowrank_delta` — 2 changes (S2 + S6):**

```python
def get_lowrank_delta(self, x_active: torch.Tensor, base_logits_active: torch.Tensor) -> torch.Tensor:
    # Base channel KEPT (S4 excluded)
    feats = torch.cat([x_active, base_logits_active.unsqueeze(1)], dim=1)  # (B, M+1, L_active)

    delta_active = self.lowrank(feats)

    # Tanh clamp KEPT (S9 excluded)
    DELTA_CLAMP = 0.5
    delta_active = DELTA_CLAMP * torch.tanh(delta_active / DELTA_CLAMP)

    # S6: per-sample centering REMOVED
    # (was: delta_active = delta_active - delta_active.mean(dim=1, keepdim=True))

    # S2: fixed gate replaces learnable gate
    # (was: DELTA_GATE_MAX = 0.2; gate = torch.sigmoid(self.raw_gate) * DELTA_GATE_MAX)
    return delta_active * FIXED_GATE
```

**`forward` — unchanged** (still passes `base_logits_active` to `get_lowrank_delta`):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected input of shape (batch, n_models, n_labels), got {x.shape}")

    w_eff = self.effective_w().unsqueeze(0)  # (1, M, L)
    base_logits = (x * w_eff).sum(dim=1) + self.bias

    if self.n_active == 0:
        return base_logits

    x_active = x.index_select(dim=2, index=self.active_idx)
    base_logits_active = base_logits.index_select(dim=1, index=self.active_idx)

    gated_delta_active = self.get_lowrank_delta(x_active, base_logits_active)

    # Multiplicative application KEPT (S7 excluded)
    out_active = base_logits_active * (1.0 + gated_delta_active)

    out = base_logits.clone()
    out.index_copy_(dim=1, index=self.active_idx, source=out_active)
    return out

def delta_l2(self) -> torch.Tensor:
    return (self.w_delta**2).mean()  # KEPT
```

#### `_delta_stats` helper — unchanged

The function signature and body remain identical to the original. It still
receives `base_logits_active` from the model's `get_lowrank_delta` method.

```python
def _delta_stats(
    model: ActiveLowRankEnsemble, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[float, float]:
    sum_abs = 0.0
    n_abs = 0
    max_samples = 1_000_000
    samples: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                xb = batch[0]
            else:
                xb = batch
            xb = xb.to(device, non_blocking=True)

            x_active = xb.index_select(dim=2, index=model.active_idx)
            w_eff = model.effective_w().unsqueeze(0)
            base_logits = (xb * w_eff).sum(dim=1) + model.bias
            base_logits_active = base_logits.index_select(dim=1, index=model.active_idx)

            delta = model.get_lowrank_delta(x_active, base_logits_active)
            a = delta.abs().detach().cpu().reshape(-1)

            sum_abs += float(a.sum().item())
            n_abs += int(a.numel())

            if max_samples > 0:
                remaining = max_samples - sum(int(s.numel()) for s in samples)
                if remaining <= 0:
                    max_samples = 0
                else:
                    if a.numel() <= remaining:
                        samples.append(a)
                    else:
                        idx = torch.randperm(a.numel())[:remaining]
                        samples.append(a.index_select(0, idx))

    if n_abs == 0:
        return 0.0, 0.0

    mean_abs = sum_abs / float(n_abs)
    if not samples:
        return float(mean_abs), 0.0

    v = torch.cat(samples, dim=0)
    v, _ = torch.sort(v)
    q_idx = min(int(round(0.95 * (v.numel() - 1))), v.numel() - 1)
    p95_abs = float(v[q_idx].item())
    return float(mean_abs), float(p95_abs)
```

#### Optimizer — adjusted (S2: remove `raw_gate` group)

```python
# Two-tiered LR KEPT (S5 single_lr was unsuccessful)
optimizer = optim.AdamW(
    [
        {"params": [model.g_raw, model.w_delta, model.bias], "lr": BEST_LR, "weight_decay": BEST_WEIGHT_DECAY},
        {"params": model.lowrank.parameters(), "lr": LOWRANK_LR, "weight_decay": LOWRANK_WEIGHT_DECAY},
        # REMOVED: {"params": [model.raw_gate], ...} (S2 — gate is now a constant)
    ],
    eps=1e-8,
)
```

#### Training loop — adjusted logging (S2: no `gate_val`)

```python
# In the epoch logging, replace gate_val with the constant:
delta_mean_abs, delta_p95_abs = _delta_stats(model, train_eval_loader, DEVICE)

# REMOVED: gate_val = torch.sigmoid(model.raw_gate).item()

print(
    f"Epoch {epoch:02d} | loss={float(last_loss or 0.0):.6f} | "
    f"train_ndcg@1000(subset)={train_ndcg1000:.6f} train_ndcg@10={train_ndcg10:.6f} | "
    f"test_ndcg@10={test_metrics['ndcg@10']:.6f} "
    f"test_ndcg@1000={test_metrics['ndcg@1000']:.6f} "
    f"test_f1@5={test_metrics['f1@5']:.6f} | "
    f"gate={FIXED_GATE:.4f}(fixed) LowRank_delta_mean={delta_mean_abs:.6f} p95={delta_p95_abs:.6f} | "
    f"total={epoch_dt:.3f}s"
)
```

#### `main()` function — only `model_name` changes

```python
model_name = f"torch_per_label_softmax_global_active_lowrank_combined_v2({','.join(ensemble_keys)})"
```

**All other code (data loading, early stopping, scoreboard update, evaluation,
preprocessing) remains identical to the original.**

---

### Diff summary vs. original

Only **4 lines** in the model code change:

1. **Remove** `self.raw_gate = nn.Parameter(...)` from `__init__`.
2. **Remove** `delta_active = delta_active - delta_active.mean(dim=1, keepdim=True)` from `get_lowrank_delta`.
3. **Replace** the gate computation (`gate = sigmoid(raw_gate) * 0.2`) with `return delta_active * FIXED_GATE`.
4. **Remove** the `raw_gate` optimizer param group.

Plus 2 constant changes:
- **Add** `FIXED_GATE = 0.02`.
- **Remove** `DELTA_GATE_MAX = 0.2` (not used anywhere else).

### Parameter comparison (M=3, L_active ≈ 10000)

| Component | Original | Combined v2 | Reduction |
|-----------|----------|-------------|-----------|
| `g_raw` | 3 | 3 | — |
| `w_delta` | 3 × L | 3 × L | — |
| `bias` | L | L | — |
| `U` | L_active × 64 | L_active × 64 | — |
| `V` | L_active × 64 | L_active × 64 | — |
| `W` | 4 × 64 = 256 | 4 × 64 = 256 | — |
| `raw_gate` | 1 | **removed** | ∞ |
| **Total** | N + 1 | **N** | 1 param |

### Hyperparameter comparison

| Hyperparameter | Original | Combined v2 | Status |
|----------------|----------|-------------|--------|
| `BEST_LR` | 0.003 | 0.003 | unchanged |
| `BEST_WEIGHT_DECAY` | 0.0 | 0.0 | unchanged |
| `BEST_BATCH_SIZE` | 256 | 256 | unchanged |
| `LAMBDA_DELTA_L2` | 1e-3 | 1e-3 | unchanged |
| `LOWRANK_LR` | 1e-4 | 1e-4 | unchanged |
| `LOWRANK_WEIGHT_DECAY` | 1e-2 | 1e-2 | unchanged |
| `DEFAULT_RANK` | 64 | 64 | unchanged |
| `DELTA_CLAMP` | 0.5 | 0.5 | unchanged |
| `DELTA_GATE_MAX` | 0.2 | **removed** | replaced by `FIXED_GATE=0.02` |
| `EPOCHS` | 20 | 20 | unchanged |
| `PATIENCE` | 3 | 3 | unchanged |
| `MIN_EPOCHS` | 2 | 2 | unchanged |

**Total unique hyperparameters: 12** (vs. original's **13**).

### Risks

- **Minimal risk:** Both S6 and S2 were individually validated with negligible
  or positive impact. Neither change reduces mixer capacity.
- **Independence:** S6 modifies the post-processing of the mixer output (centering),
  while S2 modifies the gating mechanism. These are structurally independent
  operations in the forward pass pipeline: `lowrank → tanh_clamp → [centering] → [gate]`.
  Removing centering and fixing the gate affect different stages and should not
  interact negatively.
- **Bounded delta:** Even without centering, the tanh clamp ([-0.5, 0.5]) combined
  with the fixed gate (0.02) bounds the effective gated delta to [-0.01, 0.01].
  This ensures stability.
