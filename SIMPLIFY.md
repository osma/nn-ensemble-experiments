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
