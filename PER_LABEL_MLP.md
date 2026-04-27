# Experiments: `torch_per_label_mlp` Variants

This document tracks small, controlled variations of `torch_per_label_mlp` for benchmarking.
Each variant changes **one aspect** of the model (usually stage 2) while keeping:

- The same data loading, preprocessing, and metrics (NDCG@k, F1@k)
- The same early-stopping policy (train subset NDCG@1000; patience/min-epochs unchanged)
- Roughly the same model size (no dramatic increases in parameters)

The overall goal is to find a `torch_per_label_mlp_*` variant that improves the **overall scoreboard**
criterion: **average of all three test metrics**, averaged across the **three datasets**.

---

## Baseline: `torch_per_label_mlp`

Baseline model intent (as implemented in `benchmarks/torch_per_label_mlp.py`):

Two-stage model:

- **Stage 1**: `torch_per_label` (per-label linear ensemble in logit space) trained with its usual
  early stopping. Produces base logits `base_logits[b, l]`.
- **Stage 2**: Train a residual MLP on *active labels only* using features:
  - Channels 0..2: log1p-preprocessed base predictors `x[b, m, l]`
  - Channel 3: stage-1 logits `base_logits[b, l]`

Stage-2 produces a per-label adjustment signal `delta_active[b, l_active]` which is:
- tanh-bounded (`DELTA_CLAMP`)
- centered per sample over active labels (mean subtraction)
- scaled by a bounded learnable scalar gate `gate ∈ (0, DELTA_GATE_MAX)`

Baseline application form (for active labels):
- `out_active = base_active * (1 + gate * delta_active)`

(Non-active labels remain exactly at the stage-1 logits.)

---

## Variant 1: `torch_per_label_mlp_additive_delta`

**What changes:** apply the residual as an **additive** logit-space delta instead of multiplicative reweighting.

- Baseline (multiplicative):
  - `out_active = base_active * (1 + gate * delta_active)`
- Variant (additive):
  - `out_active = base_active + gate * delta_active`

**Hypothesis:** multiplicative updates can be unstable when `base_active` has large magnitude and can be
ineffective when `base_active ≈ 0`. An additive residual is the standard residual form and may yield
more predictable optimization and better generalization.

### Implementation notes
- Implemented in: `benchmarks/torch_per_label_mlp_additive_delta.py`
- Keep the same `delta_active` computation (tanh bound + centering) and the same scalar gate.
- Only change the final combination rule for active labels.

### Results
_TBD_

### Analysis
_TBD_

---

## Variant 2: `torch_per_label_mlp_gate_per_label`

**What changes:** replace the **single scalar gate** with a **per-active-label gate vector**.

- Baseline:
  - `gate = gate_max * sigmoid(raw_gate)`  (scalar)
- Variant:
  - `gate[l] = gate_max * sigmoid(raw_gate[l])`  for `l` in active labels

Apply elementwise, e.g.:
- multiplicative form: `out_active = base_active * (1 + gate * delta_active)`
- additive form (if combined in a later experiment): `out_active = base_active + gate * delta_active`

(For controlled comparison in this variant, keep the baseline multiplicative form.)

**Hypothesis:** only a subset of labels benefit from cross-label adjustments; for others, stage 2
should remain effectively off. A per-label gate allows selective capacity while keeping changes
small and bounded.

### Implementation notes
- Adds `n_active` parameters (small compared to MLP weights).
- Keep the same gate bounds (`DELTA_GATE_MAX`) and initialization policy, but applied per label.

### Results
_TBD_

### Analysis
_TBD_

---

## Variant 3: `torch_per_label_mlp_gate_per_sample`

**What changes:** make the gate **data-dependent** (per sample) instead of a single learned scalar.

Example concept:

- Compute a pooled summary from `feats[b, c, l_active]`, such as:
  - `p[b, c] = mean_l feats[b, c, l]`
- Feed through a tiny gating head (e.g. linear layer) to produce:
  - `gate[b] = gate_max * sigmoid(gate_head(p[b]))`

Then apply:
- `out_active = base_active * (1 + gate[b] * delta_active)`

**Hypothesis:** cross-label “rules” may only be reliable for certain samples/documents. A per-sample
gate can reduce harm by turning the residual down when signals are weak or noisy, protecting stage-1
performance.

### Implementation notes
- Keep the gating head very small (e.g. a single `nn.Linear`), so model size does not change dramatically.
- Keep the same output bound via `sigmoid` and `DELTA_GATE_MAX`.

### Results
_TBD_

### Analysis
_TBD_

---

## Variant 4: `torch_per_label_mlp_layernorm_feats`

**What changes:** add **normalization** to stage-2 features before the MLP.

Options (choose one in implementation to keep it controlled):
- `LayerNorm` over the flattened feature vector per sample
- Or `LayerNorm` over the channel dimension after pooling
- Or normalize per channel across the active labels

**Hypothesis:** stage-2 inputs mix log1p base scores and stage-1 logits, which can have different
scales and distributions. Normalizing features can stabilize optimization, reduce sensitivity to
learning rate, and mitigate “early peak then degrade” behavior.

### Implementation notes
- Keep normalization lightweight (LayerNorm has a small parameter count).
- Do not change the residual head architecture otherwise.

### Results
_TBD_

### Analysis
_TBD_

---

## Variant 5: `torch_per_label_mlp_rank_bottleneck`

**What changes:** introduce a **low-rank bottleneck** in the stage-2 output projection to regularize
cross-label adjustment structure.

Replace:
- `fc2: hidden_dim -> n_active`

with:
- `fc2a: hidden_dim -> r`
- `fc2b: r -> n_active`

where `r` is small (e.g. 32).

**Hypothesis:** the residual adjustment across labels may lie in a low-dimensional subspace. A rank
bottleneck can reduce overfitting, encourage shared structure, and improve generalization while
keeping the model size in the same ballpark.

### Implementation notes
- Choose `r` so that the parameter count stays comparable to baseline.
- Keep initialization such that the final effective projection starts near zero (near-no-op).

### Results
_TBD_

### Analysis
_TBD_

---

## Variant 6: `torch_per_label_mlp_remove_centering`

**What changes:** remove the per-sample centering constraint on `delta_active`.

- Baseline:
  - `delta_active = delta_active - mean(delta_active over active labels)`
- Variant:
  - no centering

**Hypothesis:** centering forces the residual to be “redistributive” among active labels only. This
may be too restrictive: in some cases, a uniform shift (or broad boost/suppression) of active labels
could help separate active vs inactive labels. Removing centering tests whether the constraint is
helpful regularization or an unnecessary limitation.

### Implementation notes
- Keep tanh bounding and gating identical.
- Only remove the mean-subtraction step.

### Results
_TBD_

### Analysis
_TBD_
