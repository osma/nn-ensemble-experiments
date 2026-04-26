# torch_per_label variants

This document describes small architecture variations of the `torch_per_label` benchmark model.
The goal is to isolate which modeling choices matter for ranking metrics (NDCG@k, F1@k) while
keeping the model class **linear** and roughly the same size.

Baseline reference: `torch_per_label` (see `benchmarks/torch_per_label.py`)

- Inputs: `x[b, m, l]` are log1p-preprocessed base scores (non-negative).
- Output: `logits[b, l]` (used with `BCEWithLogitsLoss`; ranking metrics depend only on ordering).

All variants below keep:
- **Global model-specific weights** (per source/base model `m`)
- **Label-specific weights** (varying by label `l`)
- A bias term (either per-label or reparameterized)

Notation:
- `M` = number of base models
- `L` = number of labels
- `b` = batch index

---

## Variant: `torch_per_label_global_plus_delta`

**What changes:** Reparameterize weights as a sum of a global per-model vector plus a per-label residual.

**Parameters**
- `w_global[m]` (shape `(M,)`)
- `w_delta[m, l]` (shape `(M, L)`)
- `bias[l]` (shape `(L,)`)

**Effective weights**
- `w_eff[m, l] = w_global[m] + w_delta[m, l]`

**Forward**
- `logits[b, l] = Σ_m w_eff[m, l] * x[b, m, l] + bias[l]`

**Why test it**
- Encourages shared structure across labels via `w_global`, while still allowing per-label adaptation via `w_delta`.
- Tests whether the baseline’s fully-free per-label weights overfit or behave poorly on tail labels.

---

## Variant: `torch_per_label_softmax_global`

**What changes:** Constrain the global per-model weights to be a convex mixture (positive and sum to 1)
using a softmax, while keeping per-label deviations.

**Parameters**
- `g_raw[m]` (shape `(M,)`), with `g = softmax(g_raw)`
- `w_delta[m, l]` (shape `(M, L)`)
- `bias[l]` (shape `(L,)`)

**Effective weights**
- `w_eff[m, l] = g[m] + w_delta[m, l]`

**Forward**
- `logits[b, l] = Σ_m w_eff[m, l] * x[b, m, l] + bias[l]`

**Why test it**
- Isolates whether unconstrained global weights drifting negative or exploding harms ranking/generalization.
- Keeps most of the flexibility in `w_delta` but stabilizes the shared baseline mixture.

Notes:
- Optionally, to preserve scale freedom: `w_eff = s * g + w_delta` with a learned scalar `s`.

---

## Variant: `torch_per_label_global_times_scale`

**What changes:** Use multiplicative label-specific scaling around the global weights, instead of additive residuals.

**Parameters**
- `w_global[m]` (shape `(M,)`)
- `scale_raw[m, l]` (shape `(M, L)`), with a bounded transform such as:
  - `scale[m, l] = 1 + tanh(scale_raw[m, l])`
- `bias[l]` (shape `(L,)`)

**Effective weights**
- `w_eff[m, l] = w_global[m] * scale[m, l]`

**Forward**
- `logits[b, l] = Σ_m w_eff[m, l] * x[b, m, l] + bias[l]`

**Why test it**
- Additive deltas can fully override the global mixture per label; multiplicative scaling tends to preserve the
  “shape” of the global ensemble and may generalize better.
- Still linear in inputs; parameter count is essentially unchanged.

---

## Variant: `torch_per_label_bias_global_plus_delta`

**What changes:** Reparameterize the bias into a global intercept plus a per-label residual.

**Parameters**
- `weights[m, l]` (shape `(M, L)`) (same as baseline weights)
- `bias_global` (scalar)
- `bias_delta[l]` (shape `(L,)`)

**Effective bias**
- `bias[l] = bias_global + bias_delta[l]`

**Forward**
- `logits[b, l] = Σ_m weights[m, l] * x[b, m, l] + bias_global + bias_delta[l]`

**Why test it**
- Tests whether a dataset-wide offset is better learned once, reducing pressure on per-label biases
  (especially for rare labels).
- Keeps the model linear and almost identical in capacity.

---

## Results log (to be filled in)

Add rows here once each variant has been benchmarked.

| Variant | Dataset | Best epoch | Test NDCG@10 | Test NDCG@1000 | Test F1@5 | Notes |
|--------|---------|------------|--------------|----------------|-----------|-------|
| (baseline) torch_per_label |  |  |  |  |  |  |
| torch_per_label_global_plus_delta |  |  |  |  |  | Implemented in `benchmarks/torch_per_label_global_plus_delta.py` |
| torch_per_label_softmax_global |  |  |  |  |  |  |
| torch_per_label_global_times_scale |  |  |  |  |  |  |
| torch_per_label_bias_global_plus_delta |  |  |  |  |  |  |
