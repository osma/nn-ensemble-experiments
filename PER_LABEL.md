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

## Results log

Below are results from running:

- `./regenerate_scoreboard.sh --models torch_per_label_global_plus_delta`

(Results are copied from `SCOREBOARD.md` and the benchmark console output.)

| Variant | Dataset | Best epoch | Test NDCG@10 | Test NDCG@1000 | Test F1@5 | Notes |
|--------|---------|------------|--------------|----------------|-----------|-------|
| (baseline) torch_per_label |  |  |  |  |  | See `SCOREBOARD.md` |
| torch_per_label_global_plus_delta | yso-fi | 2 | 0.692057 | 0.803521 | 0.524475 | Selected by train subset NDCG@1000 |
| torch_per_label_global_plus_delta | yso-en | 3 | 0.635216 | 0.758246 | 0.449534 | Selected by train subset NDCG@1000 |
| torch_per_label_global_plus_delta | koko | 1 | 0.356868 | 0.465507 | 0.260260 | Selected by train subset NDCG@1000 |
| torch_per_label_softmax_global |  |  |  |  |  |  |
| torch_per_label_global_times_scale |  |  |  |  |  |  |
| torch_per_label_bias_global_plus_delta |  |  |  |  |  |  |

---

## Analysis: `torch_per_label_global_plus_delta`

**Overall:** This variant is clearly competitive but does **not** beat the baseline `torch_per_label` on any dataset in the current scoreboard snapshot.

**Relative to baseline (`torch_per_label`)**
- **yso-fi:** worse across all three test metrics.
  - baseline: NDCG@10=0.710171, NDCG@1000=0.816454, F1@5=0.544132
  - global+delta: NDCG@10=0.692057, NDCG@1000=0.803521, F1@5=0.524475
  - deltas: -0.0181 NDCG@10, -0.0129 NDCG@1000, -0.0197 F1@5
- **yso-en:** worse across all three test metrics.
  - baseline: NDCG@10=0.659227, NDCG@1000=0.771079, F1@5=0.473627
  - global+delta: NDCG@10=0.635216, NDCG@1000=0.758246, F1@5=0.449534
  - deltas: -0.0240 NDCG@10, -0.0128 NDCG@1000, -0.0241 F1@5
- **koko:** slightly worse across all three test metrics.
  - baseline: NDCG@10=0.361643, NDCG@1000=0.473905, F1@5=0.264727
  - global+delta: NDCG@10=0.356868, NDCG@1000=0.465507, F1@5=0.260260
  - deltas: -0.0048 NDCG@10, -0.0084 NDCG@1000, -0.0045 F1@5

**Training dynamics / stability**
- On **yso-fi** and **koko**, the train-subset NDCG@1000 peaked early and then degraded noticeably by later epochs.
  This suggests the model can overfit or drift away from the good initialization when trained with the same unconstrained
  parameterization and no explicit regularization on `w_delta`.
- On **yso-en**, test metrics continued to improve slightly through epoch 5, but early stopping still selected epoch 3
  based on the train-subset criterion.

**Interpretation**
- The reparameterization alone (without an explicit penalty/constraint on `w_delta`) does not appear to be enough to
  improve generalization vs the fully-free per-label weights baseline.
- Given the observed “early peak then degrade” pattern, the most direct next experiment is to **add an explicit L2
  penalty on `w_delta`** (or constrain `w_global` via softmax) to force the model to stay close to the global mixture
  unless per-label evidence strongly supports deviating.

**Next steps**
- Implement `torch_per_label_softmax_global` to stabilize the shared mixture (prevent negative / exploding global weights).
- Alternatively/additionally, introduce `lambda_delta * ||w_delta||_2^2` (or an anchor penalty to dataset init weights)
  and tune `lambda_delta`.
