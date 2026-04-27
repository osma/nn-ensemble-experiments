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
- `./regenerate_scoreboard.sh --models torch_per_label_softmax_global`

(Results are copied from `SCOREBOARD.md` and the benchmark console output.)

| Variant | Dataset | Best epoch | Test NDCG@10 | Test NDCG@1000 | Test F1@5 | Notes |
|--------|---------|------------|--------------|----------------|-----------|-------|
| (baseline) torch_per_label |  |  |  |  |  | See `SCOREBOARD.md` |
| torch_per_label_global_plus_delta | yso-fi | 2 | 0.692057 | 0.803521 | 0.524475 | Selected by train subset NDCG@1000 |
| torch_per_label_global_plus_delta | yso-en | 3 | 0.635216 | 0.758246 | 0.449534 | Selected by train subset NDCG@1000 |
| torch_per_label_global_plus_delta | koko | 1 | 0.356868 | 0.465507 | 0.260260 | Selected by train subset NDCG@1000 |
| torch_per_label_softmax_global | yso-fi | 4 | 0.704736 | 0.813194 | 0.540867 | Selected by train subset NDCG@1000 |
| torch_per_label_softmax_global | yso-en | 14 | 0.661455 | 0.776412 | 0.474309 | Selected by train subset NDCG@1000 |
| torch_per_label_softmax_global | koko | 3 | 0.363673 | 0.477691 | 0.266434 | Selected by train subset NDCG@1000 |
| torch_per_label_global_times_scale | yso-fi | 14 | 0.682260 | 0.796095 | 0.517386 | Selected by train subset NDCG@1000 |
| torch_per_label_global_times_scale | yso-en | 20 | 0.643669 | 0.765229 | 0.455654 | Selected by train subset NDCG@1000 |
| torch_per_label_global_times_scale | koko | 4 | 0.359675 | 0.476947 | 0.264296 | Selected by train subset NDCG@1000 |
| torch_per_label_bias_global_plus_delta |  |  |  |  |  |  |

---

## Analysis: `torch_per_label_global_times_scale`

**Overall:** `torch_per_label_global_times_scale` is competitive but does **not** beat the baseline `torch_per_label` or the stronger `torch_per_label_softmax_global` in this scoreboard snapshot. It is closest on `koko`, where it slightly improves NDCG@1000 but is essentially flat/slightly worse on NDCG@10 and F1@5.

**Relative to baseline (`torch_per_label`)**
- **yso-fi:** worse across all three test metrics.
  - baseline: NDCG@10=0.710171, NDCG@1000=0.816454, F1@5=0.544132
  - global×scale: NDCG@10=0.682260, NDCG@1000=0.796095, F1@5=0.517386
  - deltas: -0.0279 NDCG@10, -0.0204 NDCG@1000, -0.0267 F1@5
- **yso-en:** worse across all three test metrics.
  - baseline: NDCG@10=0.659227, NDCG@1000=0.771079, F1@5=0.473627
  - global×scale: NDCG@10=0.643669, NDCG@1000=0.765229, F1@5=0.455654
  - deltas: -0.0156 NDCG@10, -0.0059 NDCG@1000, -0.0180 F1@5
- **koko:** mixed; slightly lower NDCG@10/F1@5 but higher NDCG@1000.
  - baseline: NDCG@10=0.361643, NDCG@1000=0.473905, F1@5=0.264727
  - global×scale: NDCG@10=0.359675, NDCG@1000=0.476947, F1@5=0.264296
  - deltas: -0.0020 NDCG@10, +0.0030 NDCG@1000, -0.0004 F1@5

**Relative to `torch_per_label_softmax_global`**
- **yso-fi:** worse across all three test metrics.
  - softmax_global: NDCG@10=0.704736, NDCG@1000=0.813194, F1@5=0.540867
  - global×scale:  NDCG@10=0.682260, NDCG@1000=0.796095, F1@5=0.517386
- **yso-en:** worse across all three test metrics.
  - softmax_global: NDCG@10=0.661455, NDCG@1000=0.776412, F1@5=0.474309
  - global×scale:  NDCG@10=0.643669, NDCG@1000=0.765229, F1@5=0.455654
- **koko:** slightly worse overall (very close).
  - softmax_global: NDCG@10=0.363673, NDCG@1000=0.477691, F1@5=0.266434
  - global×scale:  NDCG@10=0.359675, NDCG@1000=0.476947, F1@5=0.264296

**Training dynamics / stability (from console output)**
- **yso-fi:** improved up to epoch 14 (selected), then plateau/slight drift; early stopping triggered at epoch 16 (patience=2).
- **yso-en:** best epoch was the final epoch (20); train-subset NDCG@1000 improved steadily, so early stopping did not trigger.
- **koko:** peaked early at epoch 4 and then degraded, consistent with faster overfitting on `koko`.

**Interpretation**
- The multiplicative form (`w_eff = g * scale`, with `g = softmax(g_raw)` and bounded `scale ∈ (0,2)`) stabilizes optimization but may be **too restrictive**: it cannot fully change the “mixture direction” per label the way additive deltas can, only rescale each model’s contribution around the global simplex weights.
- The softmax constraint likely helps stability, but removing the ability to use negative global weights and limiting per-label deviation amplitude can reduce peak ranking performance.

**Next steps**
- Add a learned global scalar `s` so `w_eff = s * g * scale` (restores overall scale freedom while keeping a stable simplex mixture).
- Consider widening the scale family: `scale = exp(scale_raw)` with clamping (e.g. `[0.25, 4]`) to allow stronger per-label deviations while staying positive.
- Add diagnostics similar to `torch_per_label` to confirm whether `scale` collapses near 1 or saturates at bounds, and whether `g` stays close to dataset init weights.

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

---

## Analysis: `torch_per_label_softmax_global`

**Overall:** This variant **improves substantially over** `torch_per_label_global_plus_delta` and is **very competitive** with the baseline `torch_per_label`. It slightly beats the baseline on yso-en (all three metrics) and is close-but-not-better on yso-fi; it improves on koko.

**Relative to baseline (`torch_per_label`)**
- **yso-fi:** slightly worse overall.
  - baseline: NDCG@10=0.710171, NDCG@1000=0.816454, F1@5=0.544132
  - softmax_global: NDCG@10=0.704736, NDCG@1000=0.813194, F1@5=0.540867
  - deltas: -0.0054 NDCG@10, -0.0033 NDCG@1000, -0.0033 F1@5
- **yso-en:** **better** across all three test metrics.
  - baseline: NDCG@10=0.659227, NDCG@1000=0.771079, F1@5=0.473627
  - softmax_global: NDCG@10=0.661455, NDCG@1000=0.776412, F1@5=0.474309
  - deltas: +0.0022 NDCG@10, +0.0053 NDCG@1000, +0.0007 F1@5
- **koko:** **better** across all three test metrics.
  - baseline: NDCG@10=0.361643, NDCG@1000=0.473905, F1@5=0.264727
  - softmax_global: NDCG@10=0.363673, NDCG@1000=0.477691, F1@5=0.266434
  - deltas: +0.0020 NDCG@10, +0.0038 NDCG@1000, +0.0017 F1@5

**Relative to `torch_per_label_global_plus_delta`**
- **yso-fi:** big improvement.
  - global+delta: NDCG@10=0.692057, NDCG@1000=0.803521, F1@5=0.524475
  - softmax_global: NDCG@10=0.704736, NDCG@1000=0.813194, F1@5=0.540867
  - deltas: +0.0127 NDCG@10, +0.0097 NDCG@1000, +0.0164 F1@5
- **yso-en:** big improvement.
  - global+delta: NDCG@10=0.635216, NDCG@1000=0.758246, F1@5=0.449534
  - softmax_global: NDCG@10=0.661455, NDCG@1000=0.776412, F1@5=0.474309
  - deltas: +0.0262 NDCG@10, +0.0182 NDCG@1000, +0.0248 F1@5
- **koko:** slight improvement.
  - global+delta: NDCG@10=0.356868, NDCG@1000=0.465507, F1@5=0.260260
  - softmax_global: NDCG@10=0.363673, NDCG@1000=0.477691, F1@5=0.266434
  - deltas: +0.0068 NDCG@10, +0.0122 NDCG@1000, +0.0062 F1@5

**Training dynamics / stability**
- **yso-fi:** peaked at epoch 4, then degraded (epoch 6 was clearly worse). Early stopping selected epoch 4 correctly.
- **yso-en:** improved steadily through epoch 14 and then plateaued/slightly drifted; early stopping selected epoch 14.
- **koko:** peaked at epoch 3 and then degraded; early stopping selected epoch 3.

**Interpretation**
- Constraining the shared/global weights to a convex mixture (softmax) + keeping an explicit L2 penalty on `w_delta` seems to help prevent the “drift away from a good initialization” failure mode seen in the unconstrained global+delta setup.
- Gains are largest on yso-en, suggesting that the dataset benefits from a stable global mixture with limited per-label deviations.

**Next steps**
- Consider adding a learned global scale `s` so `w_eff = s * g + w_delta` (keeps convex mixture shape but allows overall scale).
- Try a small grid for `LAMBDA_DELTA_L2` (e.g. {3e-4, 1e-3, 3e-3, 1e-2}) to see if yso-fi can match baseline while preserving yso-en gains.
