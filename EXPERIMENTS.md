# Experiments: `torch_mean_residual` Variants

This document tracks small, controlled variations of `torch_mean_residual` for benchmarking.
Each variant changes **one aspect** of the model while keeping the core structure:

- **Global per-model weights** (shared across all labels)
- **Label-specific per-model weights** (per-label deviations / adjustments)
- A **label bias** term in some form

The goal is to isolate which modeling choices improve ranking metrics (NDCG@k, F1@k)
under the existing training/evaluation protocol.

---

## Baseline: `torch_mean_residual`

Current reference formulation (conceptual):

- Parameters:
  - `global_w[m]` (learnable, per model)
  - `delta_w[m, l]` (learnable, per model × label; initialized to 0)
  - `bias[l]` (learnable, per label)
- Effective weights:
  - `w_eff[m, l] = global_w[m] + delta_w[m, l]`
- Logits:
  - `logits[b, l] = sum_m w_eff[m, l] * x[b, m, l] + bias[l]`
- Regularization:
  - L2 penalty on `delta_w` (shrink residuals toward 0)
  - L2 penalty on `bias`

Baseline reference (from `SCOREBOARD.md`, test metrics):
- `yso-fi`: NDCG@10=0.687398, NDCG@1000=0.799336, F1@5=0.521631
- `yso-en`: NDCG@10=0.634044, NDCG@1000=0.757152, F1@5=0.447385
- `koko`:   NDCG@10=0.357736, NDCG@1000=0.467187, F1@5=0.261571

---

## Variant 1: `torch_mean_residual_softmax_global`

**What changes:** constrain global weights to a convex combination via softmax.

- Replace global weights parameterization with:
  - learn `global_logits[m]` (unconstrained)
  - compute `global_w = softmax(global_logits)` (non-negative, sums to 1)
- Keep residual weights additive:
  - `w_eff[m, l] = global_w[m] + delta_w[m, l]`

**Hypothesis:** stabilizes optimization and discourages degenerate negative/large global weights;
tests whether unconstrained global scaling is important.

### Results (2026-03-16)

Command:
- `./regenerate_scoreboard.sh --models torch_mean_residual_softmax_global`

Best epoch is selected by **train subset NDCG@1000** early-stopping (same as baseline).

Test metrics (best epoch per dataset from run output):
- `yso-fi` (best epoch=5):  NDCG@10=0.696986, NDCG@1000=0.804392, F1@5=0.536324
- `yso-en` (best epoch=8):  NDCG@10=0.648991, NDCG@1000=0.766754, F1@5=0.461473
- `koko`   (best epoch=4):  NDCG@10=0.361896, NDCG@1000=0.477831, F1@5=0.266607

Delta vs baseline `torch_mean_residual` (test metrics):
- `yso-fi`: +0.009588 NDCG@10, +0.005056 NDCG@1000, +0.014693 F1@5
- `yso-en`: +0.014947 NDCG@10, +0.009602 NDCG@1000, +0.014088 F1@5
- `koko`:   +0.004160 NDCG@10, +0.010644 NDCG@1000, +0.005036 F1@5

### Analysis

- This variant is a **consistent improvement across all three datasets**, and improves
  **all three reported test metrics** on each dataset in this run.
- The biggest gains are on `yso-en` (notably NDCG@10 and F1@5), suggesting that
  constraining global weights to a convex combination may reduce harmful global
  weight drift and let `delta_w` and `bias` do the fine-grained per-label work.
- `koko` gains are smaller in absolute terms but still positive, with the largest
  lift on NDCG@1000 (+0.0106). This suggests the effect is not purely a top-10
  precision bump; it also improves deeper ranking.

Notes / follow-ups:
- The run used the default regularization (`lambda_delta=1e-2`, `lambda_bias=1e-3`).
  Since softmax removes the possibility of negative global weights, it may be worth
  re-sweeping `lambda_delta` slightly downward (e.g. 3e-3, 1e-3) to see if residuals
  can be allowed to carry more signal without overfitting.
- Consider adding debug prints of `global_w()` at the best epoch to verify whether
  the learned convex weights remain close to dataset-provided initial weights or
  meaningfully shift.

---

## Variant 2: `torch_mean_residual_globalxdelta`

**What changes:** make label-specific deviations multiplicative around the global weights.

- Effective weights become:
  - `w_eff[m, l] = global_w[m] * (1 + delta_w[m, l])`
- Initialize `delta_w = 0` so the starting behavior matches the baseline.

**Hypothesis:** per-label adjustments should be relative to the global strength of each base model.
May be more stable than additive residuals when base score scales differ.

---

## Variant 3: `torch_mean_residual_delta_tanh_clamp`

**What changes:** bound the per-label residual weights using a smooth clamp.

- Learn `delta_raw[m, l]` but compute:
  - `delta_w[m, l] = delta_max * tanh(delta_raw[m, l])`
- Effective weights remain additive:
  - `w_eff[m, l] = global_w[m] + delta_w[m, l]`

**Hypothesis:** prevents extreme per-label weights for a subset of labels; tests whether occasional
large residuals help ranking or primarily cause overfitting / instability.

---

## Variant 4: `torch_mean_residual_bias_per_model`

**What changes:** add a tiny per-model global bias term, in addition to per-label bias.

- Add parameter:
  - `bias_model[m]` (learnable, size `M`)
- Example formulation:
  - `logits[b, l] = sum_m w_eff[m, l] * (x[b, m, l] + bias_model[m]) + bias_label[l]`

(Equivalent algebraic forms are acceptable; the intent is a per-model offset applied globally.)

**Hypothesis:** compensates for systematic per-model offsets after preprocessing (e.g. log1p);
minimal parameter increase (just 3 scalars in the 3-model case).

---

## Variant 5: `torch_mean_residual_bias_residual`

**What changes:** split label bias into a global scalar + per-label residual (mirrors the weight design).

- Bias becomes:
  - `bias[l] = bias_global + bias_delta[l]`
- Regularize `bias_delta` strongly (shrink toward 0); optionally do not regularize `bias_global`
(or do so very weakly).

**Hypothesis:** reduces per-label bias overfitting while retaining a dataset-wide intercept; tests whether
the per-label bias term is doing “too much work”.

---

## Variant 6: `torch_mean_residual_l2_anchor_global`

**What changes:** add an explicit penalty anchoring `global_w` to the dataset-provided init weights.

- Let `w0[m]` be `DatasetConfig.ensemble3_init_weights` (or uniform if unavailable).
- Add regularization term:
  - `lambda_global * mean((global_w - w0)**2)`

**Hypothesis:** clarifies how much improvement comes from moving global weights vs. relying on per-label residuals;
may improve cross-dataset stability and reduce overfitting.

---

## Notes for future result logging

For each variant, we will eventually record:
- Command used (dataset, lambda settings, any new hyperparameters like `delta_max`)
- Best epoch selected by early stopping criterion
- Train/Test metrics written to `SCOREBOARD.md`
- Short qualitative notes (stability, runtime, failure modes)

A simple template per run:

- Variant:
- Dataset:
- Best epoch:
- Train: NDCG@10 / NDCG@1000
- Test:  NDCG@10 / NDCG@1000 / F1@5
- Notes:
