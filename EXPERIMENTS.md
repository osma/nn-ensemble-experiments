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

### Implementation

- Script: `benchmarks/torch_mean_residual_globalxdelta.py`
- Model name written to scoreboard: `torch_mean_residual_globalxdelta(...)`
- CLI: identical to `torch_mean_residual` (no new flags)

### Results (2026-03-16)

Command:
- `./regenerate_scoreboard.sh --models torch_mean_residual_globalxdelta`

Best epoch is selected by **train subset NDCG@1000** early-stopping (same as baseline).

Test metrics (best epoch per dataset from run output):
- `yso-fi` (best epoch=1): NDCG@10=0.676377, NDCG@1000=0.792537, F1@5=0.508068
- `yso-en` (best epoch=1): NDCG@10=0.624606, NDCG@1000=0.751390, F1@5=0.441895
- `koko`   (best epoch=1): NDCG@10=0.356553, NDCG@1000=0.469326, F1@5=0.260765

Delta vs baseline `torch_mean_residual` (test metrics):
- `yso-fi`: −0.011021 NDCG@10, −0.006799 NDCG@1000, −0.013563 F1@5
- `yso-en`: −0.009438 NDCG@10, −0.005762 NDCG@1000, −0.005490 F1@5
- `koko`:   −0.001183 NDCG@10, +0.002139 NDCG@1000, −0.000806 F1@5

### Analysis

- This variant is **mostly worse than the additive baseline** on the two YSO datasets, across
  all three metrics. It is close to neutral on `koko`, with a small gain on NDCG@1000 but
  slight losses on NDCG@10 and F1@5.
- Early stopping consistently picked **epoch 1** for all datasets, and train-subset NDCG@1000
  tended to **decrease after the first epoch**. This suggests the multiplicative parameterization
  may be more sensitive to optimization (even though the residuals start at zero), or that the
  model quickly overfits/shifts away from a good initial regime.
- A plausible failure mode is that multiplicative residuals effectively couple the scale of per-label
  adjustments to the learned `global_w[m]`. When `global_w[m]` is small, the model has limited
  ability to “rescue” that source on specific labels; when `global_w[m]` is large, per-label tweaks
  can become too influential unless `delta_w` is very tightly regularized.

Notes / follow-ups:
- Try a **smaller learning rate** (e.g. `LR=0.001`) for this variant; multiplicative residuals
  can make gradients more scale-sensitive.
- Consider regularizing `global_w` (either via softmax parameterization like Variant 1, or an anchor-to-init penalty as in Variant 6) to reduce drift.
- If keeping this variant, it may be best framed as “not beneficial under current training defaults”
  rather than as a new default candidate.

---

## Variant 3: `torch_mean_residual_delta_tanh_clamp`

**What changes:** bound the per-label residual weights using a smooth clamp.

- Learn `delta_raw[m, l]` but compute:
  - `delta_w[m, l] = delta_max * tanh(delta_raw[m, l])`
- Effective weights remain additive:
  - `w_eff[m, l] = global_w[m] + delta_w[m, l]`

**Hypothesis:** prevents extreme per-label weights for a subset of labels; tests whether occasional
large residuals help ranking or primarily cause overfitting / instability.

### Implementation

- Script: `benchmarks/torch_mean_residual_delta_tanh_clamp.py`
- Model name written to scoreboard: `torch_mean_residual_delta_tanh_clamp(...)`
- CLI: identical to `torch_mean_residual` (no new flags)

Notes:
- `delta_max` is implemented as a module-level constant to keep comparisons controlled (no CLI flag).
- Regularization is applied to `delta_raw` (not `delta_w`) for smoother optimization when tanh saturates.

### Results (2026-03-16)

Command:
- `./regenerate_scoreboard.sh --models torch_mean_residual_delta_tanh_clamp`

Best epoch is selected by **train subset NDCG@1000** early-stopping (same as baseline).

Observed behavior:
- Early stopping picked **epoch 1 for all datasets**.
- Train-subset NDCG@1000 tended to **decrease after epoch 1**, suggesting the model quickly moves away
  from a good initial regime under the default LR/regularization.

Test metrics (best epoch per dataset from run output):
- `yso-fi` (best epoch=1): NDCG@10=0.676865, NDCG@1000=0.793029, F1@5=0.509067
- `yso-en` (best epoch=1): NDCG@10=0.623692, NDCG@1000=0.750391, F1@5=0.441138
- `koko`   (best epoch=1): NDCG@10=0.356272, NDCG@1000=0.468751, F1@5=0.260805

Delta vs baseline `torch_mean_residual` (test metrics):
- `yso-fi`: −0.010533 NDCG@10, −0.006307 NDCG@1000, −0.012564 F1@5
- `yso-en`: −0.010352 NDCG@10, −0.006761 NDCG@1000, −0.006247 F1@5
- `koko`:   −0.001464 NDCG@10, +0.001564 NDCG@1000, −0.000766 F1@5

### Analysis

- Overall this clamp variant is **worse than the additive baseline** on both YSO datasets across all three metrics.
- On `koko` it is **very close to neutral**, with a small gain on deep ranking (NDCG@1000) but slight losses on
  NDCG@10 and F1@5.
- The fact that the best epoch is consistently **1** suggests either:
  - The tanh parameterization + default `LR=0.003` causes too-aggressive movement in `delta_raw`/`bias`, or
  - The clamp simply removes beneficial capacity (i.e., some labels benefit from larger residuals than `delta_max=0.25` permits).
- Unlike the multiplicative `globalxdelta` variant, this one still preserves the additive structure; the degradation
  therefore points more specifically to the **bounded residual capacity** (and/or the optimization dynamics induced by tanh).

Notes / follow-ups:
- Try a smaller learning rate (e.g. `LR=0.001`) to see if the post-epoch-1 drop is an optimization artifact.
- If keeping the clamp idea, consider increasing `delta_max` modestly (e.g. 0.5) or making it dataset-tuned, but that
  would reduce the “controlled comparison” nature unless done systematically.

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

### Results (2026-03-16)

Command:
- `./regenerate_scoreboard.sh --models torch_mean_residual_bias_per_model`

Best epoch is selected by **train subset NDCG@1000** early-stopping (same as baseline).

Test metrics (best epoch per dataset from run output):
- `yso-fi` (best epoch=1): NDCG@10=0.683045, NDCG@1000=0.795515, F1@5=0.513818
- `yso-en` (best epoch=1): NDCG@10=0.623849, NDCG@1000=0.747570, F1@5=0.439277
- `koko`   (best epoch=4): NDCG@10=0.351774, NDCG@1000=0.466812, F1@5=0.258286

Delta vs baseline `torch_mean_residual` (test metrics):
- `yso-fi`: −0.004353 NDCG@10, −0.003821 NDCG@1000, −0.007813 F1@5
- `yso-en`: −0.010195 NDCG@10, −0.009582 NDCG@1000, −0.008108 F1@5
- `koko`:   −0.005962 NDCG@10, −0.000375 NDCG@1000, −0.003285 F1@5

### Analysis

- This variant is a **consistent regression vs the baseline** across all datasets/metrics in this run.
  The added capacity (3 extra scalars) did not translate into improved ranking; the most notable drop
  is on `yso-en` (≈−0.01 on both NDCG@10 and NDCG@1000).
- Early stopping behavior:
  - `yso-fi` and `yso-en` chose **epoch 1**, and the train-subset NDCG@1000 degraded slightly afterwards,
    suggesting the per-model bias quickly moves the model away from a good initial regime.
  - `koko` chose **epoch 4**, indicating the extra bias can be fit without immediate collapse there, but it
    still did not improve test ranking.
- Plausible interpretation: with inputs already log1p-transformed and **non-negative**, adding an additive
  per-model offset inside the weighted sum effectively introduces a second set of global degrees of freedom
  (it behaves like a shift scaled by global/residual weights). This can partially duplicate the role of the
  existing per-label bias term, while being less directly aligned with label base rates, which may make it
  easier to overfit the early-stop subset without improving generalization.

Notes / follow-ups:
- If keeping this variant for completeness, consider adding a small L2 penalty on `bias_model` (even 1e-4),
  or reparameterize it as a *multiplicative* per-model scale (closer to calibration) rather than an additive
  offset. Both would be new changes, so they should be evaluated as separate variants.

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
