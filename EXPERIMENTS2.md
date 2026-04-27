# Experiments 2: New candidates to beat `torch_per_label`

This document proposes a small set of follow-up experiments aimed at finding a model that beats
the current `torch_per_label` baseline on:

1) ideally **all three datasets** and **all three metrics** (Test NDCG@10, Test NDCG@1000, Test F1@5), or
2) minimally on the repo’s primary criterion: **Avg of 3 test metrics, averaged across datasets**.

This file follows the style of `EXPERIMENTS.md`: each variant changes one core idea, keeps the
existing training/evaluation protocol (BCEWithLogitsLoss + early stopping on train subset NDCG@1000),
and leaves Results/Analysis sections to be filled in after running.

---

## Baseline: `torch_per_label`

Reference model:
- Script: `benchmarks/torch_per_label.py`
- Model name format written to scoreboard:
  - `torch_per_label(<k1>,<k2>,<k3>)`
- Notes:
  - Per-label weights `W[m,l]` and per-label bias `b[l]`
  - Inputs are log1p-preprocessed
  - Early stopping uses train-subset NDCG@1000

Goal of the experiments below: preserve the strong aspects of per-label fitting while improving
cross-dataset generalization and long-tail ranking stability.

---

## Variant 1: `torch_per_label_softmax_global_scale`

**What changes:** start from `torch_per_label_softmax_global`, but restore **global scale freedom**
with a learned scalar.

Baseline softmax-global form:
- `g = softmax(g_raw)` (M,), `w_eff[m,l] = g[m] + w_delta[m,l]`

Proposed change:
- Add a learned scalar `s > 0`, parameterized via `log_s` and use `s = exp(log_s)`.
- Effective weights:
  - `w_eff[m,l] = s * g[m] + w_delta[m,l]`

**Hypothesis:**
- Softmax global weights are consistently strong (especially yso-en / koko), but can be slightly
  too restrictive because they remove overall amplitude freedom.
- A learned `s` should help recover any dataset-specific scaling needed (potentially improving yso-fi)
  while retaining the stability of a convex mixture.

### Implementation

- Script:
  - `benchmarks/torch_per_label_softmax_global_scale.py`
- Model name written to scoreboard:
  - `torch_per_label_softmax_global_scale(<k1>,<k2>,<k3>)`
- Training/eval:
  - Same protocol as `torch_per_label` (BCEWithLogitsLoss + early stopping on train-subset NDCG@1000)
  - Explicit L2 penalty on `w_delta` (module constant `LAMBDA_DELTA_L2 = 1e-3`)
  - Same “best hyperparameters” as `torch_per_label` for comparability (lr=0.003, wd=0, bs=256)

### Results

Command:
- `./regenerate_scoreboard.sh --models torch_per_label_softmax_global_scale`

Early stopping (selected by best train-subset NDCG@1000):
- `yso-fi`: best_epoch=4, train_ndcg@1000(subset)=0.818671
- `yso-en`: best_epoch=11, train_ndcg@1000(subset)=0.818032
- `koko`: best_epoch=2, train_ndcg@1000(subset)=0.561877

Test metrics (best epoch per dataset):
- `yso-fi`: NDCG@10=0.707331, NDCG@1000=0.814153, F1@5=0.542623
- `yso-en`: NDCG@10=0.660923, NDCG@1000=0.775213, F1@5=0.472158
- `koko`: NDCG@10=0.364180, NDCG@1000=0.476558, F1@5=0.266555

Summary vs baseline `torch_per_label` (from current `SCOREBOARD.md`):
- `yso-fi`: slightly worse across all three metrics
  - ΔNDCG@10 = -0.002840 (0.707331 vs 0.710171)
  - ΔNDCG@1000 = -0.002301 (0.814153 vs 0.816454)
  - ΔF1@5 = -0.001509 (0.542623 vs 0.544132)
- `yso-en`: better NDCG@10 / NDCG@1000, worse F1@5
  - ΔNDCG@10 = +0.001696 (0.660923 vs 0.659227)
  - ΔNDCG@1000 = +0.004134 (0.775213 vs 0.771079)
  - ΔF1@5 = -0.001469 (0.472158 vs 0.473627)
- `koko`: better across all three metrics
  - ΔNDCG@10 = +0.002537 (0.364180 vs 0.361643)
  - ΔNDCG@1000 = +0.002653 (0.476558 vs 0.473905)
  - ΔF1@5 = +0.001828 (0.266555 vs 0.264727)

Across-datasets aggregate:
- This variant is currently **#1** by “Avg of 3 Test Metrics (across datasets)” in `SCOREBOARD.md`
  (slightly ahead of `torch_per_label_softmax_global` and `torch_per_label`).

### Analysis

- The learned global scale `s` appears to help most where the pure softmax-global constraint is likely
  too restrictive, and where per-label flexibility tends to overfit:
  - Clear win on `koko` across all metrics, suggesting improved stability / calibration of the shared mixture.
  - Noticeable deep-ranking improvement on `yso-en` NDCG@1000 (+0.0041), consistent with the “stabilize
    the shared mixture, let residuals do the rest” hypothesis.
- The small regression on `yso-fi` suggests either:
  - the dataset-specific optimum is already well captured by unconstrained `torch_per_label`, or
  - the combination `(s * softmax(g_raw)) + w_delta` with L2 shrinkage on `w_delta` is still slightly
    too constrained for `yso-fi` (i.e., the best solution needs more per-label deviation than the current
    delta penalty encourages).
- The F1@5 tradeoff on `yso-en` (slightly down while NDCG improves) suggests this variant may be
  marginally better at ordering/ranking quality than at top-5 threshold-ish behavior, which aligns with
  the repo’s primary focus.
- Next follow-ups (if pursuing further gains):
  - Tune `LAMBDA_DELTA_L2` per dataset or via a small shared grid (e.g. 3e-4, 1e-3, 3e-3) to see whether
    `yso-fi` can recover without losing the `koko` benefit.
  - Consider anchoring `g` to init weights (Variant 2) to reduce drift and potentially improve `yso-fi`
    while maintaining the strong cross-dataset average.

---

## Variant 2: `torch_per_label_softmax_global_l2_anchor`

**What changes:** add an explicit penalty anchoring the global simplex weights `g` to the dataset’s
provided initialization `g0` (from `DatasetConfig.ensemble3_init_weights`, else uniform).

Keep:
- `g = softmax(g_raw)`
- `w_eff[m,l] = g[m] + w_delta[m,l]`

Add regularization term:
- `loss += lambda_global * mean((g - g0)^2)`

**Hypothesis:**
- In `torch_mean_residual`, a global-weight anchor produced the most consistent improvements across
  datasets (Variants 6/7 in `EXPERIMENTS.md`).
- Applying the same concept here may prevent harmful drift in the shared mixture and reduce sensitivity
  to dataset-specific quirks, while still allowing per-label `w_delta` to do the fine-grained work.

### Implementation

- New script:
  - `benchmarks/torch_per_label_softmax_global_l2_anchor.py`
- Model name:
  - `torch_per_label_softmax_global_l2_anchor(<k1>,<k2>,<k3>)`
- Notes:
  - Keep `lambda_global` as a module-level constant for controlled comparison.

### Results (TODO)

Command:
- `./regenerate_scoreboard.sh --models torch_per_label_softmax_global_l2_anchor`

Best epoch selection:
- TODO

Test metrics:
- TODO

### Analysis (TODO)
- TODO

---

## Variant 3: `torch_per_label_global_plus_delta_l2_delta`

**What changes:** revisit the `torch_per_label_global_plus_delta` reparameterization, but add explicit
L2 shrinkage on `w_delta` (the missing stabilizer implied by earlier results in `PER_LABEL.md`).

Parameters:
- `w_global[m]` (M,)
- `w_delta[m,l]` (M,L)
- `bias[l]` (L,)

Effective weights:
- `w_eff[m,l] = w_global[m] + w_delta[m,l]`

Add regularization term:
- `loss += lambda_delta * mean(w_delta^2)`

**Hypothesis:**
- The raw reparameterization alone underperformed the baseline `torch_per_label`, likely due to drift /
  overfitting in the residual degrees of freedom.
- L2 shrinkage should force the model to behave like a global mixture unless there is consistent per-label
  evidence to deviate, potentially improving generalization and especially long-tail metrics (NDCG@1000).

### Implementation

- New script:
  - `benchmarks/torch_per_label_global_plus_delta_l2_delta.py`
- Model name:
  - `torch_per_label_global_plus_delta_l2_delta(<k1>,<k2>,<k3>)`
- Notes:
  - Keep the existing training loop policy (same early stopping, same fixed preprocessing).

### Results (TODO)

Command:
- `./regenerate_scoreboard.sh --models torch_per_label_global_plus_delta_l2_delta`

Best epoch selection:
- TODO

Test metrics:
- TODO

### Analysis (TODO)
- TODO

---

## Variant 4: `torch_per_label_freq_weighted_anchor`

**What changes:** keep the baseline `torch_per_label` free per-label weights, but add a **frequency-weighted**
anchor penalty that shrinks rare-label weights more strongly.

Let:
- `freq[l] = number of positive training examples for label l` (from `y_train_true`)
- `alpha[l] = 1 / sqrt(freq[l] + 1)`, normalized so `mean(alpha)=1`

Add a weight anchor penalty:
- Choose an anchor point `W0[m]` (dataset init weights or uniform).
- `loss += lambda_w * mean_{m,l}( alpha[l] * (W[m,l] - W0[m])^2 )`

(Optionally also apply a frequency-weighted bias penalty; keep that out initially to isolate the effect.)

**Hypothesis:**
- `torch_per_label` is extremely flexible; its likely failure mode is tail-label overfitting that harms
  deep ranking quality (NDCG@1000) and cross-dataset stability.
- Frequency-weighted shrinkage preserves head-label capacity while protecting rare labels.

### Implementation

- New script:
  - `benchmarks/torch_per_label_freq_weighted_anchor.py`
- Model name:
  - `torch_per_label_freq_weighted_anchor(<k1>,<k2>,<k3>)`
- Notes:
  - Keep `lambda_w` a module constant for controlled comparison.
  - Use dataset `ensemble3_init_weights` as `W0` if available, else uniform.

### Results (TODO)

Command:
- `./regenerate_scoreboard.sh --models torch_per_label_freq_weighted_anchor`

Best epoch selection:
- TODO

Test metrics:
- TODO

### Analysis (TODO)
- TODO

---

## Variant 5: `torch_per_label_mlp_truth_active_only`

**What changes:** in `torch_per_label_mlp`, change the stage-2 active label set to be **truth-only**
(active if label appears at least once in train truth), instead of truth OR any train prediction.

Current baseline stage-2 active policy:
- active if label appears in train truth OR in any train prediction CSR indices

Proposed policy:
- active if label appears in train truth (optionally with a small minimum frequency threshold in a later follow-up)

**Hypothesis:**
- Several MLP stage-2 variants show the pattern “mostly no-op on yso-*; can harm `koko` NDCG@1000 unless near-no-op”.
- The current broad active set likely includes many “prediction-only” labels; even small residual behavior can perturb
  cross-block ordering and degrade deep ranking (NDCG@1000).
- Restricting stage-2 edits to truth-observed labels should reduce unintended long-tail perturbations and make stage 2
  safer to turn on.

### Implementation

- New script:
  - `benchmarks/torch_per_label_mlp_truth_active_only.py`
- Model name:
  - `torch_per_label_mlp_truth_active_only(<k1>,<k2>,<k3>)`
- Notes:
  - Keep everything else identical to `torch_per_label_mlp` for a controlled test.

### Results (TODO)

Command:
- `./regenerate_scoreboard.sh --models torch_per_label_mlp_truth_active_only`

Best epoch selection:
- TODO

Test metrics:
- TODO

### Analysis (TODO)
- TODO
