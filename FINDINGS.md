# Findings (current experiments)

This document summarizes **what seems to work** and **what has not worked (so far)** in this repository, based on the current model implementations and the current `SCOREBOARD.md`.

Scope:
- Focus is **ranking quality** (NDCG@k, F1@k), not calibration.
- Results are empirical and may change as experiments are rerun.
- Unless stated otherwise, “best” refers to the Top-10 sections in `SCOREBOARD.md`.

---

## Snapshot of best results (from current scoreboard)

### Global (avg across datasets)
- Best Avg **Test NDCG@10**: `torch_lowrank_residual` (~0.5716)
- Best Avg **Test NDCG@1000**: `torch_mean` (~0.6695)
- Best Avg **Test F1@5**: `torch_lowrank_residual` (~0.4216)

### Per-dataset best by composite (avg of Test NDCG@1000, NDCG@10, F1@5)
- `koko`: `torch_per_label_l1_delta` (~0.3672)
- `yso-en`: `torch_mean_residual` (~0.6204)
- `yso-fi`: `torch_per_label` (~0.6903)

Takeaway: **No single architecture dominates all datasets**, but a few families are consistently strong:
- `torch_per_label*` for logits-based per-label linear models
- `torch_mean*` for log1p-preprocessed mean-like models
- Some low-rank mixing/residual variants can win on specific metrics, but are less universally best.

---

## What works (reliably helpful patterns)

### 1) Strong, simple baselines are hard to beat
- **Per-label linear ensemble on logits** (`torch_per_label`) is a top performer overall and wins on `yso-fi` composite.
- **Mean-like ensemble with log1p preprocessing** (`torch_mean`) is the best on Avg Test NDCG@1000 across datasets.
- **Residual shrinkage around a good prior** (`torch_mean_residual`) is best on `yso-en` composite.

Pattern: start from a strong baseline, then add **small, controlled extra capacity** with a bias toward “do no harm”.

### 2) Logits + BCEWithLogitsLoss is a strong default for ranking (but imbalance matters)
Models trained on **raw logits** (no output clamp/sigmoid in the model) with `BCEWithLogitsLoss`:
- `torch_per_label`
- `torch_per_label_l1_delta`
- `torch_mean_residual`

These are among the strongest and most stable approaches on `yso-*`.

However, when you introduce additional degrees of freedom (e.g. an MLP correction), the **extreme class imbalance** (very sparse true labels) can create a strong optimization incentive to push scores downward to satisfy the many negatives. In such cases, additional controls are needed (bounded corrections, regularization, or different loss weighting).

### 3) Preprocessing matters (and should stay outside the model)
Successful models generally use a fixed monotonic transform:
- `log1p(clamp(x,0))` for logits-based linear models and torch_mean family.
- Some NN-style models use `sqrt(clamp(x,0))`.

Having preprocessing **outside** the model is consistently enforced in the code (and appears beneficial), because it prevents the optimizer from “undoing” calibration/scale benefits.

### 4) Good initialization helps (and is already implemented for some models)
Dataset-specific initialization (`ensemble3_init_weights`) is used in:
- `torch_per_label` (+ diagnostics)
- `torch_per_label_l1_delta`
- `torch_mean_residual` and many low-rank variants (as `init_global`)

This reduces optimization burden and makes “residual” parameterizations meaningful (start near a reasonable solution).

### 5) Sparsity/regularization on deviations can help for some datasets
- `torch_per_label_l1_delta` wins the `koko` composite leaderboard.
- The L1 penalty is applied to **delta from initialization** (not raw weights), a sensible “shrink-to-prior” formulation.

Takeaway: `koko` likely benefits from more “model selection per label” behavior (sparse deviations).

---

## torch_mean_residual_mlp: what we learned

`torch_mean_residual_mlp` was created to keep the strengths of `torch_mean_residual` (strong baseline on yso-*) while adding **cross-label influence** via an MLP correction applied only to active labels (similar in spirit to `torch_nn_split`).

### Observed failure mode: MLP learns a global negative suppressor (ranking collapse)
Without strong constraints, the MLP path quickly learned large negative corrections on active labels, dominating the base logits and destroying ranking (especially on `koko`). This presented as:
- Loss improving, while train-subset NDCG@1000 collapsed toward 0
- `mlp_add(active)` becoming large and negative (min values reaching large magnitude)
- MLP delta weights growing rapidly (`delta_layer.weight` magnitude increasing)

This is consistent with the imbalance-driven incentive: pushing many scores down reduces BCE on negatives, but hurts top-k ranking.

### Stabilization steps that helped
We added instrumentation (`decomp_debug`) to attribute effects to base logits vs MLP contribution and to inspect parameter distributions.

Stabilizers that prevented catastrophic collapse:
- **Bounded MLP gate**: `alpha = alpha_max * sigmoid(log_alpha)` with small `alpha_max`
- **Enable weight decay** (`AdamW(weight_decay=0.01)`) to curb rapid MLP weight growth
- **Increase probability clamp epsilon** (`eps=1e-3`) when inspecting sigmoid probabilities (helps detect approaching saturation and makes clamp relevant earlier)

These made the MLP contribution much smaller and stopped the “runaway suppression” behavior, though overall improvements vs the best baselines were limited.

### MLP delta bias was not the primary culprit
We suspected `delta_layer.bias` might behave like an extra per-label bias vector. Debug stats showed the main collapse was driven by **delta_layer weights and hidden activations**, not primarily by the bias term. A small bias regularizer was still added as a safety measure.

### Loss experiments: prob_epsclamp vs logits
We added a `prob_epsclamp` mode (sigmoid -> clamp -> BCELoss) because `torch_nn*` models did well with probability outputs.

Empirically:
- Switching to `prob_epsclamp` alone did **not** resolve the collapse when the MLP was unconstrained.
- Clamp at very small eps (e.g. 1e-5) does not become active until logits are extremely negative; this was too late to prevent ranking damage.
- The main stability gains came from gating/weight decay, not from clamp alone.

### pos_weight experiments: easy to misapply and easy to overdo
We attempted per-label `pos_weight` (neg/pos) in `BCEWithLogitsLoss` to counteract imbalance.

Lessons:
- It is easy to accidentally compute a weighted logits loss but still train with an unweighted prob loss (making the change ineffective).
- Once correctly applied, aggressive per-label `pos_weight` (even capped) can substantially change optimization dynamics and may degrade ranking, especially when the baseline already encodes strong ranking signal.

Current takeaway: **pos_weight is not a free win** here; if revisited, it likely needs gentler caps and/or only applying weighting on a subset of labels (e.g. active labels).

### Key difference vs torch_nn_split
`torch_nn_split` avoids the “negative runaway” behavior via multiple built-in safeties:
- it operates in **probability space** with outputs clamped to `[0,1]`
- it uses a strict **convex mean mixer** (softmax weights) as the baseline
- the correction is added on top of a bounded baseline and then clamped again

In contrast, `torch_mean_residual_mlp` operates in logit space with additional unconstrained residual/bias terms; without careful gating, this gives the MLP more room to learn degenerate global suppression.

---

## What has not worked (or is consistently risky)

### 1) Big MLPs over flattened (models × labels) inputs are unstable / degrade ranking
The “NN correction” family tends to underperform strong linear/logit baselines in global leaderboards, and can be dramatically worse:
- `torch_nn_mlp_only` is especially poor (very low scores on yso-*).
- MLP correction models can win on some isolated cases but aren’t consistently top.

Hypothesis: flattening creates huge parameter interactions; optimization focuses on calibration-like behavior rather than ranking; and/or the signal is too sparse per label.

### 2) Probability-space training with hard clamp is risky (and eps matters)
Several experimental models output probabilities and use `BCELoss` while clamping to [0,1]. This can:
- Saturate gradients (especially at 0/1).
- Make learned corrections “stick” at the bounds.

Mitigations in code:
- `torch_lowrank_residual_epsclamp` clamps to `[eps, 1-eps]`.
- `torch_lowrank_residual_sigmoid` uses `sigmoid(out_lin)` then clamps to `[eps, 1-eps]`.

Note: for sigmoid outputs, eps must be large enough to become relevant before logits reach extreme magnitude (e.g. `eps=1e-3` activates at ~-6.9 logits; `eps=1e-5` at ~-11.5).

Despite mitigations, probability-space variants are not consistently better than logits-based approaches.

### 3) Cross-label mixing is not a free win
Cross-label mixing families:
- `torch_lowrank_mix*` (mixing only)
- `torch_lowrank_residual_mix*` (mixing + residual weights)

These can be strong on some metrics/datasets, but are not consistently best overall and introduce more knobs (rank, mix_rank, regularizers, normalization tricks).
Example: mixing can help NDCG@10 in some cases while hurting NDCG@1000 (or vice versa).

Takeaway: cross-label mixing is promising but needs careful control and evaluation, especially across datasets.

---

## Model family notes (based on current scoreboard + code)

### Baselines (no learning)
- `bonsai`, `fasttext`, `mllm`, `nn`, etc. are useful references.
- Ensembles (`mean`, `mean_weighted`) show that even trivial combinations can help, but learned torch variants can do better.

### `mean_weighted`
- Often near `torch_mean`/`torch_mean_residual` for some metrics.
- Limited by coarse grid step (0.1) and inability to do per-label specialization.

### `torch_mean`
- Best Avg Test **NDCG@1000** overall.
- It’s simple and stable; acts as a strong “global weight” learner.

### `torch_mean_residual`
- Best `yso-en` composite.
- “Global weights + per-label residual + bias + explicit L2” seems like a good template for controlled extensions.

### `torch_per_label`
- Best `yso-fi` composite, very strong across yso-*.
- Per-label weights allow specialization where some base models dominate specific label subsets.

### `torch_per_label_l1_delta`
- Best `koko` composite.
- L1 on delta encourages sparse per-label deviations from a good global initialization.

### NN-based families (`torch_nn*`)
- Inconsistent; some are decent but not leading overall.
- Pure MLP is clearly not good in this current setup.
- If continuing this direction, focus on constrained capacity and/or per-label decomposition (see “next experiments”).

### Low-rank residual/mix families
- `torch_lowrank_residual` wins Avg Test NDCG@10 and Avg Test F1@5 overall.
- But other low-rank + mixing variants vary; some are strong, others marginal.

Interpretation: low-rank structure can capture useful shared label structure, but the training setup (probabilities vs logits, regularization) matters a lot.

---

## Methodological constraints that seem important (keep doing)

### Early stopping policy
All torch scripts aim to early-stop using **train subset NDCG@1000** to avoid test leakage. Keep this.
- Note: several scripts currently compute test metrics each epoch for convenience. This is “observational leakage” (not used for selection) but can still bias human decisions. Prefer printing test metrics only at the best epoch (like `torch_per_label` does via snapshot).

### CPU/GPU data handling
Most scripts keep X on CPU and move minibatches to GPU; prediction uses batched forward then moves outputs back to CPU for metric evaluation. This is a good stability/perf compromise.

### Dataset-specific initialization
Continue using `DatasetConfig.ensemble3_init_weights` for any ensemble that can be reasonably initialized.

---

## Hypotheses (why these patterns occur)

These are working hypotheses to guide next experiments:

1. **Ranking improves when the model preserves relative ordering** and doesn’t overfit calibration.
2. **Per-label independence is strong** because label spaces are huge and sparse; learning label couplings helps only when heavily regularized (low-rank).
3. **Training in logits space** avoids clamp/sigmoid saturation and provides smoother optimization for ranking-relevant improvements.
4. **Class imbalance interacts with extra capacity**: once a model can apply broad shifts across labels, BCE objectives can reward global suppression unless constrained.
5. `koko` may have different signal/noise properties than `yso-*`, benefiting from stronger sparsity priors or different preprocessing.

---

## Next experiments (grounded in current findings)

Prioritize new models that:
- Preserve the strong baselines,
- Add capacity in controlled ways,
- Prefer logits + BCEWithLogitsLoss unless there’s a strong reason otherwise.

### A) Prefer low-rank coupling over full MLP flattening
The main lesson from `torch_mean_residual_mlp` is that a full flattened MLP has too many ways to learn degenerate “global shifts”.
Prefer cross-label coupling with explicit structure:
- low-rank mixing/residuals (as in existing `torch_lowrank_*` models)
- low-rank delta on a strong prior (like `torch_per_label_l1_delta` but low-rank)

### B) If keeping an MLP correction, keep hard safety rails
If we continue exploring this family:
- keep bounded `alpha` (small `alpha_max`)
- keep weight decay for MLP parameters
- keep decomp-style diagnostics to catch when the correction dominates

### C) Revisit imbalance handling carefully
If revisiting `pos_weight`:
- use much smaller caps (e.g. 10–20)
- consider weighting only active labels (inactive labels get pos_weight=1)
- or use a single global pos_weight (gentler)

### D) Reduce test-metric printing during training
For reproducibility and to reduce human-in-the-loop leakage:
- Standardize scripts to compute test metrics only when saving a new best snapshot.

---

## Notes / caveats

- Some scoreboard rows have missing train metrics (e.g. `nn` baseline rows). Avoid drawing conclusions from missing fields.
- Results depend on:
  - preprocessing choice,
  - early-stop subset size,
  - regularization strengths,
  - and the dataset.

When adding new experiments, always compare against:
- `torch_per_label`
- `torch_mean`
- `torch_mean_residual`
- `mean_weighted`

These form a stable “regression suite” of baselines.
