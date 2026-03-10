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

### 2) Logits + BCEWithLogitsLoss is a strong default for ranking
Models trained on **raw logits** (no output clamp/sigmoid in the model) with `BCEWithLogitsLoss`:
- `torch_per_label`
- `torch_per_label_l1_delta`
- `torch_mean_residual`

These are among the strongest and most stable approaches on `yso-*`. They also avoid clamp-induced gradient saturation.

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

## What has not worked (or is consistently risky)

### 1) Big MLPs over flattened (models × labels) inputs are unstable / degrade ranking
The “NN correction” family tends to underperform strong linear/logit baselines in global leaderboards, and can be dramatically worse:
- `torch_nn_mlp_only` is especially poor (very low scores on yso-*).
- MLP correction models can win on some isolated cases but aren’t consistently top.

Hypothesis: flattening creates huge parameter interactions; optimization focuses on calibration-like behavior rather than ranking; and/or the signal is too sparse per label.

### 2) Probability-space training with hard clamp is risky
Several experimental models output probabilities and use `BCELoss` while clamping to [0,1]. This can:
- Saturate gradients (especially at 0/1).
- Make learned corrections “stick” at the bounds.

Mitigations in code:
- `torch_lowrank_residual_epsclamp` clamps to `[eps, 1-eps]`.
- `torch_lowrank_residual_sigmoid` uses `sigmoid(out_lin)` then clamps to `[eps, 1-eps]`.

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
4. `koko` may have different signal/noise properties than `yso-*`, benefiting from stronger sparsity priors or different preprocessing.

---

## Next experiments (grounded in current findings)

Prioritize new models that:
- Preserve the strong baselines,
- Add capacity in controlled ways,
- Prefer logits + BCEWithLogitsLoss unless there’s a strong reason otherwise.

### A) “Per-label logits” + low-rank coupling on delta (logit-space)
Create a logit-space variant combining:
- `torch_per_label` (logits)
- low-rank coupling on `(W - W0)` rather than probability-space clamp models

Sketch:
- `W = W0 + (U @ V)` where `U` is (M×R) and `V` is (R×L)
- Loss: `BCEWithLogitsLoss` + `lambda_uv * ||U||^2 + ||V||^2` + optional L1 on delta
This aims to keep the benefits of `torch_per_label` while enabling limited sharing across labels.

### B) Sparsity variants tuned per dataset
Since `koko` likes L1 delta:
- Try dataset-dependent `lambda_l1` or an automatic schedule (start high → decay).
- Consider “group sparsity” per label (encourage selecting one model per label).

### C) Separate objectives for top-k vs deep-k
Some models may optimize NDCG@10 at the expense of NDCG@1000. Consider:
- Two-stage training (optimize logits for NDCG@1000 proxy, then fine-tune for NDCG@10 proxy).
- Or a weighted early-stop metric blending multiple k’s.

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
