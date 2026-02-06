# Ensemble Experiments – Findings

This document summarizes what has **worked** and **not worked** so far while experimenting with ensemble methods and losses in this repository.  
The conclusions below are based on repeated runs and the results recorded in `SCOREBOARD.md`.

---

## ✅ What has worked well

### 1. Per‑label linear ensemble with bias (best overall)
**Models:**  
- `PerLabelWeightedEnsemble`  
- `PerLabelWeightedConvEnsemble` (Conv1d-based summation)

**Loss:** `BCEWithLogitsLoss` (unweighted)  
**Evaluation:** raw logits (no sigmoid)

- This is the **best-performing approach** on both:
  - **Test NDCG@10**
  - **Test NDCG@1000**
- Peak performance occurs **early** in training.

**Best checkpoints observed:**
- `torch_per_label_conv_epoch03`
  - Test NDCG@10 ≈ **0.715**
  - Test NDCG@1000 ≈ **0.821**
- `torch_per_label_epoch03`
  - Test NDCG@10 ≈ **0.713**
  - Test NDCG@1000 ≈ **0.820**

Key properties that matter:
- Raw logits (no sigmoid before NDCG)
- Per-label bias term
- No extra constraints on weights
- Simple linear structure
- **Parameterization matters**: replacing an explicit `.sum(dim=1)` with a
  fixed Conv1d summation (`kernel_size=1`, no bias) yields a **small but
  consistent improvement**, despite identical expressiveness

Interpretation:
- The Conv1d-based summation changes gradient geometry and optimizer interaction
- This improves early-epoch behavior, where peak NDCG is observed
- The gain is small but consistent across NDCG@10, NDCG@1000, and F1@5

---

### 2. Frequency-aware per-label weights can improve ranking (small, controlled residuals)
**Model:** `torch_per_label_freq_gate`  
**Mechanism:** per-label weights are adjusted using label-frequency features via a **regularized residual**:

```
w_eff[m,l] = base_w[m,l] + alpha * delta_w[m,l](freq[l])
```

Key implementation details that mattered:
- The frequency component is a **residual correction**, not a multiplicative gate.
- The residual is **explicitly regularized** (L2 penalty on `delta_w`) so it stays small unless it helps.
- Diagnostics were added to inspect what the frequency module actually learns.

Observed outcome (from the current `SCOREBOARD.md`):
- `torch_per_label_freq_gate` is **#1 on Test NDCG@10** and **#1 on Test NDCG@1000**
- It is essentially tied with `torch_per_label` on **Test F1@5** (slightly lower in the current scoreboard)

Interpretation:
- Label frequency contains useful signal for ranking, but it must be injected in a way that does not destabilize the strong per-label baseline.
- The residual + regularization approach behaves like a “safe prior”: it can help without rewriting the model.

#### What the diagnostics revealed (important)
The frequency module is **not** doing a dramatic “switch which model to trust” behavior. Instead:

1) **The frequency effect is small in magnitude**
- Example run: `alpha ≈ 0.25`
- RMS(`base_w`) ≈ 0.33
- RMS(`alpha * delta_w`) ≈ 0.007

So the frequency correction is only a few percent of the base weight scale. This matches the intuition that the per-label weights already learn most label-specific trust patterns, and frequency adds a small but meaningful adjustment.

2) **Global trust across models barely changes**
Averaged over all labels, mean weights per model change only slightly. The gains come from **label-dependent** adjustments, not from “trust model X more overall”.

3) **The learned effect looks like frequency-dependent shrinkage, not pure trust redistribution**
Across frequency bins, the residual deltas often push **all models down together** for certain frequency regimes (especially mid-frequency labels), rather than increasing one model while decreasing another.

This suggests the frequency module is acting more like:
- a **frequency-conditioned calibration/shrinkage** mechanism (reducing overconfident contributions in certain regimes),
than:
- a clean “model A for frequent labels, model B for rare labels” selector.

4) **The strongest adjustments occur in mid-frequency bins**
The largest negative deltas were observed for labels with counts like `6–20` and `21–100`, while zero-shot labels (`count==0`) had near-uniform weights and tiny deltas.

Practical takeaway:
> Frequency features can help, but in this parameterization they are primarily used to learn a *frequency-conditioned scaling/shrinkage* of contributions, not a hard or soft routing between base models.

This also suggests a future experiment:
- If we want to force “trust redistribution” rather than “shrink everyone”, constrain `delta_w` to be **zero-mean across models per label** (so it can only reallocate weight between models, not scale all of them together).

#### Update: forcing trust redistribution + bounded alpha still works (and clarifies behavior)
A follow-up refactor implemented two constraints:

- **Zero-mean delta across models per label**  
  `mean_m delta_w[m,l] = 0` for every label `l`  
  → the frequency module can only **redistribute trust** between base models, not scale all of them together.

- **Bounded, nonnegative alpha**  
  `alpha = alpha_max * sigmoid(alpha_raw)`  
  → prevents runaway scaling and keeps the frequency effect stable.

Observed outcome:
- The model **still** achieves the best NDCG metrics in the current scoreboard:
  - `torch_per_label_freq_gate`: Test NDCG@10 **0.709273**, Test NDCG@1000 **0.815026**
  - `torch_per_label`: Test NDCG@10 **0.708496**, Test NDCG@1000 **0.814356**
- Test F1@5 remains essentially tied (freq-gated is slightly lower in the current scoreboard).

What the new diagnostics show:
- The frequency module becomes an even smaller “nudge”:
  - Example run: `alpha ≈ 0.108`
  - RMS(`alpha * delta_w`) ≈ **0.000675** vs RMS(`base_w`) ≈ **0.327**
- The per-bin deltas now clearly sum to ~0 across models (up to rounding), confirming the redistribution constraint is active.
- The learned pattern is mostly:
  - **shift weight away from mLLM** for labels that appear in training (especially mid/high-frequency bins),
  - **slightly increase fastText (and sometimes Bonsai)** in those bins,
  - **near-zero changes for zero-shot labels** (`count==0`), i.e. the model does not strongly learn “mLLM for zero-shot” in this setup.

Practical takeaway:
> The best gains so far come from **very small, frequency-conditioned trust redistribution**, not large gating.  
> The model is using frequency to make subtle, label-regime-specific adjustments that improve ranking.

#### Update: regularizing the *applied* correction requires much larger λ (otherwise the gate becomes too strong)
A later refactor changed the regularizer from penalizing the raw residual to penalizing the *applied* correction:

Old:
```
loss = BCE + λ * mean(delta_w^2)
```

New:
```
loss = BCE + λ * mean((alpha * delta_w)^2)
```

This is conceptually cleaner (it penalizes what is actually applied), but it has an important practical consequence:

- In typical runs, `alpha ≈ 0.10`, so `alpha^2 ≈ 0.01`.
- Therefore, for the same `λ`, the new regularizer is effectively ~**100× weaker** on `delta_w`.

Observed behavior when keeping the old `λ` scale (e.g. `1e-4..1e-3`):
- `RMS(alpha * delta_w)` jumped from ~**7.8e-4** (small nudge) to **1e-2..1e-1** (large correction).
- The frequency module stopped being a “safe residual” and became a strong frequency-conditioned router.
- Diagnostics showed large shifts toward **fastText** and away from **mLLM** for labels seen in training.
- Training became unstable quickly: train-subset NDCG dropped sharply after epoch 1, so early stopping consistently selected **epoch 1**.
- A built-in sweep over `(alpha_max ∈ {0.5, 1.0, 2.0}, lambda_delta ∈ {1e-4, 3e-4, 1e-3})` did **not** recover the previous small-residual regime and did **not** beat the earlier best run.

Practical takeaway:
> If using the applied-correction regularizer, `lambda_delta` must be increased by roughly `1/alpha^2`.  
> With `alpha ≈ 0.1`, that means `lambda_delta` should be on the order of **1e-2..1e-1** (often around **0.1**) to match the previous effective strength.

#### Update: confirmed — larger λ restores the “small nudge” regime and improves metrics
After expanding `lambda_delta` into the correct range for the applied-correction regularizer, the model returned to stable training and small, meaningful corrections.

Runs (all with `alpha_max=0.5`, early-stop by train “mix” metric, best epoch = 4):

- `lambda_delta=0.03`
  - Test NDCG@10 = **0.709680**
  - Test NDCG@1000 = **0.815125**
  - Test F1@5 = **0.541443**
  - Diagnostics: `alpha ≈ 0.097`, RMS(`alpha * delta_w`) ≈ **0.001678**

- `lambda_delta=0.10`
  - Test NDCG@10 = **0.709472**
  - Test NDCG@1000 = **0.814974**
  - Test F1@5 = **0.539648**
  - Diagnostics: RMS(`alpha * delta_w`) ≈ **0.000837** (very close to the earlier best run’s ~0.00078)

- `lambda_delta=0.30`
  - Test NDCG@10 = **0.709147**
  - Test NDCG@1000 = **0.815096**
  - Test F1@5 = **0.539909**
  - Diagnostics: RMS(`alpha * delta_w`) ≈ **0.000266** (gate nearly inactive)

Interpretation:
- The best results occur when the frequency gate is a **small-but-nonzero** correction.
- Too-small `lambda_delta` (earlier sweep) makes the gate a strong router and hurts generalization.
- Too-large `lambda_delta` makes the gate nearly inactive and removes most of the benefit.

Practical takeaway:
> For applied-correction regularization, the useful `lambda_delta` range is roughly **0.03–0.1** (at `alpha_max=0.5`), where the model learns a stable, small correction and improves ranking.

#### Update: full sweep confirms λ scale, typical correction magnitude, and that alpha_max is not binding
A full grid sweep was run:

- `alpha_max ∈ {0.25, 0.5, 0.75}`
- `lambda_delta ∈ {0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30}`
- Early stopping selection metric: train-subset `"mix"` = `0.7*NDCG@10 + 0.3*NDCG@1000`

Key observations from the sweep:

1) **The corrected λ range is validated**
- Strong results consistently occur for `lambda_delta` in the **~0.02–0.10** range.
- `lambda_delta=0.01` tends to produce a noticeably larger correction (see below) without improving metrics.

2) **Best runs correspond to a small applied correction**
The best-performing configurations typically have:

- `alpha(best) ≈ 0.096–0.099`
- `RMS(alpha * delta_w)(best) ≈ 3e-4 .. 1.5e-3`

When `RMS(alpha * delta_w)` is larger (e.g. `~0.004–0.006` at `lambda_delta=0.01`), the gate is “stronger than needed” and does not improve results.

3) **alpha_max is not the limiting factor in this regime**
Across the sweep, `alpha(best)` stays near ~0.097 regardless of `alpha_max` being 0.25, 0.5, or 0.75.  
This indicates:
- `alpha_max` is not binding (the learned alpha is well below the cap),
- `lambda_delta` is the primary knob controlling correction strength.

4) **The early-stop selection metric is only weakly discriminative among near-ties**
Many configurations have very similar train-subset `"mix"` scores but slightly different test outcomes.  
The selection metric is good enough to find strong configs, but it is not a sharp discriminator once you are in the “small nudge” regime.

Example sweep-selected config (by train `"mix"`):
- `alpha_max=0.5, lambda_delta=0.05` (best epoch = 4)
- Test NDCG@10 = **0.709685**
- Test NDCG@1000 = **0.815720**
- Test F1@5 = **0.540462**
- `RMS(alpha * delta_w)(best) ≈ 0.00124`

Practical takeaway:
> Once λ is in the correct range, the model is robust: many nearby settings work.  
> The main goal is to keep `RMS(alpha * delta_w)` in the “small nudge” regime (roughly `1e-4..1e-3` order of magnitude), not to chase a single magic λ.

---

### 3. Frequency-aware regularization (shrinkage) is competitive but not clearly better than gating
**Model:** `torch_per_label_freq_reg`  
**Mechanism:** add frequency-scaled L2 penalties on weights and bias:

- `reg_w = mean_l g(l) * ||w[:,l]||^2`
- `reg_b = mean_l g(l) * b[l]^2`
- where `g(l)` is a function of label frequency (e.g. `1/sqrt(count+1)` or `1/(log1p(count)+1)`)

Observed outcome (from a sweep over `freq_mode ∈ {inv_sqrt, inv_log}` and small grids of `lambda_w`, `lambda_b`):
- Best results occur with **very small** regularization strengths (often `lambda_w=0` and small `lambda_b`).
- Moderate-to-large `lambda_w` (e.g. `1e-4..3e-4`) causes a **large drop** in test NDCG, indicating the per-label weights need freedom and do not tolerate strong shrinkage.
- The best run recorded in `SCOREBOARD.md` is competitive with the best models:
  - `torch_per_label_freq_reg`: Test NDCG@10 **0.708792**, Test NDCG@1000 **0.814641**, Test F1@5 **0.540988** (epoch 4)

Interpretation:
- Frequency-aware shrinkage can act as a mild regularizer / calibration prior, but it does not consistently beat the more direct frequency-residual approach (`torch_per_label_freq_gate`) on NDCG.
- The gains (when present) are small and within the “near-tie” regime; selection metrics on a small train subset can be weakly discriminative.

Practical takeaway:
> If you want the best NDCG, prefer `torch_per_label_freq_gate`.  
> If you want a simpler frequency-aware tweak that is still competitive (and can slightly help F1@5), `torch_per_label_freq_reg` is a reasonable option — but keep `lambda_w` very small.

---

### 4. Simple mean / mean+bias ensembles are strong baselines
**Models:** `torch_mean`, `torch_mean_bias`

- Extremely stable across epochs
- Strong performance at **NDCG@1000**
- Hard to beat without overfitting

These are excellent:
- Sanity checks
- Fallback submissions
- Recall-heavy metrics

---

### 5. Early epochs matter more than convergence
Across multiple runs:
- Training loss keeps improving
- **Test NDCG peaks early** (epoch 2–4)
- Later epochs consistently overfit

Even without automatic early stopping:
- Epoch selection is critical
- Epoch 3 is consistently near-optimal for per-label ensembles

---

### 6. Epoch ensembling is safe but offers limited gains

**Model:** `torch_per_label_conv_epoch02_03`  
**Method:** Simple convex combination of logits from two early checkpoints  
```
logits = α · logits_epoch02 + (1 − α) · logits_epoch03
```

Observed results:
- ✅ **Stable behavior**
- ✅ No collapse or degradation of ranking geometry
- ✅ Performance very close to the best single checkpoint
- ❌ Did **not** surpass the best epoch on Test NDCG

Example (Conv ensemble):
- Best single epoch (`epoch03`):
  - Test NDCG@10 ≈ **0.71518**
  - Test NDCG@1000 ≈ **0.82085**
- Epoch 02+03 ensemble:
  - Test NDCG@10 ≈ **0.71437**
  - Test NDCG@1000 ≈ **0.82033**

Interpretation:
- Early epochs are already very similar in ranking geometry
- The best checkpoint (epoch 03) is already near-optimal for NDCG
- Averaging slightly reduces sharpness, trading a tiny amount of peak ranking
  for robustness

Key takeaway:
> **Epoch ensembling is a safe, inference-only technique that preserves ranking,
> but it does not meaningfully improve over a well-chosen early checkpoint.**

Practical implication:
- Use **epoch 03 alone** for maximum peak performance
- Use **epoch ensembling** as a robustness hedge if checkpoint selection
  uncertainty is a concern

---

### 7. Sub-linear input scaling significantly improves results (sqrt → log1p)
Applying a simple preprocessing step to **all ensemble inputs** had a **large positive impact** on both:
- **Test NDCG@10**
- **Test NDCG@1000**

#### First improvement: `sqrt`
```
x -> sqrt(x)
```

Key observations:
- `sqrt` is monotonic, so per-model ranking is preserved
- Large (overconfident) scores are compressed more than small ones
- Over-dominance by any single base model is reduced

This revealed that:
- Base model outputs are **systematically over-confident**
- The ensemble benefits more from **relative confidence** than absolute magnitude
- Calibration and scale matter more than additional model complexity

#### Further improvement: `log1p`
Replacing `sqrt` with a slightly stronger sub-linear transform:

```
x -> log1p(x)
```

led to a **small but consistent additional improvement**:
- Higher peak **Test NDCG@10**
- Higher peak **Test NDCG@1000**
- Same overfitting pattern (early peak at epoch 2–3)

Interpretation:
- Very large base-model scores were still too dominant under `sqrt`
- `log1p` applies stronger damping to extreme values while remaining monotonic
- This further reduces the impact of overconfident false positives, especially for NDCG@1000

This also explains earlier results:
- Sigmoid hurt (too much compression)
- Raw logits helped
- `sqrt` helped a lot
- `log1p` is a **slightly better calibration point** for this data

Conclusion:
> Sub-linear input scaling is one of the highest-impact changes explored so far, and  
> `log1p(x)` currently appears to be the best calibration choice.

#### Important detail: where the transform is applied matters
A subtle but critical finding is that **`log1p` must be applied during preprocessing**, not inside the model’s `forward()` method.

Observed behavior:
- Applying `log1p` **before training** (as a fixed preprocessing step) yields strong NDCG gains.
- Applying the same `log1p` **inside the model forward pass** results in worse test NDCG, even though training loss is similar.

Interpretation:
- When the transform lives inside the model, gradients flow through it.
- BCE then adapts weights and biases to partially *undo* the beneficial compression.
- This mirrors the failure mode seen with learned power scaling.

Key lesson:
> Calibration transforms that help ranking must be **fixed and external to the model**.  
> Even a fixed transform loses effectiveness if it is optimized through BCE gradients.

As a result:
- `log1p` is treated as **input preprocessing**, not a model component.
- The model operates on already-calibrated inputs and cannot learn around the transform.

#### Negative result: learned global power scaling
An experiment was run where the sub-linear power exponent `α` was made **learnable** and trained jointly with the ensemble using `BCEWithLogitsLoss`.

Observed outcome:
- Training loss behaved normally
- **Test NDCG degraded**, reverting toward pre-scaling performance
- The benefit of fixed `sqrt` / `log1p` scaling disappeared

Interpretation:
- BCE optimizes probability calibration, not ranking
- When `α` is learnable, the optimizer pushes it back toward `α ≈ 1`
- This effectively **undoes the beneficial compression** needed for good NDCG

Key lesson:
> Calibration transforms that improve ranking must be **fixed or externally tuned**.  
> Learning them directly with BCE causes the optimizer to remove their ranking benefit.

As a result:
- Fixed transforms (`sqrt`, `log1p`, fixed power laws) are preferred
- Learned calibration parameters coupled to BCE are avoided

---

## ❌ What has not worked (and was reverted)

### 1. Trainable cross-label interactions (low-rank label residuals)

#### 1a. Jointly trained residuals (catastrophic failure)

**Model:** `PerLabelWeightedConvResidualEnsemble`  
**Mechanism:** Low-rank residual on logits  
```
logits = logits + (logits @ V) @ Uᵀ
```

Observed results:
- **Catastrophic collapse of NDCG**
  - Test NDCG@10 ≈ 0.09–0.12
  - Test NDCG@1000 ≈ 0.22–0.28
- Far worse than any base model or simple mean
- Training loss remained small and stable

Interpretation:
- `BCEWithLogitsLoss` exploits the trainable cross-label path to optimize
  probability calibration, not ranking
- The residual destroys relative score geometry across labels
- This is the same failure mode observed with:
  - learned power scaling
  - sigmoid before NDCG
  - class imbalance weighting
- Even very low-rank (≪ L²) label interactions are unsafe when trained jointly
  with BCE on logits

Key lesson:
> **Any trainable cross-label interaction inside the BCE training loop is a dead end
> for this architecture.**

---

#### 1b. Post-hoc residuals trained on frozen base model (no collapse, still worse)

**Model:** `torch_per_label_conv_posthoc_residual`  
**Setup:**
1. Train `PerLabelWeightedConvEnsemble` normally
2. Stop at early peak (epoch ≈ 3)
3. Freeze base model
4. Train a low-rank residual on frozen logits to predict errors  
   ```
   target = y − sigmoid(logits_frozen)
   ```
5. Apply residual additively at inference time only

Observed results:
- ✅ **No catastrophic collapse**
- ✅ Ranking geometry preserved
- ❌ **Consistent drop in NDCG vs frozen base model**
  - Test NDCG@10 ≈ 0.685 (vs ≈ 0.715 best base)
  - Test NDCG@1000 also reduced
- Residual training converges almost immediately with very small MSE

Interpretation:
- Two-stage training successfully prevents BCE from hijacking ranking
- However, the residual still learns **probability smoothing**, not ranking corrections
- Cross-label adjustments systematically:
  - reduce score sharpness
  - smooth logits toward marginal frequencies
- This hurts top‑k ordering even when applied weakly and post-hoc

Key takeaway:
> **Even carefully constrained, post-hoc learned cross-label corrections do not
> improve NDCG once a well-calibrated per-label ensemble is in place.**

Overall implication:
- Label co-occurrence information is already implicitly encoded by:
  - base models
  - per-label biases
  - fixed sub-linear input calibration
- Explicitly modeling cross-label correlations (jointly or post-hoc) consistently
  shifts the model toward better calibration but worse ranking

Conclusion:
> Cross-label learning is not just risky — it is *unnecessary* for this setup.
> The limiting factor is ranking sharpness, not missing label interactions.

---

### 2. Pairwise / ranking-based losses
Tried variants:
- Naive pairwise loss
- Hard-negative mining pairwise loss

Results:
- Training loss decreases
- **Test NDCG gets worse**
- Especially harmful for NDCG@1000

Conclusion:
- Ranking losses do not match this architecture
- BCE provides a stronger and more stable signal for this problem

---

### 2. Per-label `pos_weight` (class imbalance weighting)
**Loss:** `BCEWithLogitsLoss(pos_weight=…)`

Observed effects:
- Inflates rare-label scores
- Pollutes top‑k rankings
- Degrades both NDCG@10 and NDCG@1000

Conclusion:
- Class imbalance correction hurts ranking here
- Bias term + base models already encode frequency information

---

### 3. Softmax constraints on per-label weights
Enforcing:
```
sum_m w[m, l] = 1,  w ≥ 0
```

Effects:
- Reduced model expressiveness
- Worse NDCG, especially @1000

Conclusion:
- The ensemble benefits from **unconstrained linear weights**
- Negative or amplified weights are useful

---

### 4. Sigmoid before NDCG evaluation
Applying:
```python
scores = sigmoid(logits)
```

Effects:
- Compresses score differences
- Hurts top‑k separation

Conclusion:
- Always evaluate NDCG on **raw logits**

---

## ✅ Current best configuration (recommended)

- **Model:** `PerLabelWeightedEnsemble` or `torch_per_label_freq_gate` (depending on metric priority)
- **Loss:** `BCEWithLogitsLoss()` (unweighted)
- **Outputs:** raw logits
- **Training:** ~2–4 epochs
- **Selection:** choose best epoch by train/validation NDCG (avoid test leakage)

Notes:
- `torch_per_label` remains best on **F1@5** in the current scoreboard.
- `torch_per_label_freq_gate` is best on **NDCG@10** and **NDCG@1000** in the current scoreboard.

---

## 🔍 Promising next steps (not yet tried)

Low-risk ideas that may still help:

1. **Constrain frequency residuals to be trust redistribution**
   - Force `delta_w` to be zero-mean across models per label
   - This would separate “shrinkage/calibration” effects from “source selection” effects

2. **Inference-time temperature scaling** on logits
3. **Checkpointing + epoch selection** (formalize early stopping)
4. **Input normalization across base models** (train statistics only)
5. **Inspect learned per-label weights and biases** for diagnostics

---

## 🚫 Ideas deprioritized based on evidence

- More ranking losses
- Deeper / nonlinear ensemble models
- Class rebalancing losses
- Additional weight constraints
- **Over-stabilizing the optimizer** (e.g. separate learning rates for weights/bias,
  zero weight decay, or otherwise smoothing early training dynamics)

### Why optimizer “improvements” failed

A targeted experiment replaced the default `AdamW(lr=1e-3, weight_decay=0.001)`
with a seemingly better-behaved configuration:
- `Adam` instead of `AdamW`
- Separate learning rates for weights and bias
- No weight decay

Observed outcome:
- Training loss decreased faster and more smoothly
- Per-model weights stayed almost perfectly uniform
- **Test NDCG consistently degraded**

Interpretation:
- The best-performing ensembles rely on **early, slightly unstable optimization**
- Mild noise and coupling between weight and bias updates in the first 2–3 epochs
  help create sharp per-label ranking geometry
- Making optimization “cleaner” and more decoupled suppresses this effect,
  leading to mean-like behavior and worse NDCG

Key lesson:
> For this problem, *imperfect* optimization is a feature, not a bug.
> Optimizer tuning that improves BCE convergence often harms ranking quality.

These have repeatedly reduced NDCG in practice.

---

## Bottom line

> A simple, expressive per-label linear ensemble trained with unweighted BCE
> and evaluated on raw logits is the right solution for this repository.

Most remaining gains will come from **careful selection and calibration**, not added complexity.
