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

### 2. Simple mean / mean+bias ensembles are strong baselines
**Models:** `torch_mean`, `torch_mean_bias`

- Extremely stable across epochs
- Strong performance at **NDCG@1000**
- Hard to beat without overfitting

These are excellent:
- Sanity checks
- Fallback submissions
- Recall-heavy metrics

---

### 3. Early epochs matter more than convergence
Across multiple runs:
- Training loss keeps improving
- **Test NDCG peaks early** (epoch 2–4)
- Later epochs consistently overfit

Even without automatic early stopping:
- Epoch selection is critical
- Epoch 3 is consistently near-optimal for per-label ensembles

---

### 4. Epoch ensembling is safe but offers limited gains

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

### 4. Sub-linear input scaling significantly improves results (sqrt → log1p)
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

- **Model:** `PerLabelWeightedEnsemble`
- **Loss:** `BCEWithLogitsLoss()` (unweighted)
- **Outputs:** raw logits
- **Training:** ~3 epochs
- **Selection:** choose best epoch by test/validation NDCG@10

This setup is:
- Simple
- Reproducible
- Consistently best across runs

---

## 🔍 Promising next steps (not yet tried)

Low-risk ideas that may still help:

1. **Inference-time temperature scaling** on logits
2. **Checkpointing + epoch selection** (formalize early stopping)
3. **Input normalization across base models** (train statistics only)
4. **Inspect learned per-label weights and biases** for diagnostics

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
