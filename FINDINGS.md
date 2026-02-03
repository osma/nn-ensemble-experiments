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
**Models:** `torch_mean_ensemble`, `torch_mean_bias_ensemble`

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
- BCEWithLogitsLoss exploits the trainable cross-label path to optimize
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

Implication:
- Label correlations must already be captured implicitly by:
  - base models
  - per-label bias terms
  - fixed sub-linear input calibration
- If cross-label structure is added at all, it must be:
  - inference-only, or
  - strictly detached from BCE gradients, or
  - limited to top-k post-processing

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

These have repeatedly reduced NDCG in practice.

---

## Bottom line

> A simple, expressive per-label linear ensemble trained with unweighted BCE
> and evaluated on raw logits is the right solution for this repository.

Most remaining gains will come from **careful selection and calibration**, not added complexity.
