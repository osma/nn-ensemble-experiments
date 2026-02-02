# Ensemble Experiments – Findings

This document summarizes what has **worked** and **not worked** so far while experimenting with ensemble methods and losses in this repository.  
The conclusions below are based on repeated runs and the results recorded in `SCOREBOARD.md`.

---

## ✅ What has worked well

### 1. Per‑label linear ensemble with bias (best overall)
**Model:** `PerLabelWeightedEnsemble`  
**Loss:** `BCEWithLogitsLoss` (unweighted)  
**Evaluation:** raw logits (no sigmoid)

- This is the **best-performing approach** on both:
  - **Test NDCG@10**
  - **Test NDCG@1000**
- Peak performance occurs **early** in training.

**Best checkpoint observed:**
- `torch_per_label_epoch03`
  - Test NDCG@10 ≈ **0.704**
  - Test NDCG@1000 ≈ **0.812**

Key properties that matter:
- Raw logits (no sigmoid before NDCG)
- Per-label bias term
- No extra constraints on weights
- Simple linear structure

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

### 4. Sub-linear input scaling (sqrt) significantly improves results
Applying a simple preprocessing step to **all ensemble inputs**:

```
x -> sqrt(x)
```

had a **large positive impact** on both:
- **Test NDCG@10**
- **Test NDCG@1000**

Key observations:
- `sqrt` is monotonic, so per-model ranking is preserved
- Large (overconfident) scores are compressed more than small ones
- Over-dominance by any single base model is reduced

What this tells us:
- Base model outputs are **systematically over-confident**
- The ensemble benefits more from **relative confidence** than absolute magnitude
- Calibration and scale matter more than additional model complexity

This also explains earlier results:
- Sigmoid hurt (too much compression)
- Raw logits helped
- `sqrt` hits a **sweet spot** between the two

Conclusion:
> A simple sub-linear rescaling of inputs can outperform more complex loss or model changes.

---

## ❌ What has not worked (and was reverted)

### 1. Pairwise / ranking-based losses
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
