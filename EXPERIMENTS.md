# Experiments and Findings

This document centralizes the findings, analysis, and detailed experiment logs for the various model architectures tried in this repository. The primary goal is to understand what actually improves multilabel ranking for large label spaces — and what reliably makes things worse.

---

## High-Level Takeaways

- **No single architecture dominates all datasets**, but a few families are consistently strong: `torch_per_label*` for logits-based per-label linear models and `torch_mean*` for log1p-preprocessed mean-like models.
- **Strong, simple baselines are hard to beat.** Small, controlled extra capacity with a bias toward "do no harm" is the most successful pattern.
- **Logits + BCEWithLogitsLoss is a strong default.** It avoids saturation issues and provides smoother optimization for ranking.
- **Preprocessing and initialization are critical.** Keeping preprocessing outside the model and using dataset-specific initialization reduces optimization burden and stabilizes results.
- **Extra capacity interacts with class imbalance.** Without strong constraints, high-capacity models (like MLPs) can quickly learn to suppress all scores to satisfy the sparse objective, destroying ranking.

---

## What Works (Reliably helpful patterns)

### 1. Simple Linear/Mean Ensembles
- **Per-label linear ensemble on logits** (`torch_per_label`) is a top performer overall.
- **Mean-like ensemble with log1p preprocessing** (`torch_mean`) is best for deep ranking (NDCG@1000).
- **Global softmax scaling** (`torch_per_label_softmax_global`) stabilizes the mixture and improves generalization.

### 2. Principled Regularization
- **Shrink-to-prior**: L1/L2 penalties on **deltas from initialization** (rather than raw weights) are very effective.
- **Frequency-weighted shrinkage**: Shrinking rare labels harder while allowing flexibility for common labels (as seen in `torch_per_label_l1_delta`).

### 3. Controlled Extensions
- **Global weights + per-label residuals + bias + explicit L2** is a robust template for extensions.
- **Softmax global weights + Bias decomposition** (`bias_global + bias_delta`) is a powerful combination for cross-dataset robustness, particularly for `koko` and `yso-en`.
- **Low-rank structure** can capture useful shared label structure when trained with careful regularization.

---

## What Has Not Worked (or is consistently risky)

### 1. High-Capacity MLPs over Flattened Inputs
- Big MLPs over flattened (models × labels) inputs are unstable and consistently degrade ranking.
- **Failure Mode**: The MLP learns a global negative suppressor (ranking collapse) to satisfy the BCE objective on sparse negatives.

### 2. Probability-Space Training with Hard Clamping
- Outputting probabilities and using `BCELoss` with hard clamps can saturate gradients and make learned corrections "stick" at the bounds.

### 3. Unconstrained Cross-Label Mixing
- Cross-label mixing can help in isolated cases but is not a free win. It introduces many hyperparameters and can help one metric while hurting another.
- **Mitigation**: Requires strong safety rails like bounded gates, per-example centering, and explicit penalties.

---

## Detailed Experiment Logs

### 1. `torch_mean_residual` Family
*Original source: EXPERIMENTS.md*

This family focuses on global per-model weights shared across all labels, with label-specific deviations.

| Variant | Description | Finding |
|---------|-------------|---------|
| `baseline` | `global_w[m] + delta_w[m,l] + bias[l]` with L2. | Strong default; baseline for others. |
| `softmax_global` | Global weights constrained via softmax. | **Consistent improvement** across all datasets. Reduces harmful drift. |
| `globalxdelta` | Multiplicative label-specific deviations. | Mostly worse than additive. Sensitive to scale and optimization. |
| `tanh_clamp` | Bounded per-label residual weights via tanh. | Worse than baseline. May remove beneficial capacity. |
| `bias_per_model` | Added per-model global bias term. | Consistent regression. Adds redundant degrees of freedom. |
| `bias_residual` | Split bias into global scalar + per-label residual. | Effectively identical to baseline. |
| `l2_anchor_global` | Anchor `global_w` to dataset init weights via L2. | **Consistent improvement**. Stabilizes later-epoch training. |
| `softmax_global_l2_anchor` | Combines softmax and anchoring. | **Clear improvement** over baseline; tie with anchor-only. |
| `freq_weighted_delta` | L2 penalty on `delta_w` weighted by label frequency. | Near-tie with baseline. Not a strong stabilizer on its own. |

### 2. `torch_per_label` Family
*Original source: PER_LABEL.md and EXPERIMENTS2.md*

These models allow fully independent weights per label, often reparameterized for better generalization.

| Variant | Description | Finding |
|---------|-------------|---------|
| `global_plus_delta` | `w_eff[m,l] = w_global[m] + w_delta[m,l]`. | Neutral-to-negative. Needs explicit constraints on `w_delta`. |
| `softmax_global` | `w_eff[m,l] = softmax(g_raw)[m] + w_delta[m,l]`. | **Very competitive**. Beats baseline on `yso-en` and `koko`. |
| `global_times_scale` | Multiplicative label scaling around global weights. | Competitive but rarely beats the baseline. Too restrictive. |
| `bias_global_plus_delta` | Reparameterized bias into global + residual. | Neutral-to-slightly-positive. Low-risk change. |
| `softmax_global_scale` | `s * softmax(g) + w_delta`. | **Top overall** by Avg Test Metrics. Recovers scale freedom. |
| `softmax_global_l2_anchor` | Softmax + L2 anchor to init weights. | Neutral on `yso-fi`, worse on `yso-en`. Tie on `koko`. |
| `elastic_anchor` (v1) | Softmax + Elastic Net (L1+L2) + L2 Anchor. | **Regression**. Over-regularized; `w_delta` frozen near zero. |
| `elastic_anchor` (v2) | Softmax + L2 Delta + Bias Decomposition. | **Strong result (#4 Overall)**. #2 on Avg NDCG@1000. |
| `elastic_anchor` (v3) | Softmax + Bias Decomposition (no δ-L2). | **Improved #4 Overall (0.5338)**. Confirms soft constraints are better. |
| `apex` | Softmax + Full Per-label Deltas + Strong L2 (1e-2) + No Global Bias. | Ranked high on `yso-fi` but regressed on `yso-en` and `koko`. Overall weighted average is 0.5315 (not in top 10). |

> **Note on Elastic Anchor**: v1 failed because combining L1, L2, and a strong L2 anchor to initialization restricted per-label adaptation too much. v3 proved that removing explicit L2 on the weight residuals (matching the champion's settings) and using only a light bias shrinkage allows for the best performance while maintaining stability via the softmax global weights.

### 3. `torch_per_label_mlp` Family
*Original source: PER_LABEL_MLP.md*

Two-stage and end-to-end models: base logits are corrected by a residual MLP.

| Variant | Description | Finding |
|---------|-------------|---------|
| `softmax_global_active_lowrank` | End-to-end: Softmax global base + Active-label Low-Rank Factorization (rank 64). Multiplicative reweighting, tanh bounds, centering. | **#1 Overall Model**. Successfully captures cross-label correlation via an efficient bottleneck. Outperforms the base and the MLP variant across datasets. |
| `softmax_global_active_mlp` | End-to-end: Softmax global base + Active-label MLP. Uses multiplicative reweighting, tanh bounds, and centering. | **Strong result (#3 Overall)**. Safely integrates cross-label correlation without ranking collapse. Ties baseline model on Weighted Avg. |
| `baseline` | Multiplicative: `base * (1 + gate * delta)`. | Safe but often a no-op on `yso-*`. |
| `additive_delta` | `base + gate * delta`. | Regression on `yso-en` and `koko`. Perturbs low-confidence labels. |
| `gate_per_label` | Per-active-label gate vector. | Regression on `koko`. Too many degrees of freedom. |
| `gate_per_sample` | Data-dependent gate via pooling features. | Safe but effectively constant in practice. |
| `layernorm_feats` | LayerNorm over flattened features before MLP. | Regression. LN amplifies small widespread signals. |
| `rank_bottleneck` | Low-rank bottleneck (rank=32) in MLP output. | No gain. `koko` still regresses. |
| `remove_centering` | Remove per-sample centering on `delta`. | **Severe regression**. Learns degenerate global shift. |
| `no_delta` (S1) | Remove `w_delta` and rely only on low-rank mixer. | **Major Regression**. `w_delta` is critical; low-rank mixer at current settings does not learn fast enough to compensate. |
| `fixed_gate` (S2) | Replace learnable gate with fixed constant 0.02. | **Negligible regression** (-0.0002). Confirms learning the gate is unnecessary for the #1 model. |
| `rank16` (S3) | Reduce rank from 64 to 16 in low-rank mixer. | **Minimal regression** (-0.0005). Shows 4× reduction in rank capacity retains ~99.9% of performance. |
| `no_base_ch` (S4) | Remove `base_logits` channel from low-rank mixer. | **Minimal regression** (-0.0005). Confirms base logits are mostly redundant when raw model inputs are present, though they provide a small performance boost on `yso-*`. |
| `single_lr` (S5) | Use single optimizer group with LR=1e-3 for all params. | **Regression** (Avg 0.5350 → 0.5327). Two-tiered learning rate (3e-3 for base, 1e-4 for low-rank) is superior for balancing convergence. |
| `no_centering` (S6) | Remove per-sample centering from low-rank delta. | **Improvement (#1 Overall: 0.535290)**. Unlike the MLP variant which failed without centering, the low-rank mixer is stable and actually benefits slightly from the removal. |
| `additive` (S7) | Replace multiplicative with additive delta application. | **Slight regression** (Avg 0.5350 → 0.5347). Multiplicative scaling, which biases adjustments toward high-confidence labels, remains slightly superior to additive application for the low-rank mixer. |
| `symmetric` (S8) | Tie U and V (symmetric factorization) in low-rank mixer. | **Minor regression** (Avg 0.5350 → 0.5348). Halving the label-projection parameters via symmetry provides a strong structural prior but slightly restricts the flexibility of the cross-label coupling. |
| `no_clamp` (S9) | Remove tanh clamp from low-rank delta. | **Minor regression** (Avg 0.5350 → 0.5341). Removing the tanh clamp allows for theoretically larger deltas but did not yield performance gains, suggesting the clamp serves as a useful (if rarely hit) safety rail. |
| `combined` (S11) | Combines S2, S3, S4, S6, and S8 (Fixed gate, Rank 16, No base channel, No centering, Symmetric). | **Strong simplification**. Achieves 99.88% of champion performance (0.5344) while removing ~8× parameters and 2 hyperparameters. Slightly worse than the base model (0.5349) on average, suggesting that while individual simplifications held up, their combination marginally over-constrains the mixer. |

---

## Methodology and Hypotheses

### Early Stopping Policy
All torch scripts use **train subset NDCG@1000** for early stopping to avoid test leakage. Standardizing on computing test metrics only when saving a new best snapshot is preferred.

### Preprocessing
Consistent use of `log1p(clamp(x,0))` for logits-based models. Keeping this outside the model prevents the optimizer from undoing calibration benefits.

### Hypotheses
1. **Preservation of Ordering**: Ranking improves when the model preserves relative ordering rather than overfitting calibration.
2. **Label Independence**: Per-label independence is a strong prior because label spaces are huge and sparse.
3. **Logit Space Advantage**: Training in logit space avoids saturation and provides smoother gradients.
4. **Imbalance vs. Capacity**: Class imbalance interacts with capacity; broad shifts are rewarded by BCE but harm ranking.
5. **Base Weights vs. Correction**: Simple base weights (like `w_delta`) provide essential label-specific signals that complex corrections (like low-rank mixers) struggle to recover if trained from scratch with low learning rates.

---

## Future Directions

1. **Prefer Low-Rank Coupling**: Move away from flattened MLPs. Use low-rank structure on top of strong linear priors.
2. **Safety Rails for MLP**: If using MLPs, keep bounded gates, weight decay, and centering.
3. **Refine Imbalance Handling**: Revisit `pos_weight` with gentler caps or apply only to active labels.
4. **Sparsity Priors**: Explore stronger sparsity-inducing penalties (like L1 on deltas) for datasets like `koko`.
5. **Truth-Active Only**: Restrict stage-2 corrections to labels seen in truth to reduce unintended long-tail perturbations.
