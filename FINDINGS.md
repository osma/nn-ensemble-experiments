# Ensemble Experiments – Findings (multi-dataset)

This document summarizes what currently **works** and what **doesn't** in this repository,
based on the multi-dataset results in [`SCOREBOARD.md`](SCOREBOARD.md).

Focus:
- Primary evaluation is **average across datasets** (yso-fi, yso-en, koko).
- Notes call out **dataset-specific** differences when they matter.
- Baseline backends (bonsai/fasttext/mllm) are intentionally *not* discussed here.
  The only baseline referenced is **`nn`**, as a practical point of comparison.

---

## ✅ What works (reliably)

### 1) Per-label linear ensembles trained with unweighted BCE on raw logits
**Models:** `torch_per_label`, `torch_per_label_conv`  
**Loss:** `BCEWithLogitsLoss()` (unweighted)  
**Evaluation:** ranking metrics computed on **raw logits** (no sigmoid)

This remains the strongest *learned* family on average, and it is consistently competitive
with the best methods across datasets.

Practical notes:
- **Early stopping matters**. On yso-fi and koko the best epoch is typically early (≈2–5).
  yso-en can tolerate later epochs more often.
- `torch_per_label_conv` is usually close to `torch_per_label` but not consistently better.

### 2) Simple torch mean ensemble (trained, but effectively stable)
**Model:** `torch_mean`

On average across datasets, `torch_mean` is a strong and very stable option, and it
often lands near the top learned methods.

Dataset notes:
- On **yso-en**, any method that effectively downweights the weak component model tends to do well.
- On **koko**, improvements over the mean are smaller than on yso-fi (harder dataset, different components).

### 3) Fixed log1p preprocessing on inputs (outside the model)
Applying `log1p(clamp(x, min=0))` as a fixed preprocessing step (not a learnable part of
the model) remains one of the highest-impact “free wins” for learned ensembles.

---

## ⚠️ Mixed / fragile

### Frequency-aware variants
**Models:** `torch_per_label_freq_gate`, `torch_per_label_freq_reg`

These can be competitive on some datasets/runs, but they are currently **not robust**
as a “default best” across datasets.

What we see now:
- With the current defaults (`lambda_delta=0.001` in `torch_per_label_freq_gate`),
  the frequency correction can become **too large**, and early stopping often picks
  **epoch 1** (a sign the model is destabilizing quickly).
- `torch_per_label_freq_reg` can be competitive, but its best configs tend to require
  tuning and it is not clearly superior on average.

Dataset notes:
- **yso-fi**: frequency methods can be competitive, but are sensitive to regularization strength.
- **yso-en**: results are highly sensitive because one component behaves very differently; frequency heuristics
  can easily overfit to this.
- **koko**: frequency-gated behavior varies and has not consistently improved over `torch_per_label`.

---

## ❌ What doesn’t work (and why)

### 1) Cross-label interaction / residual layers (trained with BCE)
Any trainable mechanism that mixes information *between labels* (even low-rank) tends to:
- improve calibration-like objectives while
- **destroying ranking geometry**, causing large NDCG drops.

Rule of thumb:
> If gradients can change a label’s score based on *other labels’ scores*, BCE will
> often learn “smoothing” that harms top-k ranking.

### 2) Ranking losses / pairwise losses (in this repo’s setup)
Pairwise / ranking-style objectives tried so far have not improved NDCG, and often
hurt NDCG@1000 in particular.

### 3) Class-imbalance weighting in BCE (`pos_weight`)
Per-label `pos_weight` pushes rare labels up, which tends to pollute top-k predictions
and reduce NDCG.

### 4) Constraints on per-label weights (e.g. softmax / simplex)
Forcing per-label weights to be nonnegative and sum to 1 reduces expressiveness and
typically reduces NDCG.

### 5) Sigmoid before computing NDCG
Always compute NDCG on **raw logits**. Sigmoid compresses score differences and harms ranking.

---

## Practical recommendation (today)

If you want a strong default that behaves well across datasets:

1. Start with `torch_per_label` (early-stop by train subset NDCG@1000).
2. Compare against `torch_mean` as a very stable baseline.
3. Treat frequency-aware methods as “research variants” requiring tuning and careful validation.

For a baseline comparison point (not a target): `nn` is consistently competitive in places,
but learned ensembles still beat it on some datasets/metrics; the gap is dataset-dependent.

---

## Dataset-specific notes (high-level)

- **yso-fi**: learned per-label ensembles are strongest and benefit from early stopping.
- **yso-en**: component behavior differs strongly from yso-fi; methods that implicitly reduce reliance on the weak component do best.
- **koko**: different components and label space; gains from learning are smaller and training is slower; early stopping is important.

---

## Regenerating results

```bash
./regenerate_scoreboard.sh
```
