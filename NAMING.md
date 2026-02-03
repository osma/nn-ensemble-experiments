# Naming Conventions

This document defines **authoritative naming conventions** for models, scripts,
and scoreboard entries in this repository.  
Its goal is to eliminate ambiguity between **Python filenames**, **model
identifiers**, and **SCOREBOARD.md entries**.

These rules are **normative**: new code and refactors must follow them.

---

## 1. General principles

1. **One name per concept**
   - A model should have **one canonical name**
   - That name should be reflected consistently in:
     - Python module filename
     - `update_markdown_scoreboard(..., model=...)`
     - Documentation references

2. **Torch models are always explicit**
   - Any model trained with PyTorch **must** start with the prefix:
     ```
     torch_
     ```

3. **Avoid overloaded terminology**
   - The word `ensemble` is reserved for *true combinations* of:
     - multiple checkpoints, or
     - multiple independently trained models
   - Base models must not use `_ensemble` in their names

---

## 2. Torch base ensemble models (most important rule)

### ✅ Definition
A **torch base ensemble** is a single PyTorch model trained end‑to‑end that
combines multiple base predictors (e.g. bonsai, fastText, mLLM).

### ✅ Required naming rules
Torch base ensembles:

- ✅ **Must start with** `torch_`
- ✅ **Must NOT include** `_ensemble`
- ✅ Should use concise, descriptive suffixes

### ✅ Examples (correct)

| Model purpose | Filename | Scoreboard name |
|--------------|----------|-----------------|
| Mean ensemble | `torch_mean.py` | `torch_mean` |
| Mean + bias | `torch_mean_bias.py` | `torch_mean_bias` |
| Per‑label linear | `torch_per_label.py` | `torch_per_label` |
| Per‑label conv | `torch_per_label_conv.py` | `torch_per_label_conv` |

### ❌ Examples (incorrect)

- `torch_mean_ensemble`
- `torch_per_label_weighted_ensemble`
- `torch_per_label_conv_ensemble`

---

## 3. Epoch / checkpoint ensembles

### ✅ Definition
Models that **combine outputs from multiple training epochs** of the *same*
base model.

### ✅ Naming rules

- ✅ Must start with the base model name
- ✅ May encode epochs explicitly
- ✅ Must NOT include `_ensemble`

### ✅ Example

```
torch_per_label_conv_epoch02_03
```

This clearly indicates:
- base model: `torch_per_label_conv`
- epochs used: 02 and 03
- method: simple logit combination

---

## 4. Post‑hoc or auxiliary models

### ✅ Definition
Models trained **after** a base model is frozen, typically to adjust or refine
logits (e.g. residuals).

### ✅ Naming rules

- ✅ Start with the base model name
- ✅ Append a clear, descriptive suffix
- ✅ Do not use `_ensemble`

### ✅ Example

```
torch_per_label_conv_posthoc_residual
```

---

## 5. Non‑torch models

Non‑torch baselines (e.g. Annif backends) **do not** use the `torch_` prefix.

### ✅ Examples

- `bonsai`
- `fasttext`
- `mllm`
- `nn`
- `mean`
- `mean_weighted`

These names must match:
- data filenames where applicable
- scoreboard entries

---

## 6. Mapping checklist (use before committing)

Before adding or renaming a model, verify:

- [ ] Python filename matches the intended model name
- [ ] `model=` passed to `update_markdown_scoreboard` matches filename
- [ ] Torch models start with `torch_`
- [ ] No base model includes `_ensemble`
- [ ] SCOREBOARD.md entries match exactly

---

## 7. Rationale (why this matters)

Clear naming prevents:
- accidental duplication of results
- misleading comparisons
- confusion between *architectural ensembles* and *checkpoint ensembles*

Consistent names make:
- scoreboard diffs meaningful
- experimental conclusions reproducible
- future refactors safe

---

## 8. Short version (TL;DR)

- **Torch base model?** → `torch_*`, no `_ensemble`
- **Multiple epochs combined?** → encode epochs, no `_ensemble`
- **Post‑hoc add‑on?** → descriptive suffix, no `_ensemble`
- **If in doubt:** prefer clarity over brevity

This document is the source of truth.
