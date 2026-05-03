# nn-ensemble-experiments

Experiments for improving the **Annif neural‑network ensemble backend** using
simple, reproducible ensemble models evaluated with ranking‑oriented metrics.

The focus of this repository is **ranking quality** (NDCG@k, F1@k), not
probability calibration. Models are intentionally kept simple so that empirical
effects are easy to isolate and reason about.

---

## What this repository is

This repository contains:
- Baseline Annif model outputs (stored as sparse NPZ matrices)
- Simple non‑torch ensemble baselines
- Torch‑based ensemble models trained end‑to‑end (CPU or CUDA)
- Controlled experiments on preprocessing, losses, and architectures
- A fully reproducible benchmark scoreboard (`SCOREBOARD.md`)

The primary goal is to understand **what actually improves multilabel ranking**
for large label spaces — and what reliably makes things worse.

---

## Documentation

- **[SCOREBOARD.md](SCOREBOARD.md)**: The single source of truth for all model performance metrics and rankings.
- **[EXPERIMENTS.md](EXPERIMENTS.md)**: Detailed history of all architectures tried, findings, and analysis of what worked vs. what didn't.

---

## Model summary (current)

This table lists the **intended kept set** of benchmark scripts/models. If you removed
additional scripts locally, update this table to match the remaining files.

| Model name | Type | Description | Source |
|-----------|------|-------------|--------|
| `bonsai` | Baseline | Annif Bonsai backend predictions (yso-*) | data only |
| `fasttext` | Baseline | Annif fastText backend predictions (yso-*) | data only |
| `mllm` | Baseline | Annif mLLM backend predictions (yso-* and koko) | data only |
| `nn` | Baseline | Annif NN backend predictions (yso-* and koko test only) | data only |
| `bonsai_gemma3` | Baseline | Annif Bonsai (Gemma3) backend predictions (koko) | data only |
| `bonsai_ovis2` | Baseline | Annif Bonsai (Ovis2) backend predictions (koko) | data only |
| `baseline` | Non‑torch | Evaluate each individual base model and update scoreboard | [benchmarks/baseline.py](benchmarks/baseline.py) |
| `mean` | Non‑torch | Simple arithmetic mean ensemble | [benchmarks/mean.py](benchmarks/mean.py) |
| `mean_weighted` | Non‑torch | Grid‑searched weighted mean ensemble (select by train NDCG@1000) | [benchmarks/mean_weighted.py](benchmarks/mean_weighted.py) |
| `torch_mean` | Torch | Learned 1×1 Conv1d over base models (probabilities), BCE loss, fixed log1p preprocessing | [benchmarks/torch_mean.py](benchmarks/torch_mean.py) |
| `torch_mean_residual` | Torch | Global per-model weights + per-label residual weights + bias (logits), BCEWithLogitsLoss, explicit L2 penalties; early stopping on train subset NDCG@1000 | [benchmarks/torch_mean_residual.py](benchmarks/torch_mean_residual.py) |
| `torch_mean_residual_globalxdelta` | Torch (experimental) | `torch_mean_residual` variant with multiplicative per-label residuals: `w_eff[m,l]=global_w[m]*(1+delta_w[m,l])` | [benchmarks/torch_mean_residual_globalxdelta.py](benchmarks/torch_mean_residual_globalxdelta.py) |
| `torch_mean_residual_delta_tanh_clamp` | Torch (experimental) | `torch_mean_residual` variant with bounded residuals: `delta_w = delta_max * tanh(delta_raw)`; `w_eff[m,l]=global_w[m]+delta_w[m,l]` | [benchmarks/torch_mean_residual_delta_tanh_clamp.py](benchmarks/torch_mean_residual_delta_tanh_clamp.py) |
| `torch_mean_residual_softmax_global_l2_anchor` | Torch (experimental) | `torch_mean_residual` variant combining softmax global weights + L2 anchor to dataset init weights | [benchmarks/torch_mean_residual_softmax_global_l2_anchor.py](benchmarks/torch_mean_residual_softmax_global_l2_anchor.py) |
| `torch_nn_simple` | Torch | Learned 1×1 Conv1d over base models (probabilities), BCE loss, fixed log1p preprocessing | [benchmarks/torch_nn_simple.py](benchmarks/torch_nn_simple.py) |
| `torch_nn_split` | Torch | Learned 1×1 Conv1d over base models (probabilities), BCE loss, fixed log1p preprocessing | [benchmarks/torch_nn_split.py](benchmarks/torch_nn_split.py) |
| `torch_nn_split_per_label` | Torch | Learned 1×1 Conv1d over base models (probabilities), BCE loss, fixed log1p preprocessing | [benchmarks/torch_nn_split_per_label.py](benchmarks/torch_nn_split_per_label.py) |
| `torch_per_label` | Torch | Per‑label linear ensemble (logits) + bias, BCEWithLogitsLoss, fixed log1p preprocessing; early stopping on train subset NDCG@1000; writes diagnostics JSON | [benchmarks/torch_per_label.py](benchmarks/torch_per_label.py) |
| `torch_per_label_global_plus_delta` | Torch (experimental) | `torch_per_label` variant with weights reparameterized as `w_eff[m,l]=w_global[m]+w_delta[m,l]` | [benchmarks/torch_per_label_global_plus_delta.py](benchmarks/torch_per_label_global_plus_delta.py) |
| `torch_per_label_l1_delta` | Torch | `torch_per_label` + L1 regularization on mean(|W − W0|) (logits) | [benchmarks/torch_per_label_l1_delta.py](benchmarks/torch_per_label_l1_delta.py) |
| `torch_per_label_residual_lowrank_mix_active` | Torch (experimental) | Two-stage: `torch_per_label` base (frozen) + active-label low-rank cross-label residual in logit space (rank=32) | [benchmarks/torch_per_label_residual_lowrank_mix_active.py](benchmarks/torch_per_label_residual_lowrank_mix_active.py) |
| `torch_per_label_mlp` | Torch (experimental) | Two-stage: `torch_per_label` base (frozen) + active-label dense residual MLP in logit space (cross-label adjustments) | [benchmarks/torch_per_label_mlp.py](benchmarks/torch_per_label_mlp.py) |
| `torch_per_label_mlp_additive_delta` | Torch (experimental) | Variant of `torch_per_label_mlp` where stage-2 residual is applied as `base_logits + gate * delta` (additive) instead of multiplicative reweighting | [benchmarks/torch_per_label_mlp_additive_delta.py](benchmarks/torch_per_label_mlp_additive_delta.py) |
| `torch_per_label_mlp_layernorm_feats` | Torch (experimental) | Variant of `torch_per_label_mlp` adding `LayerNorm` over flattened stage-2 features (per sample) before the residual MLP | [benchmarks/torch_per_label_mlp_layernorm_feats.py](benchmarks/torch_per_label_mlp_layernorm_feats.py) |
| `torch_lowrank_residual_epsclamp` | Torch (experimental) | Low-rank residual ensemble trained in probability space with eps clamp | [benchmarks/torch_lowrank_residual_epsclamp.py](benchmarks/torch_lowrank_residual_epsclamp.py) |
| `torch_lowrank_mix` | Torch (experimental) | Cross-label mixing only (A1), probability-space | [benchmarks/torch_lowrank_mix.py](benchmarks/torch_lowrank_mix.py) |
| `torch_lowrank_residual_mix_temp` | Torch (experimental) | Residual + mixing with learnable scaling | [benchmarks/torch_lowrank_residual_mix_temp.py](benchmarks/torch_lowrank_residual_mix_temp.py) |

---

## Benchmarks

To update the scoreboard:

```bash
./regenerate_scoreboard.sh
```

```bash
./regenerate_scoreboard.sh --models baseline,mean_weighted,torch_per_label,torch_per_label_residual_lowrank_mix_active
```

By default this runs a **small, fast** benchmark set and **updates incrementally**
(it does not delete `SCOREBOARD.md`).

Common workflows:

```bash
# Full clean rebuild
./regenerate_scoreboard.sh --clean

# Single dataset
./regenerate_scoreboard.sh --dataset yso-fi

# Only a subset of benchmarks (comma-separated module names under benchmarks.*)
./regenerate_scoreboard.sh --models baseline,mean_weighted,torch_per_label
```

---

## Development setup

This project uses **uv** for Python version and dependency management.

### Requirements
- `uv` installed: https://docs.astral.sh/uv/

### Setup (non-torch)
If you only want to run the non-torch baselines (e.g. `mean`, `mean_weighted`):

```bash
uv python install
uv sync
```

### Installing PyTorch (CPU vs CUDA)

PyTorch is an **optional dependency** in this repo. You must select exactly one
of the following extras depending on your environment:

- `torch-cpu`  → CPU-only wheels from `https://download.pytorch.org/whl/cpu`
- `torch-cu126` → CUDA 12.6 wheels from `https://download.pytorch.org/whl/cu126`
- `torch-cu130` → CUDA 13.0 wheels from `https://download.pytorch.org/whl/cu130`

Install one of them like this:

```bash
# CPU-only
uv sync --extra torch-cpu

# CUDA 12.6
uv sync --extra torch-cu126

# CUDA 13.0
uv sync --extra torch-cu130
```

To verify CUDA is working after installing a CUDA wheel:

```bash
uv run python benchmarks/cuda_smoketest.py
```

### Running tools
```bash
uv run python
uv run pytest
```

---

## License

This project is licensed under:

➡ **[LICENSE](LICENSE)**

---

## Important note on code generation

> **All code in this repository is AI‑generated**, with human guidance,
> review, and iterative experimentation.

The code should be treated as **research code**, not a production library.
Clarity, reproducibility, and empirical correctness are prioritized over API
stability.
