# nn-ensemble-experiments

Experiments for improving the **Annif neural‑network ensemble backend** using
simple, reproducible ensemble models evaluated with ranking‑oriented metrics.

The focus of this repository is **ranking quality** (NDCG@k, F1@k), not
probability calibration. Models are intentionally kept simple so that empirical
effects are easy to isolate and reason about.

---

## What this repository is

This repository contains:
- Baseline Annif model outputs
- Simple non‑torch ensemble baselines
- Torch‑based ensemble models trained end‑to‑end
- Controlled experiments on preprocessing, losses, and architectures
- A fully reproducible benchmark scoreboard

The primary goal is to understand **what actually improves multilabel ranking**
for large label spaces — and what reliably makes things worse.

---

## Model summary

| Model name | Type | Description | Source |
|-----------|------|-------------|--------|
| `bonsai` | Baseline | Annif Bonsai backend predictions | data only |
| `fasttext` | Baseline | Annif fastText backend predictions | data only |
| `mllm` | Baseline | Annif mLLM backend predictions | data only |
| `nn` | Baseline | Annif NN backend predictions | data only |
| `mean` | Non‑torch | Simple arithmetic mean ensemble | [benchmarks/mean.py](benchmarks/mean.py) |
| `mean_weighted` | Non‑torch | Grid‑searched weighted mean ensemble | [benchmarks/mean_weighted.py](benchmarks/mean_weighted.py) |
| `torch_mean` | Torch | Fixed mean via Conv1d, trained with BCE | [benchmarks/torch_mean.py](benchmarks/torch_mean.py) |
| `torch_mean_bias` | Torch | Mean ensemble with per‑label bias | [benchmarks/torch_mean_bias.py](benchmarks/torch_mean_bias.py) |
| `torch_per_label` | Torch | Per‑label linear ensemble with bias (best overall) | [benchmarks/torch_per_label.py](benchmarks/torch_per_label.py) |
| `torch_per_label_conv` | Torch | Conv1d‑based summation variant | [benchmarks/torch_per_label_conv.py](benchmarks/torch_per_label_conv.py) |
| `torch_per_label_conv_epochXX_YY` | Torch (inference) | Epoch‑level logit ensembling | [benchmarks/epoch_ensemble.py](benchmarks/epoch_ensemble.py) |

---

## Benchmarks

All benchmark results are stored in a single, auto‑generated scoreboard:

➡ **[SCOREBOARD.md](SCOREBOARD.md)**

It contains:
- Train/test NDCG@10 and NDCG@1000
- Test F1@5
- Top‑10 rankings by each metric

To regenerate everything from scratch:

```bash
./regenerate_scoreboard.sh
```

---

## Findings and conclusions

Detailed experimental conclusions are documented in:

➡ **[FINDINGS.md](FINDINGS.md)**

Key takeaways:
- Simple per‑label linear ensembles outperform more complex architectures
- Early epochs (≈ 2–4) consistently give the best ranking
- Fixed **log1p input preprocessing** has a large positive impact
- Learned calibration, cross‑label interactions, and ranking losses hurt NDCG

---

## Naming conventions

Model and file naming is strictly standardized.

➡ **[NAMING.md](NAMING.md)** (authoritative)

Highlights:
- Torch base models must start with `torch_`
- Base models must not include `_ensemble`
- Epoch and post‑hoc variants encode their relationship explicitly

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

If you are unsure which CUDA version you need, start with `torch-cpu` (it will
work everywhere, just slower), or match the extra to your installed NVIDIA
driver/CUDA runtime.

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

---

## Bottom line

For this problem setting:

> **Simple, well‑calibrated linear ensembles beat complex models.**

Most improvements come from preprocessing and careful evaluation — not added
architectural complexity.
