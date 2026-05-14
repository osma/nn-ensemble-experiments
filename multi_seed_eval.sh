#!/usr/bin/env bash
set -eu

# Multi-seed evaluation for model selection.
#
# Runs each specified model with N different random seeds across all datasets,
# collects the per-seed weighted-average scores, and produces a summary table
# with mean, std, min, max, and a paired-difference test.
#
# Usage:
#   ./multi_seed_eval.sh                          # defaults: 5 seeds, 2 finalist models
#   ./multi_seed_eval.sh --seeds 10               # run 10 seeds
#   ./multi_seed_eval.sh --models "torch_per_label_softmax_global"  # single model
#
# Output:
#   - SCOREBOARD.md gets rows for each seed run (model_name/seed=N)
#   - multi_seed_results.csv is created with per-seed metrics
#   - Summary statistics are printed to stdout

NUM_SEEDS=5
MODELS="torch_per_label_softmax_global torch_per_label_softmax_global_active_lowrank_no_centering"
DATASETS="yso-fi yso-en koko"
RESULTS_CSV="multi_seed_results.csv"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --seeds)
      NUM_SEEDS="$2"
      shift 2
      ;;
    --models)
      MODELS=$(printf "%s" "$2" | tr ',' ' ')
      shift 2
      ;;
    --datasets)
      DATASETS=$(printf "%s" "$2" | tr ',' ' ')
      shift 2
      ;;
    --output)
      RESULTS_CSV="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--seeds N] [--models m1,m2] [--datasets d1,d2] [--output file.csv]" >&2
      exit 2
      ;;
  esac
done

echo "========================================"
echo "Multi-Seed Model Evaluation"
echo "========================================"
echo "Models:   $MODELS"
echo "Datasets: $DATASETS"
echo "Seeds:    1 .. $NUM_SEEDS"
echo "Output:   $RESULTS_CSV"
echo ""

# Write CSV header
echo "model,dataset,seed,test_ndcg10,test_ndcg1000,test_f1_5,best_epoch" > "$RESULTS_CSV"

# Run each model × dataset × seed
for m in $MODELS; do
  for seed in $(seq 1 "$NUM_SEEDS"); do
    echo ""
    echo "===== $m | seed=$seed ====="

    # Run all datasets for this model+seed
    ./regenerate_scoreboard.sh --models "$m" --seed "$seed"

    # Extract metrics from SCOREBOARD.md for each dataset
    for ds in $DATASETS; do
      # Build the model_name as it appears in SCOREBOARD.md
      model_name="$(uv run python -c "from benchmarks.datasets import ensemble3_keys; e=ensemble3_keys('$ds'); print('$m(' + ','.join(e) + ')/seed=$seed')")"

      # Extract the row from SCOREBOARD.md
      uv run python - "$model_name" "$ds" "$RESULTS_CSV" "$m" "$seed" <<'PY'
import sys
from pathlib import Path

_, model_name, ds, csv_path, short_model, seed = sys.argv

lines = Path("SCOREBOARD.md").read_text().splitlines()
for line in lines:
    if not line.startswith("|"):
        continue
    cols = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cols) != 8:
        continue
    if cols[0] == model_name and cols[1] == ds:
        epoch = cols[2]
        test_ndcg10 = cols[5]
        test_ndcg1000 = cols[6]
        test_f1_5 = cols[7]
        with open(csv_path, "a") as f:
            f.write(f"{short_model},{ds},{seed},{test_ndcg10},{test_ndcg1000},{test_f1_5},{epoch}\n")
        print(f"  Extracted: {short_model} | {ds} | seed={seed} | ndcg@10={test_ndcg10} ndcg@1000={test_ndcg1000} f1@5={test_f1_5}")
        break
else:
    print(f"  WARNING: Could not find row for {model_name} / {ds}", file=sys.stderr)
PY
    done
  done
done

echo ""
echo "========================================"
echo "All seed runs complete. Analyzing..."
echo "========================================"

# Compute summary statistics and paired differences
uv run python - "$RESULTS_CSV" <<'ANALYSIS'
from __future__ import annotations
import csv
import sys
from collections import defaultdict
import math

csv_path = sys.argv[1]

# Parse CSV
rows: list[dict] = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

if not rows:
    print("No results found!")
    sys.exit(1)

# Compute weighted average per (model, seed)
# Weight: 0.4 * NDCG@10, 0.4 * F1@5, 0.2 * NDCG@1000 — averaged across datasets
WEIGHTS = {"test_ndcg10": 0.4, "test_f1_5": 0.4, "test_ndcg1000": 0.2}
datasets = sorted(set(r["dataset"] for r in rows))
models = sorted(set(r["model"] for r in rows))
seeds = sorted(set(int(r["seed"]) for r in rows))

# Build lookup: (model, dataset, seed) -> metrics
lookup: dict[tuple[str, str, int], dict[str, float]] = {}
for r in rows:
    key = (r["model"], r["dataset"], int(r["seed"]))
    lookup[key] = {
        "test_ndcg10": float(r["test_ndcg10"]),
        "test_ndcg1000": float(r["test_ndcg1000"]),
        "test_f1_5": float(r["test_f1_5"]),
    }

# Compute weighted average across datasets for each (model, seed)
model_seed_scores: dict[str, list[float]] = defaultdict(list)

for model in models:
    for seed in seeds:
        # Compute weighted average across datasets
        total = 0.0
        n_ds = 0
        for ds in datasets:
            key = (model, ds, seed)
            if key not in lookup:
                continue
            m = lookup[key]
            wt_avg = sum(m[k] * w for k, w in WEIGHTS.items())
            total += wt_avg
            n_ds += 1
        if n_ds > 0:
            model_seed_scores[model].append(total / n_ds)

# Summary statistics
print()
print("=" * 80)
print("MULTI-SEED EVALUATION SUMMARY")
print("=" * 80)
print()
print(f"{'Model':<60s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'N':>4s}")
print("-" * 92)

model_stats: dict[str, tuple[float, float]] = {}
for model in models:
    scores = model_seed_scores[model]
    if not scores:
        continue
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
    std = math.sqrt(variance)
    lo, hi = min(scores), max(scores)
    model_stats[model] = (mean, std)
    print(f"{model:<60s} {mean:>8.6f} {std:>8.6f} {lo:>8.6f} {hi:>8.6f} {len(scores):>4d}")

# Per-dataset breakdown
print()
print("PER-DATASET BREAKDOWN (Mean ± Std across seeds)")
print("-" * 100)
print(f"{'Model':<55s} {'Dataset':<8s} {'NDCG@10':>10s} {'NDCG@1000':>10s} {'F1@5':>10s}")
print("-" * 100)

for model in models:
    for ds in datasets:
        ndcg10s, ndcg1000s, f1s = [], [], []
        for seed in seeds:
            key = (model, ds, seed)
            if key not in lookup:
                continue
            m = lookup[key]
            ndcg10s.append(m["test_ndcg10"])
            ndcg1000s.append(m["test_ndcg1000"])
            f1s.append(m["test_f1_5"])
        if not ndcg10s:
            continue
        def _fmt(vals: list[float]) -> str:
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                var = sum((v - mean)**2 for v in vals) / (len(vals) - 1)
                std = math.sqrt(var)
                return f"{mean:.6f}±{std:.6f}"
            return f"{mean:.6f}"
        print(f"{model:<55s} {ds:<8s} {_fmt(ndcg10s):>22s} {_fmt(ndcg1000s):>22s} {_fmt(f1s):>22s}")

# Paired difference test (if exactly 2 models)
if len(models) == 2:
    m_a, m_b = models
    print()
    print("=" * 80)
    print(f"PAIRED DIFFERENCE TEST: {m_a}  vs  {m_b}")
    print("=" * 80)

    diffs = []
    for seed in seeds:
        scores_a = model_seed_scores[m_a]
        scores_b = model_seed_scores[m_b]
        idx = seeds.index(seed)
        if idx < len(scores_a) and idx < len(scores_b):
            diffs.append(scores_a[idx] - scores_b[idx])

    if len(diffs) >= 2:
        mean_diff = sum(diffs) / len(diffs)
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
        std_diff = math.sqrt(var_diff)
        se_diff = std_diff / math.sqrt(len(diffs))

        # t-statistic (two-sided)
        t_stat = mean_diff / se_diff if se_diff > 0 else float('inf')

        # Approximate p-value using normal approximation (conservative for small N)
        # For a proper t-test with df=N-1, but normal is fine for a rough guide
        ci_lo = mean_diff - 2.0 * se_diff  # ~95% CI (approx for df>=4)
        ci_hi = mean_diff + 2.0 * se_diff

        print()
        print(f"  Mean difference (A - B):  {mean_diff:+.6f}")
        print(f"  Std of differences:       {std_diff:.6f}")
        print(f"  Standard error:           {se_diff:.6f}")
        print(f"  t-statistic:              {t_stat:+.4f}")
        print(f"  95% CI:                   [{ci_lo:+.6f}, {ci_hi:+.6f}]")
        print(f"  N (paired seeds):         {len(diffs)}")
        print()

        if ci_lo <= 0 <= ci_hi:
            print("  ⇒ 95% CI includes zero: models are STATISTICALLY INDISTINGUISHABLE.")
            print("  ⇒ Recommendation: choose the SIMPLER model.")
        elif mean_diff > 0:
            print(f"  ⇒ {m_a} is significantly BETTER (95% CI excludes zero).")
        else:
            print(f"  ⇒ {m_b} is significantly BETTER (95% CI excludes zero).")
    else:
        print("  Not enough paired seeds for a paired test (need ≥ 2).")

print()
print(f"Full per-seed results saved to: {csv_path}")
ANALYSIS
