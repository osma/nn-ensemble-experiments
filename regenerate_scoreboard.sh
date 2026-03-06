#!/usr/bin/env sh
set -eu

echo "Regenerating SCOREBOARD.md..."

# Remove old scoreboard
rm -f SCOREBOARD.md

for ds in yso-fi yso-en koko; do
  echo ""
  echo "============================="
  echo "DATASET: $ds"
  echo "============================="

  # Baselines
  uv run python -m benchmarks.baseline --dataset "$ds"

  # Non-torch ensembles (3-way as defined in benchmarks/datasets.py)
  uv run python -m benchmarks.mean --dataset "$ds"
  uv run python -m benchmarks.mean_weighted --dataset "$ds"

  # Torch-based ensembles (3-way as defined in benchmarks/datasets.py)
  uv run python -m benchmarks.torch_mean --dataset "$ds"
  uv run python -m benchmarks.torch_mean_bias --dataset "$ds"
  uv run python -m benchmarks.torch_per_label --dataset "$ds"
  uv run python -m benchmarks.torch_per_label_conv --dataset "$ds"
  uv run python -m benchmarks.torch_per_label_freq_gate --dataset "$ds"
  uv run python -m benchmarks.torch_per_label_freq_reg --dataset "$ds"
done

echo "Done. SCOREBOARD.md regenerated."
