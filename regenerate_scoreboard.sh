#!/usr/bin/env sh
set -eu

echo "Regenerating SCOREBOARD.md..."

# Remove old scoreboard
rm -f SCOREBOARD.md

# Base benchmarks
uv run python -m benchmarks.inspect_and_ndcg
uv run python -m benchmarks.mean_ensemble
uv run python -m benchmarks.weighted_mean_ensemble

# Torch-based benchmarks
uv run python -m benchmarks.torch_mean_ensemble
uv run python -m benchmarks.torch_mean_bias_ensemble

echo "Done. SCOREBOARD.md regenerated."
