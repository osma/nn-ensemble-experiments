#!/usr/bin/env sh
set -eu

echo "Regenerating SCOREBOARD.md..."

# Remove old scoreboard
rm -f SCOREBOARD.md

# Baselines
uv run python -m benchmarks.baseline

# Non-torch ensembles
uv run python -m benchmarks.mean
uv run python -m benchmarks.mean_weighted

# Torch-based ensembles
uv run python -m benchmarks.torch_mean
uv run python -m benchmarks.torch_mean_bias
uv run python -m benchmarks.torch_per_label
uv run python -m benchmarks.torch_per_label_conv
uv run python -m benchmarks.torch_per_label_conv_residual
uv run python -m benchmarks.torch_per_label_conv_posthoc_residual

echo "Done. SCOREBOARD.md regenerated."
