#!/usr/bin/env sh
set -eu

# Incremental by default:
# - We do NOT delete SCOREBOARD.md unless --clean is provided.
# - We run a small default benchmark set to speed up iteration.
#
# Usage:
#   ./regenerate_scoreboard.sh                # all datasets, default models
#   ./regenerate_scoreboard.sh --clean        # rebuild from scratch
#   ./regenerate_scoreboard.sh --dataset yso-fi
#   ./regenerate_scoreboard.sh --models baseline,mean_weighted,torch_per_label
#
# Notes:
# - "models" are benchmark modules under `benchmarks.*` (without the prefix).
# - The default suite is intentionally small and focused on best-performing models.

CLEAN=0
DATASETS="yso-fi yso-en koko"
MODELS="baseline mean_weighted torch_mean torch_mean_bias torch_mean_residual torch_per_label torch_per_label_l1_delta torch_nn"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --dataset)
      DATASETS="$2"
      shift 2
      ;;
    --models)
      MODELS=$(printf "%s" "$2" | tr ',' ' ')
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "Updating SCOREBOARD.md (incremental=${CLEAN}==0)..."
echo "Datasets: $DATASETS"
echo "Models:   $MODELS"

if [ "$CLEAN" -eq 1 ]; then
  echo "Cleaning SCOREBOARD.md..."
  rm -f SCOREBOARD.md
fi

for ds in $DATASETS; do
  echo ""
  echo "============================="
  echo "DATASET: $ds"
  echo "============================="

  for m in $MODELS; do
    echo ""
    echo "---- Running: benchmarks.$m (dataset=$ds) ----"
    uv run python -m "benchmarks.$m" --dataset "$ds"
  done
done

echo ""
echo "Done. SCOREBOARD.md updated."
