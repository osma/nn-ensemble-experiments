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
NO_CACHE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --no-cache)
      NO_CACHE=1
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

CACHE_DIR=".cache/benchmarks"
CACHE_VER="v1"

echo "Updating SCOREBOARD.md (incremental=${CLEAN}==0, no_cache=$NO_CACHE)..."
echo "Datasets: $DATASETS"
echo "Models:   $MODELS"
echo "Cache:    $CACHE_DIR ($CACHE_VER)"

mkdir -p "$CACHE_DIR"

if [ "$CLEAN" -eq 1 ]; then
  echo "Cleaning SCOREBOARD.md..."
  rm -f SCOREBOARD.md
fi

dataset_hash() {
  ds="$1"
  # Hash all npz files under data/<ds>/ in a stable order.
  # sha256sum output includes filename, so we hash that stream for a single digest.
  find "data/$ds" -maxdepth 1 -type f -name "*.npz" -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | awk '{print $1}'
}

script_hash() {
  m="$1"
  # Prefer the module file if it exists; otherwise hash the resolved module path by asking python.
  if [ -f "benchmarks/$m.py" ]; then
    sha256sum "benchmarks/$m.py" | awk '{print $1}'
    return 0
  fi
  uv run python -c "import importlib, pathlib; p=pathlib.Path(importlib.import_module('benchmarks.$m').__file__); print(p.read_bytes().hex())" \
    | sha256sum \
    | awk '{print $1}'
}

cache_path() {
  m="$1"
  ds="$2"
  dh="$(dataset_hash "$ds")"
  sh="$(script_hash "$m")"
  echo "$CACHE_DIR/$CACHE_VER--$m--$ds--$dh--$sh.metrics"
}

apply_cached_metrics() {
  m="$1"
  ds="$2"
  model_name="$3"
  cache_file="$4"

  # Each cache file stores two lines:
  #   train|epoch|ndcg@10|ndcg@1000|f1@5
  #   test|epoch|ndcg@10|ndcg@1000|f1@5
  while IFS='|' read -r split epoch ndcg10 ndcg1000 f1; do
    [ -n "$split" ] || continue
    uv run python -c "from pathlib import Path; from benchmarks.metrics import update_markdown_scoreboard; update_markdown_scoreboard(path=Path('SCOREBOARD.md'), model='$model_name', dataset='$ds', split='$split', metrics={'ndcg@10': float('$ndcg10'), 'ndcg@1000': float('$ndcg1000'), 'f1@5': float('$f1')}, n_samples=0, epoch=(int('$epoch') if '$epoch' else None))"
  done < "$cache_file"
}

extract_and_cache_metrics() {
  m="$1"
  ds="$2"
  model_name="$3"
  cache_file="$4"

  uv run python - "$m" "$ds" "$model_name" "$cache_file" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

def _find_row(lines: list[str], model: str, dataset: str) -> dict[str, str] | None:
    for line in lines:
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) != 8:
            continue
        if cols[0] == "Model":
            continue
        if cols[0] == model and cols[1] == dataset:
            return {
                "epoch": cols[2],
                "train_ndcg10": cols[3],
                "train_ndcg1000": cols[4],
                "test_ndcg10": cols[5],
                "test_ndcg1000": cols[6],
                "test_f1": cols[7],
            }
    return None

_, m, ds, model_name, cache_path = sys.argv
score_path = Path("SCOREBOARD.md")
if not score_path.exists():
    raise SystemExit("SCOREBOARD.md not found after run; cannot cache metrics.")

lines = score_path.read_text().splitlines()
row = _find_row(lines, model_name, ds)
if row is None:
    raise SystemExit(f"Could not find scoreboard row for model={model_name!r} dataset={ds!r}")

def _safe_float(s: str) -> float:
    return float(s) if s else float("nan")

epoch = row["epoch"]

out_lines = []
out_lines.append(
    "train|{epoch}|{n10}|{n1000}|{f1}\n".format(
        epoch=epoch,
        n10=_safe_float(row["train_ndcg10"]),
        n1000=_safe_float(row["train_ndcg1000"]),
        f1=float("nan"),
    )
)
out_lines.append(
    "test|{epoch}|{n10}|{n1000}|{f1}\n".format(
        epoch=epoch,
        n10=_safe_float(row["test_ndcg10"]),
        n1000=_safe_float(row["test_ndcg1000"]),
        f1=_safe_float(row["test_f1"]),
    )
)

Path(cache_path).write_text("".join(out_lines))
PY
}

for m in $MODELS; do
  echo ""
  echo "============================="
  echo "MODEL: $m"
  echo "============================="

  for ds in $DATASETS; do
    echo ""
    echo "---- Running: benchmarks.$m (dataset=$ds) ----"

    model_name="$m"
    if [ "$m" != "baseline" ]; then
      # Most modules format the model name as: "<module>(<ensemble_keys...>)"
      model_name="$(uv run python -c "from benchmarks.datasets import ensemble3_keys; e=ensemble3_keys('$ds'); print('$m(' + ','.join(e) + ')')")"
    fi

    cache_file="$(cache_path "$m" "$ds")"

    if [ "$NO_CACHE" -eq 0 ] && [ -f "$cache_file" ]; then
      echo "Cache hit: $cache_file"
      apply_cached_metrics "$m" "$ds" "$model_name" "$cache_file"
      continue
    fi

    echo "Cache miss (or disabled). Running benchmark..."
    uv run python -m "benchmarks.$m" --dataset "$ds"

    # Cache only final metrics written to scoreboard.
    extract_and_cache_metrics "$m" "$ds" "$model_name" "$cache_file"
    echo "Cached metrics: $cache_file"
  done
done

echo ""
echo "Done. SCOREBOARD.md updated."
