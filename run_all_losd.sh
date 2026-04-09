#!/usr/bin/env bash
set -euo pipefail

EMBEDDER="${1:-sentence-transformer}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

for d in exp_out*; do
  [ -d "$d" ] || continue
  [ -f "$d/results_all_methods.csv" ] || continue

  echo "=== Running $d ==="
  python run_losd.py --source-outdir "$d" --embedder "$EMBEDDER"
  python plot_losd_results.py --results-dir "$d/losd"
done
