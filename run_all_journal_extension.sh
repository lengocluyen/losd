#!/usr/bin/env bash
set -euo pipefail

EMBEDDER="${1:-sentence-transformer}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

for d in exp_out*; do
  [ -d "$d" ] || continue
  [ -f "$d/results_all_methods.csv" ] || continue

  echo "=== Running $d ==="
  python run_journal_extension.py --source-outdir "$d" --embedder "$EMBEDDER"
  python plot_journal_results.py --results-dir "$d/journal_extension"
done
