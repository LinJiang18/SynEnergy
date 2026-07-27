#!/usr/bin/env bash
# Run run_extreme_experiment.py once per *.npy under data/.
# Expects filenames like: <project>_seq_<T>_sample_<N>.npy  ->  --project-name <project>
# Edit data/*.npy set or this script if your naming differs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

shopt -s nullglob
npys=(data/*.npy)
shopt -u nullglob

if [[ ${#npys[@]} -eq 0 ]]; then
  echo "No data/*.npy found under $ROOT/data" >&2
  exit 1
fi

echo "Found ${#npys[@]} dataset(s) under data/"
for f in "${npys[@]}"; do
  b="$(basename "$f")"
  if [[ "$b" =~ ^(.+)_seq_[0-9]+_sample_[0-9]+\.npy$ ]]; then
    project="${BASH_REMATCH[1]}"
  else
    project="${b%.npy}"
    echo "WARN: $b does not match *_seq_*_sample_*.npy; using project=$project" >&2
  fi

  echo "========== project=$project  data-file=$b =========="
  FFLOWS_PROJECT_NAME="$project" FFLOWS_DATA_FILE="$b" python3 run_extreme_experiment.py
done

echo "All runs finished."
