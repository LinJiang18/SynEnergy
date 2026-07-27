#!/usr/bin/env bash
# Train + unconditional sample for five Extreme datasets.
# Expects ./data/<prefix>*.npy (e.g. precipitation_seq_90_sample_1056.npy).
#
# Usage:
#   chmod +x run_five_datasets_train_and_sample.sh
#   ./run_five_datasets_train_and_sample.sh
#
# Optional env:
#   MAX_EPOCHS=200     default 200; used only when MAX_STEPS is unset
#   MAX_STEPS=         if set (non-empty), train with --max_steps and ignore MAX_EPOCHS
#   DEVICE=cuda:0      passed to both train and sample

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATASETS=(precipitation temperature toyexample transportation windspeed)

MAX_EPOCHS="${MAX_EPOCHS:-200}"
DEVICE="${DEVICE:-cuda:0}"

for name in "${DATASETS[@]}"; do
  echo "=============================="
  echo "Dataset: $name (train)"
  echo "=============================="
  if [[ -n "${MAX_STEPS:-}" ]]; then
    python3 train_unconditional.py -E "$name" --device "$DEVICE" --max_steps "$MAX_STEPS"
  else
    python3 train_unconditional.py -E "$name" --device "$DEVICE" --max_epochs "$MAX_EPOCHS"
  fi

  echo "=============================="
  echo "Dataset: $name (sample)"
  echo "=============================="
  python3 sample_unconditional.py -E "$name" --device "$DEVICE"
done

echo "Done. Checkpoints: Output/<NAME>/model.pth"
echo "Generated: Output/<NAME>/<NAME>_generated_data.npy"
