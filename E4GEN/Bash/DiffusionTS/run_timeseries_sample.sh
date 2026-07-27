#!/bin/bash
cd "$(dirname "$0")/../.."
SAMPLE_SEED="${SAMPLE_SEED:-$RANDOM}"
python main.py \
  --name power1 \
  --config_file Config/Extreme/power.yaml \
  --sample \
  --condition \
  --sample_seed "${SAMPLE_SEED}" \
  "$@"
