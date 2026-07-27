#!/bin/bash
cd "$(dirname "$0")/../.."
SAMPLE_SEED="${SAMPLE_SEED:-$RANDOM}"
python main.py \
  --name power_heavy_50000 \
  --config_file Config/Extreme/power.yaml \
  --sample \
  --heavy \
  --condition \
  --sample_seed "${SAMPLE_SEED}" \
  "$@"
