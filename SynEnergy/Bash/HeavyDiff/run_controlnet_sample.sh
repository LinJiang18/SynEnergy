#!/usr/bin/env bash
# HeavyDiff ControlNet: sample with trained ControlNet + z_a from OUTPUT/intermediate/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNENERGY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SYNENERGY_ROOT}"

CONFIG="Config/Extreme/power_controlnet.yaml"
NAME="power_heavy_controlnet_50000"
# Stratified cond sampling seed; override: SAMPLE_SEED=123 ./run_controlnet_sample.sh
SAMPLE_SEED="${SAMPLE_SEED:-$RANDOM}"

python main.py \
  --name "${NAME}" \
  --config_file "${CONFIG}" \
  --sample \
  --controlnet \
  --condition \
  --heavy \
  --sample_seed "${SAMPLE_SEED}" \
  "$@"
