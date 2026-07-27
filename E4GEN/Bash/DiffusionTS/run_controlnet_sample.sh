#!/usr/bin/env bash
# ControlNet: sample with trained ControlNet + z_a from OUTPUT/intermediate/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E4GEN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${E4GEN_ROOT}"

CONFIG="Config/Extreme/power_controlnet.yaml"
NAME="power_controlnet_1000_size_720"
# Stratified cond sampling seed; override: SAMPLE_SEED=123 ./run_controlnet_sample.sh
SAMPLE_SEED="${SAMPLE_SEED:-$RANDOM}"

python main.py \
  --name "${NAME}" \
  --config_file "${CONFIG}" \
  --sample \
  --controlnet \
  --condition \
  --sample_seed "${SAMPLE_SEED}" \
  "$@"
