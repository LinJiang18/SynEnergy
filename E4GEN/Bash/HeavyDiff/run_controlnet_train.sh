#!/usr/bin/env bash
# HeavyDiff ControlNet: train AE + z_a then ControlNet (requires backbone at OUTPUT/power_heavyN)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E4GEN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${E4GEN_ROOT}"

CONFIG="Config/Extreme/power_controlnet.yaml"
NAME="power_heavy_controlnet_50000"

python main.py \
  --name "${NAME}" \
  --config_file "${CONFIG}" \
  --train \
  --controlnet \
  --condition \
  --heavy \
  "$@"
