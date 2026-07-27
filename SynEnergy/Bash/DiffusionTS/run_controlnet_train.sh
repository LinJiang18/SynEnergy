#!/usr/bin/env bash
# ControlNet: train AE + z_a then ControlNet (requires backbone at OUTPUT/power)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNENERGY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SYNENERGY_ROOT}"

CONFIG="Config/Extreme/power_controlnet.yaml"
NAME="power_controlnet_1000_size_720"

python main.py \
  --name "${NAME}" \
  --config_file "${CONFIG}" \
  --train \
  --controlnet \
  --condition \
  "$@"
