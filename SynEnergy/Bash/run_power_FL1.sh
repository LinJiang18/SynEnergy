#!/usr/bin/env bash
# Full FL pipeline: Stage-1 backbone -> Stage-2 ControlNet (AE + dual graph) -> sample.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNENERGY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SYNENERGY_ROOT}"

CONFIG="Config/Extreme/power_FL.yaml"
NAME="power_FL1"

echo "=========================================="
echo " SynEnergy FL pipeline: ${NAME}"
echo " Config: ${CONFIG}"
echo "=========================================="

echo "=== [1/3] Stage 1 — train backbone ==="
python main.py \
  --name "${NAME}" \
  --config_file "${CONFIG}" \
  --train \
  --heavy \
  --condition \
  "$@"

echo "=== [2/3] Stage 2 — train ControlNet (AE + graph + ControlNet) ==="
python main.py \
  --name "${NAME}" \
  --config_file "${CONFIG}" \
  --train \
  --controlnet \
  --heavy \
  --condition \
  "$@"

echo "=== [3/3] Sample ==="
python main.py \
  --name "${NAME}" \
  --config_file "${CONFIG}" \
  --sample \
  --controlnet \
  --heavy \
  --condition \
  "$@"

echo "Done. Outputs: OUTPUT/${NAME}/"
