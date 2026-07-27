#!/usr/bin/env bash
# ControlNet full pipeline: train then sample for each (config_name, name) pair.
# Requires matching backbone: power_controlnet_* -> power_* (backbone_run_name: auto).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNENERGY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SYNENERGY_ROOT}"

# config_name (under Config/Extreme/) | run name (--name)
RUNS=(
  "power_controlnet_CA.yaml power_controlnet_CA1"
)

for RUN in "${RUNS[@]}"; do
  read -r CONFIG_NAME NAME <<< "${RUN}"
  CONFIG="Config/Extreme/${CONFIG_NAME}"

  echo "=========================================="
  echo "=== Run: ${NAME} (${CONFIG}) ==="
  echo "=========================================="

  echo "=== [${NAME} 1/2] Train (AE + ControlNet) ==="
  python main.py \
    --name "${NAME}" \
    --config_file "${CONFIG}" \
    --train \
    --controlnet \
    --condition \
    "$@"

  echo "=== [${NAME} 2/2] Sample ==="
  python main.py \
    --name "${NAME}" \
    --config_file "${CONFIG}" \
    --sample \
    --controlnet \
    --condition \
    "$@"

  echo "=== ${NAME} finished ==="
  echo
done

echo "All runs finished (${#RUNS[@]} jobs)."
