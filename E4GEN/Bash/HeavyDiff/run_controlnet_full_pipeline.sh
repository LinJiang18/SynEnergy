#!/usr/bin/env bash
# HeavyDiff ControlNet full pipeline: train then sample for each (config_name, name) pair.
# Requires matching backbone: power_heavy_controlnet_* -> power_heavy_* (backbone_run_name: auto).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E4GEN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${E4GEN_ROOT}"

# config_name (under Config/Extreme/) | run name (--name)
RUNS=(
  "power_controlnet_CA.yaml power_heavy_controlnet_CA1"
)

for RUN in "${RUNS[@]}"; do
  read -r CONFIG_NAME NAME <<< "${RUN}"
  CONFIG="Config/Extreme/${CONFIG_NAME}"

  echo "=========================================="
  echo "=== Run: ${NAME} (${CONFIG}) ==="
  echo "=========================================="

  echo "=== [${NAME} 1/2] Train (AE + ControlNet, --heavy) ==="
  python main.py \
    --name "${NAME}" \
    --config_file "${CONFIG}" \
    --train \
    --controlnet \
    --condition \
    --heavy \
    "$@"

  echo "=== [${NAME} 2/2] Sample (--heavy) ==="
  python main.py \
    --name "${NAME}" \
    --config_file "${CONFIG}" \
    --sample \
    --controlnet \
    --condition \
    --heavy \
    "$@"

  echo "=== ${NAME} finished ==="
  echo
done

echo "All runs finished (${#RUNS[@]} jobs)."
