#!/usr/bin/env bash
# HeavyDiff pipeline: train then sample for each (config_name, name) pair.
# USE_CONDITION=true -> adds --condition (requires cond_path in yaml).
# --heavy enables Student-t noise (overrides model.params.heavy in yaml).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E4GEN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${E4GEN_ROOT}"

USE_CONDITION=true
CONDITION_FLAG=""
if [[ "${USE_CONDITION}" == true ]]; then
  CONDITION_FLAG="--condition"
fi

# config_name (under Config/Extreme/) | run name (--name)
# ControlNet power_heavy_controlnet_NY1 auto-maps to backbone power_heavy_NY1 — train this first.
RUNS=(
  "power_CA.yaml power_heavy_CA1"
)

for RUN in "${RUNS[@]}"; do
  read -r CONFIG_NAME NAME <<< "${RUN}"
  CONFIG="Config/Extreme/${CONFIG_NAME}"

  echo "=========================================="
  echo "=== Run: ${NAME} (${CONFIG}) ==="
  echo "=========================================="

  echo "=== [${NAME} 1/2] Train (--heavy) ==="
  python main.py --name "${NAME}" --config_file "${CONFIG}" --train --heavy ${CONDITION_FLAG} "$@"

  echo "=== [${NAME} 2/2] Sample (--heavy) ==="
  python main.py --name "${NAME}" --config_file "${CONFIG}" --sample --heavy ${CONDITION_FLAG} "$@"

  echo "=== ${NAME} finished ==="
  echo
done

echo "All runs finished (${#RUNS[@]} jobs)."
