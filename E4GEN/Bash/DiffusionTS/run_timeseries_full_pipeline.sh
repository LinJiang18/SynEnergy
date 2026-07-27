#!/usr/bin/env bash
# DiffusionTS pipeline: train then sample for each (config_name, name) pair.
# USE_CONDITION=true -> adds --condition (requires cond_path in yaml).
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
# ControlNet power_controlnet_NY1 auto-maps to backbone power_NY1 — train this first.
RUNS=(
  "power_CA.yaml power_CA1"
)


for RUN in "${RUNS[@]}"; do
  read -r CONFIG_NAME NAME <<< "${RUN}"
  CONFIG="Config/Extreme/${CONFIG_NAME}"

  echo "=========================================="
  echo "=== Run: ${NAME} (${CONFIG}) ==="
  echo "=========================================="

  echo "=== [${NAME} 1/2] Train ==="
  python main.py --name "${NAME}" --config_file "${CONFIG}" --train ${CONDITION_FLAG} "$@"

  echo "=== [${NAME} 2/2] Sample ==="
  python main.py --name "${NAME}" --config_file "${CONFIG}" --sample ${CONDITION_FLAG} "$@"

  echo "=== ${NAME} finished ==="
  echo
done

echo "All runs finished (${#RUNS[@]} jobs)."
