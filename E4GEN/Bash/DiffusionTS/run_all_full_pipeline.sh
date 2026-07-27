#!/usr/bin/env bash
# Full DiffusionTS stack: timeseries backbone (power1–5) then ControlNet (power_controlnet1–5).
# ControlNet requires matching backbone OUTPUT/powerN (backbone_run_name: auto).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "##########################################"
echo "### Stage A: Timeseries full pipeline  ###"
echo "##########################################"
bash "${SCRIPT_DIR}/run_timeseries_full_pipeline.sh" "$@"

echo
echo "##########################################"
echo "### Stage B: ControlNet full pipeline  ###"
echo "##########################################"
bash "${SCRIPT_DIR}/run_controlnet_full_pipeline.sh" "$@"

echo
echo "All pipelines finished: timeseries (power_50000) + controlnet (power_controlnet_50000)."
