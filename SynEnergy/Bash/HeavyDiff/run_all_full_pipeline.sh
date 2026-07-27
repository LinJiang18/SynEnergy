#!/usr/bin/env bash
# Full HeavyDiff stack: timeseries backbone (power_heavy1–5) then ControlNet (power_heavy_controlnet1–5).
# ControlNet requires matching backbone OUTPUT/power_heavyN (backbone_run_name: auto).
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
echo "All pipelines finished: timeseries (power_heavy_50000) + controlnet (power_heavy_controlnet_50000)."
