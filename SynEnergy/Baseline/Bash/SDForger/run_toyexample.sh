#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../fms-dgt"

source "${HOME}/.virtualenvs/E4GEN/Baselines/SDForger/bin/activate"

# Optional: set data root explicitly (default: repo data/; see DGT_DATA_DIR in fms_dgt/constants.py)
# export DGT_DATA_DIR="${SCRIPT_DIR}/../../fms-dgt/data"

python -m fms_dgt.public \
  --task-paths ./tasks/public/time_series/toyexample_multisample/task.yaml
