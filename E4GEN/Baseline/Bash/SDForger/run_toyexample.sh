#!/usr/bin/env bash
cd /home/ec2-user/project/E4GEN/Baseline/fms-dgt

source "${HOME}/.virtualenvs/E4GEN/Baselines/SDForger/bin/activate"

# Optional: set data root explicitly (default: repo data/; see DGT_DATA_DIR in fms_dgt/constants.py)
# export DGT_DATA_DIR=/home/ec2-user/project/E4GEN/Baseline/fms-dgt/data

python -m fms_dgt.public \
  --task-paths ./tasks/public/time_series/toyexample_multisample/task.yaml
