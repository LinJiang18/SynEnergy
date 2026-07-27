#!/usr/bin/env bash
set -eu

# SDForger time-series runner (default: Tallahassee 1000 × 180 quick trial).
# 1) Export npy -> parquet if needed
# 2) Finetune GPT-2 + vLLM generation
# 3) Copy final_data.jsonl to Baseline/Results/SDForger/

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E4GEN_ROOT="$(cd "$ROOT/../.." && pwd)"
TASKS_ROOT="$ROOT/tasks/public/time_series"
OUTPUT_ROOT="$ROOT/output/time_series"
VENV_ACTIVATE="${HOME}/.virtualenvs/E4GEN/Baselines/SDForger/bin/activate"
EXPORT_SCRIPT="$ROOT/scripts/export_Florida_multisample_parquet.py"

# Tallahassee 1000-sample assets (must match task.yaml + export_Florida_multisample_parquet.py)
TALLAHASSEE_NPY="$E4GEN_ROOT/Data/Power/Tallahassee_seq_180_sample_1000.npy"
TALLAHASSEE_PARQUET="$ROOT/data/public/time_series/Tallahassee_seq_180_sample_1000.parquet"

# Args: task name and directory for saved results
TASK_NAME="${1:-Tallahassee_multisample}"
SAVE_DIR="${2:-$ROOT/../Results/SDForger}"

cd "$ROOT"

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "ERROR: SDForger venv not found at $VENV_ACTIVATE"
  echo "Create it with:"
  echo "  mkdir -p ~/.virtualenvs/E4GEN/Baselines"
  echo "  python3 -m venv ~/.virtualenvs/E4GEN/Baselines/SDForger"
  echo "  source ~/.virtualenvs/E4GEN/Baselines/SDForger/bin/activate"
  echo "  pip install -e \".[time_series]\""
  echo "  pip install -r Bash/requirements-sdforger-cu128.txt"
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

python - <<'PY'
import sys

import torch
import transformers

issues = []
if not torch.cuda.is_available():
    issues.append(
        "CUDA is unavailable. Use torch cu128 wheels, e.g.\n"
        "  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 "
        "--index-url https://download.pytorch.org/whl/cu128"
    )

major = int(transformers.__version__.split(".", 1)[0])
if major >= 5:
    issues.append(
        f"transformers {transformers.__version__} is incompatible with vllm 0.10.x.\n"
        "  pip install 'transformers>=4.45,<5'"
    )

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer

    get_tokenizer("gpt2")
except Exception as err:  # pragma: no cover - runtime env check
    issues.append(f"vLLM tokenizer check failed: {err}")

if issues:
    print("ERROR: SDForger environment check failed:", file=sys.stderr)
    for item in issues:
        print(f"  - {item}", file=sys.stderr)
    print(
        "  Install pinned deps with: pip install -r Bash/requirements-sdforger-cu128.txt",
        file=sys.stderr,
    )
    sys.exit(1)
PY

TASK_YAML="$TASKS_ROOT/${TASK_NAME}/task.yaml"
TASK_OUT="$OUTPUT_ROOT/${TASK_NAME}"
CHECKPOINT="$TASK_OUT/model_iter-1/best/model.safetensors"
DATA_MARKER=""

if [ ! -f "$TASK_YAML" ]; then
  echo "ERROR: task config not found: $TASK_YAML"
  exit 1
fi

if [ "$TASK_NAME" = "Tallahassee_multisample" ]; then
  if [ ! -f "$TALLAHASSEE_NPY" ]; then
    echo "ERROR: missing input npy: $TALLAHASSEE_NPY"
    exit 1
  fi
  if [ ! -f "$EXPORT_SCRIPT" ]; then
    echo "ERROR: missing export script: $EXPORT_SCRIPT"
    exit 1
  fi
  if [ ! -f "$TALLAHASSEE_PARQUET" ] || [ "$TALLAHASSEE_NPY" -nt "$TALLAHASSEE_PARQUET" ]; then
    echo "===> Exporting Tallahassee parquet (1000 trajectories × 180 steps)"
    python "$EXPORT_SCRIPT" \
      --input "$TALLAHASSEE_NPY" \
      --output "$TALLAHASSEE_PARQUET" \
      --n-samples 1000 \
      --seq-len 180
  else
    echo "===> Parquet up to date: $TALLAHASSEE_PARQUET"
  fi
  DATA_MARKER="$TALLAHASSEE_PARQUET"
fi

if [ -f "$CHECKPOINT" ]; then
  if [ -n "$DATA_MARKER" ] && [ "$DATA_MARKER" -nt "$CHECKPOINT" ]; then
    echo "===> Parquet newer than checkpoint; will retrain"
    unset SDFORGER_SKIP_TRAIN
  else
    export SDFORGER_SKIP_TRAIN=1
    echo "===> Found checkpoint; skipping finetune (SDFORGER_SKIP_TRAIN=1)"
    echo "===> Checkpoint: $CHECKPOINT"
  fi
else
  unset SDFORGER_SKIP_TRAIN
fi

echo "===> Running SDForger task: $TASK_NAME"
echo "===> ROOT: $ROOT"
if [ "$TASK_NAME" = "Tallahassee_multisample" ]; then
  echo "===> Data: $TALLAHASSEE_PARQUET (train/generate 1000 samples)"
fi
# NOTE: time_series is a TransformationTask; do not pass --restart
# because it injects restart_generation (unsupported for this task type).
python -m fms_dgt.public --task-paths "$TASK_YAML"

FINAL_JSONL="$TASK_OUT/final_data.jsonl"

if [ ! -f "$FINAL_JSONL" ]; then
  echo "ERROR: $FINAL_JSONL task failed"
  exit 1
fi

mkdir -p "$SAVE_DIR"
cp "$FINAL_JSONL" "$SAVE_DIR/final_data_${TASK_NAME}.jsonl"

echo "===> Copied to $SAVE_DIR/final_data_${TASK_NAME}.jsonl"

rm -rf "$TASK_OUT"

echo "===> Cleanup done; kept only $SAVE_DIR/final_data_${TASK_NAME}.jsonl"
