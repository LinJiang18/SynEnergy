#!/usr/bin/env bash
# Detached submit for run_all_full_pipeline.sh (survives SSH / local disconnect).
# Usage:
#   bash Bash/HeavyDiff/submit_all_full_pipeline.sh
#   bash Bash/HeavyDiff/submit_all_full_pipeline.sh --sample_seed 42   # extra args forwarded
#
# Optional:
#   PYTHON_BIN=/path/to/python bash Bash/HeavyDiff/submit_all_full_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E4GEN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${E4GEN_ROOT}"

LOG_DIR="${E4GEN_ROOT}/OUTPUT/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/heavy_all_full_pipeline_${STAMP}.log"
PID_FILE="${LOG_DIR}/heavy_all_full_pipeline_${STAMP}.pid"

# Resolve python: explicit PYTHON_BIN > Diffusion-TS venv > E4GEN conda > PATH
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${HOME}/.virtualenvs/Diffusion-TS/bin/python" ]]; then
    PYTHON_BIN="${HOME}/.virtualenvs/Diffusion-TS/bin/python"
  elif [[ -x "${HOME}/miniconda3/envs/E4GEN/bin/python" ]]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/E4GEN/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi
export PYTHON_BIN
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

# Wrapper so child scripts' `python` resolves to the chosen env.
WRAPPER="${LOG_DIR}/_python_wrapper_heavy_${STAMP}.sh"
cat >"${WRAPPER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="$(dirname "${PYTHON_BIN}"):\${PATH}"
export PYTHON_BIN="${PYTHON_BIN}"
exec bash "${SCRIPT_DIR}/run_all_full_pipeline.sh" "\$@"
EOF
chmod +x "${WRAPPER}"

nohup bash "${WRAPPER}" "$@" >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" >"${PID_FILE}"
disown "${PID}" 2>/dev/null || true

echo "Submitted heavy_all_full_pipeline (pid=${PID})"
echo "  python: ${PYTHON_BIN}"
echo "  log:    ${LOG_FILE}"
echo "  pid:    ${PID_FILE}"
echo
echo "Monitor:  tail -f ${LOG_FILE}"
echo "Stop:     kill ${PID}"
