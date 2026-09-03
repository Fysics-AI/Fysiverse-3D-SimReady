#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_ROOT="${WORKFLOW_ROOT:-${SCRIPT_DIR}}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-${WORKFLOW_ROOT}/config/main.yaml}"
CONFIG_LOADER_PYTHON="${CONFIG_LOADER_PYTHON:-python3}"

if [[ -f "${PIPELINE_CONFIG}" ]]; then
  CONFIG_EXPORTS="$("${CONFIG_LOADER_PYTHON}" "${WORKFLOW_ROOT}/scripts/load_pipeline_config.py" --config "${PIPELINE_CONFIG}" 2>/dev/null || true)"
  while IFS= read -r line; do
    case "${line}" in
      "export "*)
        eval "${line}"
        ;;
    esac
  done <<< "${CONFIG_EXPORTS}"
fi

LOCAL_ENV="${LOCAL_ENV:-${WORKFLOW_ROOT}/.env.local}"
if [[ -f "${LOCAL_ENV}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${LOCAL_ENV}"
  set +a
fi

CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-${FYSIVERSE_CONDA_ENV:-fysiverse-scene}}"
SERVER_NAME="${SERVER_NAME:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-7860}"
PORT_SEARCH_COUNT="${PORT_SEARCH_COUNT:-100}"

export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"

cd "${WORKFLOW_ROOT}"
mkdir -p logs sessions

echo "Launching Fysiverse Gradio app"
echo "  workflow: ${WORKFLOW_ROOT}"
echo "  config:   ${PIPELINE_CONFIG}"
echo "  env:      ${CONDA_ENV}"
echo "  address:  ${SERVER_NAME}:${SERVER_PORT}"

exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python "${WORKFLOW_ROOT}/gradio_app.py" \
  --config "${PIPELINE_CONFIG}" \
  --server-name "${SERVER_NAME}" \
  --server-port "${SERVER_PORT}" \
  --port-search-count "${PORT_SEARCH_COUNT}"
