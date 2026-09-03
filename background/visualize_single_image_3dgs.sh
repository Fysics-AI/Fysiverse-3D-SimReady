#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/ckpt_last.pt" >&2
  echo "Optional env: MESH_OBJECT=/path/model.glb MESH_DIR=/path/meshes ENV_NAME=/path/env.hdr PORT=8080" >&2
  exit 2
fi

CKPT_PATH="$(realpath "$1")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THREEDGRUT_ROOT="${THREEDGRUT_ROOT:-${WORKFLOW_ROOT}/third_party/3dgrut}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
ENV_NAME="${ENV_NAME:-Black}"
MESH_OBJECT="${MESH_OBJECT:-}"
MESH_DIR="${MESH_DIR:-}"
BLACK_BACKGROUND="${BLACK_BACKGROUND:-1}"

cd "${THREEDGRUT_ROOT}"
if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing 3DGRUT virtualenv: ${THREEDGRUT_ROOT}/.venv" >&2
  exit 2
fi
source .venv/bin/activate

ARGS=(
  python threedgrut_playground/viser_gui.py
  --gs_object "${CKPT_PATH}"
  --default_gs_config apps/playground_3dgrt_3dgut_adapter.yaml
  --host "${HOST}"
  --port "${PORT}"
)

if [[ -n "${ENV_NAME}" ]]; then
  ARGS+=(--env_name "${ENV_NAME}")
fi
if [[ -n "${MESH_OBJECT}" ]]; then
  ARGS+=(--mesh_object "${MESH_OBJECT}")
fi
if [[ -n "${MESH_DIR}" ]]; then
  ARGS+=(--mesh_dir "${MESH_DIR}")
fi
if [[ "${BLACK_BACKGROUND}" == "1" || "${BLACK_BACKGROUND}" == "true" ]]; then
  ARGS+=(--black_background)
fi

echo "[single-image-3dgs] viewer: http://<server-host>:${PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${ARGS[@]}"
