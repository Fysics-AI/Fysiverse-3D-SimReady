#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-18080}"
HOST="${2:-127.0.0.1}"
SERVER_PY="${ROOT_DIR}/project_page/serve_3dgs_check.py"

echo "Serving Fysiverse 3DGS preview"
echo "  root: ${ROOT_DIR}"
echo "  url:  http://${HOST}:${PORT}/project_page/3dgs_check.html?manifest=../sessions/<session_id>/results/3dgs_bg/scene.json"
echo "  url:  http://${HOST}:${PORT}/project_page/3dgs_check.html#manifest=../sessions/<session_id>/results/3dgs_bg/scene.json"

exec python3 "${SERVER_PY}" --directory "${ROOT_DIR}" --port "${PORT}" --host "${HOST}"
