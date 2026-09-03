#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/sessions/<session_id>" >&2
  exit 2
fi

SESSION_DIR="$(realpath "$1")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-${WORKFLOW_ROOT}/config/main.yaml}"
CONFIG_LOADER="${CONFIG_LOADER:-${WORKFLOW_ROOT}/scripts/load_pipeline_config.py}"
CONFIG_LOADER_PYTHON="${CONFIG_LOADER_PYTHON:-python3}"
PRESERVE_ENV_NAMES=(
  CONDA_BIN
  QWEN_ENV
  MOGE_ENV
  THREEDGRUT_ROOT
  GAUSSIAN_SPLATS_ROOT
  CUDA_VISIBLE_DEVICES
  ITERATIONS
  MAX_POINTS
  MOGE_ROOT
  MOGE_CKPT
  MOGE_RESIZE_MAX
  MOGE_RESOLUTION_LEVEL
  MOGE_DEVICE
  USE_REFERENCE_MOGE_FUSION
  REFERENCE_MOGE_CACHE
  FOREGROUND_MASK
  FUSION_DEPTH_ALIGN
  FOREGROUND_MASK_DILATE
  FORCE_EDIT
  SKIP_BACKGROUND_EDIT
  BACKGROUND_IMAGE_BACKEND
  BACKGROUND_IMAGE_TIMEOUT
  BACKGROUND_LOCAL_FALLBACK
  BACKGROUND_LOCAL_INPAINT_DILATE
  BACKGROUND_LOCAL_INPAINT_RADIUS
  KSPLAT_COMPRESSION
  KSPLAT_ALPHA_THRESHOLD
  KSPLAT_SCENE_CENTER
  KSPLAT_BLOCK_SIZE
  KSPLAT_BUCKET_SIZE
  KSPLAT_SH_DEGREE
)
declare -A PRESERVED_ENV=()
for name in "${PRESERVE_ENV_NAMES[@]}"; do
  if [[ -n "${!name:-}" ]]; then
    PRESERVED_ENV["${name}"]="${!name}"
  fi
done
if [[ -f "${PIPELINE_CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source <("${CONFIG_LOADER_PYTHON}" "${CONFIG_LOADER}" --config "${PIPELINE_CONFIG}")
fi
for name in "${PRESERVE_ENV_NAMES[@]}"; do
  if [[ -n "${PRESERVED_ENV[$name]:-}" ]]; then
    export "${name}=${PRESERVED_ENV[$name]}"
  fi
done
RESULT_DIR="${SESSION_DIR}/results"
BG_DIR="${RESULT_DIR}/3dgs_bg"
INPUT_IMAGE="${SESSION_DIR}/input/image.png"
BACKGROUND_IMAGE="${SESSION_DIR}/input/background.png"
EDIT_REPORT="${SESSION_DIR}/input/background_edit_report.json"
EXPORT_DIR="${BG_DIR}/3dgs_train_export"
TRAIN_OUT_DIR="${BG_DIR}/3dgrut_train_diy_3dgut"
PLY_PATH="${TRAIN_OUT_DIR}/export_last.ply"
KSPLAT_PATH="${BG_DIR}/background.ksplat"
FINAL_SCENE_MANIFEST="${RESULT_DIR}/final_scene_manifest.json"
DEFAULT_REFERENCE_MOGE_CACHE="${RESULT_DIR}/sam3d_moge_cache.npz"
DEFAULT_FOREGROUND_MASK="${SESSION_DIR}/input/mask_label.png"

CONDA_BIN="${CONDA_BIN:-conda}"
QWEN_ENV="${QWEN_ENV:-${FYSIVERSE_CONDA_ENV:-fysiverse-scene}}"
MOGE_ENV="${MOGE_ENV:-${FYSIVERSE_CONDA_ENV:-fysiverse-scene}}"
THREEDGRUT_ROOT="${THREEDGRUT_ROOT:-${WORKFLOW_ROOT}/third_party/3dgrut}"
GAUSSIAN_SPLATS_ROOT="${GAUSSIAN_SPLATS_ROOT:-${WORKFLOW_ROOT}/third_party/GaussianSplats3D}"
case "${THREEDGRUT_ROOT}" in /*) ;; *) THREEDGRUT_ROOT="${WORKFLOW_ROOT}/${THREEDGRUT_ROOT}" ;; esac
case "${GAUSSIAN_SPLATS_ROOT}" in /*) ;; *) GAUSSIAN_SPLATS_ROOT="${WORKFLOW_ROOT}/${GAUSSIAN_SPLATS_ROOT}" ;; esac
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ITERATIONS="${ITERATIONS:-300}"
MAX_POINTS="${MAX_POINTS:-300000}"
MOGE_RESIZE_MAX="${MOGE_RESIZE_MAX:-1024}"
MOGE_RESOLUTION_LEVEL="${MOGE_RESOLUTION_LEVEL:-5}"
MOGE_DEVICE="${MOGE_DEVICE:-cuda}"
USE_REFERENCE_MOGE_FUSION="${USE_REFERENCE_MOGE_FUSION:-1}"
REFERENCE_MOGE_CACHE="${REFERENCE_MOGE_CACHE:-}"
FOREGROUND_MASK="${FOREGROUND_MASK:-}"
FUSION_DEPTH_ALIGN="${FUSION_DEPTH_ALIGN:-scale}"
FOREGROUND_MASK_DILATE="${FOREGROUND_MASK_DILATE:-3}"
WITH_VISER_GUI="${WITH_VISER_GUI:-0}"
FORCE_EDIT="${FORCE_EDIT:-0}"
SKIP_BACKGROUND_EDIT="${SKIP_BACKGROUND_EDIT:-${SKIP_QWEN_EDIT:-0}}"
BACKGROUND_IMAGE_BACKEND="${BACKGROUND_IMAGE_BACKEND:-seeddream}"
BACKGROUND_LOCAL_FALLBACK="${BACKGROUND_LOCAL_FALLBACK:-0}"
BACKGROUND_LOCAL_INPAINT_DILATE="${BACKGROUND_LOCAL_INPAINT_DILATE:-9}"
BACKGROUND_LOCAL_INPAINT_RADIUS="${BACKGROUND_LOCAL_INPAINT_RADIUS:-5}"
KSPLAT_COMPRESSION="${KSPLAT_COMPRESSION:-2}"
KSPLAT_ALPHA_THRESHOLD="${KSPLAT_ALPHA_THRESHOLD:-5}"
KSPLAT_SCENE_CENTER="${KSPLAT_SCENE_CENTER:-0,0,0}"
KSPLAT_BLOCK_SIZE="${KSPLAT_BLOCK_SIZE:-5.0}"
KSPLAT_BUCKET_SIZE="${KSPLAT_BUCKET_SIZE:-256}"
KSPLAT_SH_DEGREE="${KSPLAT_SH_DEGREE:-0}"

if [[ ! -d "${THREEDGRUT_ROOT}" ]]; then
  echo "Missing 3DGRUT root: ${THREEDGRUT_ROOT}" >&2
  echo "Set THREEDGRUT_ROOT or config/background_3dgs.yaml: dependencies.threedgrut_root." >&2
  exit 2
fi
if [[ ! -f "${GAUSSIAN_SPLATS_ROOT}/util/create-ksplat.js" ]]; then
  echo "Missing GaussianSplats3D create-ksplat.js under ${GAUSSIAN_SPLATS_ROOT}" >&2
  echo "Set GAUSSIAN_SPLATS_ROOT or config/background_3dgs.yaml: dependencies.gaussian_splats_root." >&2
  exit 2
fi

if [[ ! -f "${INPUT_IMAGE}" ]]; then
  echo "Missing session input image: ${INPUT_IMAGE}" >&2
  exit 2
fi

CAMERA_JSON="${CAMERA_JSON:-}"
if [[ -z "${CAMERA_JSON}" ]]; then
  for candidate in \
    "${RESULT_DIR}/sam3d_moge_separated_original_input_camera.json" \
    "${RESULT_DIR}/sam3d_moge_optimized_original_input_camera.json" \
    "${RESULT_DIR}/sam3d_moge_differentiable_camera_pose_optimization.json"; do
    if [[ -f "${candidate}" ]]; then
      CAMERA_JSON="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CAMERA_JSON}" || ! -f "${CAMERA_JSON}" ]]; then
  echo "Missing optimized OOD camera JSON. Expected sam3d_moge_separated_original_input_camera.json or differentiable camera optimization report." >&2
  exit 2
fi

mkdir -p "${BG_DIR}"
echo "[session-bg-3dgs] session: ${SESSION_DIR}"
echo "[session-bg-3dgs] optimized camera: ${CAMERA_JSON}"

if [[ "${USE_REFERENCE_MOGE_FUSION}" == "0" || "${USE_REFERENCE_MOGE_FUSION}" == "false" ]]; then
  REFERENCE_MOGE_CACHE=""
  FOREGROUND_MASK=""
fi
if [[ "${USE_REFERENCE_MOGE_FUSION}" != "0" && "${USE_REFERENCE_MOGE_FUSION}" != "false" && -z "${REFERENCE_MOGE_CACHE}" && -f "${DEFAULT_REFERENCE_MOGE_CACHE}" ]]; then
  REFERENCE_MOGE_CACHE="${DEFAULT_REFERENCE_MOGE_CACHE}"
fi
if [[ "${USE_REFERENCE_MOGE_FUSION}" != "0" && "${USE_REFERENCE_MOGE_FUSION}" != "false" && -z "${FOREGROUND_MASK}" && -f "${DEFAULT_FOREGROUND_MASK}" ]]; then
  FOREGROUND_MASK="${DEFAULT_FOREGROUND_MASK}"
fi
if [[ -n "${REFERENCE_MOGE_CACHE}" && -n "${FOREGROUND_MASK}" ]]; then
  echo "[session-bg-3dgs] reference MoGe cache: ${REFERENCE_MOGE_CACHE}"
  echo "[session-bg-3dgs] foreground mask: ${FOREGROUND_MASK}"
else
  echo "[session-bg-3dgs] reference MoGe fusion disabled; missing cache or foreground mask"
  REFERENCE_MOGE_CACHE=""
  FOREGROUND_MASK=""
fi

run_local_background_fallback() {
  eval "$("${CONDA_BIN}" shell.bash hook)"
  conda activate "${QWEN_ENV}"
  python "${WORKFLOW_ROOT}/scripts/create_local_background_fallback.py" \
    --session "${SESSION_DIR}" \
    --input "${INPUT_IMAGE}" \
    --mask "${DEFAULT_FOREGROUND_MASK}" \
    --output "${BACKGROUND_IMAGE}" \
    --report "${EDIT_REPORT}" \
    --dilate "${BACKGROUND_LOCAL_INPAINT_DILATE}" \
    --radius "${BACKGROUND_LOCAL_INPAINT_RADIUS}"
  local status=$?
  conda deactivate
  return "${status}"
}

background_report_backend() {
  if [[ ! -f "${EDIT_REPORT}" ]]; then
    return 0
  fi
  python - "${EDIT_REPORT}" <<'PY'
import json
import sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    data = {}
print(data.get("backend") or "")
PY
}

BACKGROUND_REPORT_BACKEND="$(background_report_backend)"
NEED_BACKGROUND_EDIT=0
if [[ "${FORCE_EDIT}" == "1" || ! -f "${BACKGROUND_IMAGE}" ]]; then
  NEED_BACKGROUND_EDIT=1
elif [[ "${SKIP_BACKGROUND_EDIT}" != "1" && "${BACKGROUND_REPORT_BACKEND}" == "local_inpaint" && "${BACKGROUND_LOCAL_FALLBACK}" != "1" && "${BACKGROUND_LOCAL_FALLBACK}" != "true" ]]; then
  echo "[session-bg-3dgs] existing background.png came from local_inpaint; regenerating with backend=${BACKGROUND_IMAGE_BACKEND}"
  NEED_BACKGROUND_EDIT=1
fi

if [[ "${SKIP_BACKGROUND_EDIT}" != "1" && "${NEED_BACKGROUND_EDIT}" == "1" ]]; then
  echo "[session-bg-3dgs] editing foreground out of input image with backend=${BACKGROUND_IMAGE_BACKEND}"
  eval "$("${CONDA_BIN}" shell.bash hook)"
  conda activate "${QWEN_ENV}"
  set +e
  python "${WORKFLOW_ROOT}/scripts/qwen_img_edit.py" \
    --session "${SESSION_DIR}" \
    --input "${INPUT_IMAGE}" \
    --output "${BACKGROUND_IMAGE}" \
    --report "${EDIT_REPORT}" \
    --backend "${BACKGROUND_IMAGE_BACKEND}" \
    --timeout "${BACKGROUND_IMAGE_TIMEOUT:-300}"
  edit_status=$?
  set -e
  conda deactivate
  if [[ "${edit_status}" != "0" || ! -f "${BACKGROUND_IMAGE}" ]]; then
    echo "[session-bg-3dgs] online background edit failed (exit=${edit_status})"
    if [[ "${BACKGROUND_LOCAL_FALLBACK}" == "1" || "${BACKGROUND_LOCAL_FALLBACK}" == "true" ]]; then
      echo "[session-bg-3dgs] trying local inpaint background fallback"
      run_local_background_fallback
    else
      echo "[session-bg-3dgs] local fallback is disabled. Set BACKGROUND_LOCAL_FALLBACK=1 only for smoke tests." >&2
      exit "${edit_status}"
    fi
  fi
else
  echo "[session-bg-3dgs] reusing existing background image: ${BACKGROUND_IMAGE}"
fi

if [[ ! -f "${BACKGROUND_IMAGE}" ]]; then
  echo "Missing background image after edit step: ${BACKGROUND_IMAGE}" >&2
  exit 1
fi

echo "[session-bg-3dgs] training aligned background 3DGS"
RUN_DIR="${BG_DIR}" \
EXPORT_DIR="${EXPORT_DIR}" \
TRAIN_OUT_DIR="${TRAIN_OUT_DIR}" \
CAMERA_JSON="${CAMERA_JSON}" \
CONDA_BIN="${CONDA_BIN}" \
MOGE_ENV="${MOGE_ENV}" \
THREEDGRUT_ROOT="${THREEDGRUT_ROOT}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
ITERATIONS="${ITERATIONS}" \
MAX_POINTS="${MAX_POINTS}" \
MOGE_RESIZE_MAX="${MOGE_RESIZE_MAX}" \
MOGE_RESOLUTION_LEVEL="${MOGE_RESOLUTION_LEVEL}" \
MOGE_DEVICE="${MOGE_DEVICE}" \
REFERENCE_MOGE_CACHE="${REFERENCE_MOGE_CACHE}" \
FOREGROUND_MASK="${FOREGROUND_MASK}" \
FUSION_DEPTH_ALIGN="${FUSION_DEPTH_ALIGN}" \
FOREGROUND_MASK_DILATE="${FOREGROUND_MASK_DILATE}" \
WITH_VISER_GUI="${WITH_VISER_GUI}" \
"${SCRIPT_DIR}/train_single_image_3dgs.sh" "${BACKGROUND_IMAGE}" "${BG_DIR}"

if [[ ! -f "${PLY_PATH}" ]]; then
  echo "Missing trained Gaussian PLY: ${PLY_PATH}" >&2
  exit 1
fi

echo "[session-bg-3dgs] converting Gaussian PLY to ksplat"
cd "${GAUSSIAN_SPLATS_ROOT}"
if [[ ! -f build/gaussian-splats-3d.module.js ]]; then
  echo "[session-bg-3dgs] GaussianSplats3D build output missing; running npm run build"
  npm run build
fi
node util/create-ksplat.js \
  "${PLY_PATH}" \
  "${KSPLAT_PATH}" \
  "${KSPLAT_COMPRESSION}" \
  "${KSPLAT_ALPHA_THRESHOLD}" \
  "${KSPLAT_SCENE_CENTER}" \
  "${KSPLAT_BLOCK_SIZE}" \
  "${KSPLAT_BUCKET_SIZE}" \
  "${KSPLAT_SH_DEGREE}"

CREATE_KSPLAT_COMMAND="cd ${GAUSSIAN_SPLATS_ROOT} && node util/create-ksplat.js ${PLY_PATH} ${KSPLAT_PATH} ${KSPLAT_COMPRESSION} ${KSPLAT_ALPHA_THRESHOLD} \"${KSPLAT_SCENE_CENTER}\" ${KSPLAT_BLOCK_SIZE} ${KSPLAT_BUCKET_SIZE} ${KSPLAT_SH_DEGREE}"

cd "${WORKFLOW_ROOT}"
python3 "${SCRIPT_DIR}/write_session_bg_manifest.py" \
  --session "${SESSION_DIR}" \
  --bg-dir "${BG_DIR}" \
  --background-image "${BACKGROUND_IMAGE}" \
  --edit-report "${EDIT_REPORT}" \
  --export-dir "${EXPORT_DIR}" \
  --train-out-dir "${TRAIN_OUT_DIR}" \
  --ksplat "${KSPLAT_PATH}" \
  --camera-json "${CAMERA_JSON}" \
  --final-scene-manifest "${FINAL_SCENE_MANIFEST}" \
  --create-ksplat-command "${CREATE_KSPLAT_COMMAND}" \
  --compression-level "${KSPLAT_COMPRESSION}" \
  --alpha-threshold "${KSPLAT_ALPHA_THRESHOLD}" \
  --sh-degree "${KSPLAT_SH_DEGREE}"

echo "[session-bg-3dgs] done"
echo "  background: ${BACKGROUND_IMAGE}"
echo "  checkpoint: ${TRAIN_OUT_DIR}/ckpt_last.pt"
echo "  gaussian_ply: ${PLY_PATH}"
echo "  ksplat: ${KSPLAT_PATH}"
echo "  manifest: ${BG_DIR}/manifest.json"
echo "  web_scene: ${BG_DIR}/scene.json"
