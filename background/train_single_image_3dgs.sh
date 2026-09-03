#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/image.png [run_dir]" >&2
  exit 2
fi

IMAGE_PATH="$(realpath "$1")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STEM="$(basename "${IMAGE_PATH}")"
STEM="${STEM%.*}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${2:-${RUN_DIR:-${SCRIPT_DIR}/outputs/${STEM}_${RUN_ID}}}"
EXPORT_DIR="${EXPORT_DIR:-${RUN_DIR}/3dgs_train_export}"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-${RUN_DIR}/3dgrut_train_diy_3dgut}"

CONDA_BIN="${CONDA_BIN:-conda}"
MOGE_ENV="${MOGE_ENV:-${FYSIVERSE_CONDA_ENV:-fysiverse-scene}}"
THREEDGRUT_ROOT="${THREEDGRUT_ROOT:-${WORKFLOW_ROOT}/third_party/3dgrut}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ITERATIONS="${ITERATIONS:-300}"
MOGE_RESIZE_MAX="${MOGE_RESIZE_MAX:-1024}"
MOGE_RESOLUTION_LEVEL="${MOGE_RESOLUTION_LEVEL:-5}"
MOGE_DEVICE="${MOGE_DEVICE:-cuda}"
MOGE_ROOT="${MOGE_ROOT:-${WORKFLOW_ROOT}/third_party/MoGe}"
MOGE_CKPT="${MOGE_CKPT:-${WORKFLOW_ROOT}/checkpoints/moge-2-vitl-normal/model.pt}"
case "${THREEDGRUT_ROOT}" in /*) ;; *) THREEDGRUT_ROOT="${WORKFLOW_ROOT}/${THREEDGRUT_ROOT}" ;; esac
case "${MOGE_ROOT}" in /*) ;; *) MOGE_ROOT="${WORKFLOW_ROOT}/${MOGE_ROOT}" ;; esac
case "${MOGE_CKPT}" in /*) ;; *) MOGE_CKPT="${WORKFLOW_ROOT}/${MOGE_CKPT}" ;; esac
MAX_POINTS="${MAX_POINTS:-300000}"
CAMERA_JSON="${CAMERA_JSON:-}"
REFERENCE_MOGE_CACHE="${REFERENCE_MOGE_CACHE:-}"
FOREGROUND_MASK="${FOREGROUND_MASK:-}"
FUSION_DEPTH_ALIGN="${FUSION_DEPTH_ALIGN:-scale}"
FOREGROUND_MASK_DILATE="${FOREGROUND_MASK_DILATE:-3}"
VISER_HOST="${VISER_HOST:-0.0.0.0}"
VISER_PORT="${VISER_PORT:-8081}"
WITH_VISER_GUI="${WITH_VISER_GUI:-1}"

if [[ ! -d "${THREEDGRUT_ROOT}" ]]; then
  echo "Missing 3DGRUT root: ${THREEDGRUT_ROOT}" >&2
  echo "Set THREEDGRUT_ROOT or config/background_3dgs.yaml: dependencies.threedgrut_root." >&2
  exit 2
fi
if [[ ! -f "${THREEDGRUT_ROOT}/train_diy.py" ]]; then
  echo "Missing 3DGRUT train_diy.py under ${THREEDGRUT_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"

echo "[single-image-3dgs] preparing MoGe export"
echo "  image: ${IMAGE_PATH}"
echo "  export_dir: ${EXPORT_DIR}"
if [[ -n "${CAMERA_JSON}" ]]; then
  echo "  camera_json: ${CAMERA_JSON}"
fi
if [[ -n "${REFERENCE_MOGE_CACHE}" ]]; then
  echo "  reference_moge_cache: ${REFERENCE_MOGE_CACHE}"
fi
if [[ -n "${FOREGROUND_MASK}" ]]; then
  echo "  foreground_mask: ${FOREGROUND_MASK}"
fi

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${MOGE_ENV}"
PREPARE_ARGS=(
  python "${SCRIPT_DIR}/prepare_single_image_3dgs.py"
  --image "${IMAGE_PATH}" \
  --export-dir "${EXPORT_DIR}" \
  --device "${MOGE_DEVICE}" \
  --resize-max "${MOGE_RESIZE_MAX}" \
  --resolution-level "${MOGE_RESOLUTION_LEVEL}" \
  --max-points "${MAX_POINTS}" \
  --moge-root "${MOGE_ROOT}" \
  --checkpoint "${MOGE_CKPT}"
)
if [[ -n "${CAMERA_JSON}" ]]; then
  PREPARE_ARGS+=(--camera-json "${CAMERA_JSON}")
fi
if [[ -n "${REFERENCE_MOGE_CACHE}" ]]; then
  PREPARE_ARGS+=(--reference-moge-cache "${REFERENCE_MOGE_CACHE}")
fi
if [[ -n "${FOREGROUND_MASK}" ]]; then
  PREPARE_ARGS+=(--foreground-mask "${FOREGROUND_MASK}")
fi
PREPARE_ARGS+=(
  --fusion-depth-align "${FUSION_DEPTH_ALIGN}"
  --foreground-mask-dilate "${FOREGROUND_MASK_DILATE}"
)
"${PREPARE_ARGS[@]}"
conda deactivate

echo "[single-image-3dgs] training 3DGUT"
echo "  train_out_dir: ${TRAIN_OUT_DIR}"

cd "${THREEDGRUT_ROOT}"
if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing 3DGRUT virtualenv: ${THREEDGRUT_ROOT}/.venv" >&2
  echo "Install 3DGRUT and set THREEDGRUT_ROOT to a runnable checkout." >&2
  exit 2
fi
source .venv/bin/activate

TRAIN_ARGS=(
  python train_diy.py
  --iterations "${ITERATIONS}"
  --export-dir "${EXPORT_DIR}"
  --out-dir "${TRAIN_OUT_DIR}"
  --viser-host "${VISER_HOST}"
  --viser-port "${VISER_PORT}"
)
if [[ "${WITH_VISER_GUI}" == "1" || "${WITH_VISER_GUI}" == "true" ]]; then
  TRAIN_ARGS+=(--with-viser-gui)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${TRAIN_ARGS[@]}"

echo "[single-image-3dgs] done"
echo "  checkpoint: ${TRAIN_OUT_DIR}/ckpt_last.pt"
echo "  gaussian_ply: ${TRAIN_OUT_DIR}/export_last.ply"
echo "  export_dir: ${EXPORT_DIR}"
