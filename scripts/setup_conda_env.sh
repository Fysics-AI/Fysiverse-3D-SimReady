#!/usr/bin/env bash
set -euo pipefail

# Create a new environment for the open-source scene generation pipeline.
# This script intentionally does not modify any existing conda environment.
# Model weights are not downloaded here; set the *_ROOT and *_CKPT variables to
# existing local paths after creating the environment.

ENV_NAME="${ENV_NAME:-fysiverse-scene}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_BIN="${CONDA_BIN:-conda}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1+cu121}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1+cu121}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1+cu121}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
INSTALL_EDITABLE_THIRD_PARTY="${INSTALL_EDITABLE_THIRD_PARTY:-0}"
NVDIFFRAST_REPO="${NVDIFFRAST_REPO:-git+https://gh-proxy.com/https://github.com/NVlabs/nvdiffrast.git}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] conda env already exists: ${ENV_NAME}"
else
  echo "[setup] creating new conda env: ${ENV_NAME}"
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing uv from ${PIP_INDEX_URL}"
python -m pip install uv --index-url "${PIP_INDEX_URL}"

echo "[setup] installing PyTorch stack from official CUDA wheels: ${PYTORCH_INDEX_URL}"
python -m pip --isolated install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"

echo "[setup] installing common runtime packages with uv from ${PIP_INDEX_URL}"
uv pip install --python "$(command -v python)" --index-url "${PIP_INDEX_URL}" \
  gradio \
  pillow \
  "numpy>=1.24,<2" \
  "opencv-python<4.12" \
  scipy \
  matplotlib \
  imageio imageio-ffmpeg \
  trimesh \
  pyyaml \
  omegaconf hydra-core \
  tqdm \
  fast-simplification \
  coacd \
  sapien \
  openai \
  dashscope \
  requests \
  timm \
  transformers \
  supervision \
  pycocotools \
  addict \
  yapf \
  iopath \
  safetensors \
  einops \
  rich \
  loguru \
  seaborn \
  lightning \
  utils3d \
  gsplat

echo "[setup] installing nvdiffrast from source"
python -m pip install --no-build-isolation "${NVDIFFRAST_REPO}" || {
  echo "[setup][warn] nvdiffrast source install failed. Install it manually from NVlabs/nvdiffrast in this env."
}

if [[ "${INSTALL_EDITABLE_THIRD_PARTY}" == "1" ]]; then
  for repo in \
    "${GSAM2_ROOT:-${PROJECT_ROOT}/third_party/Grounded-SAM-2}" \
    "${SAM3D_ROOT:-${PROJECT_ROOT}/third_party/SAM3D}" \
    "${MOGE_ROOT:-${PROJECT_ROOT}/third_party/MoGe}"; do
    if [[ -d "${repo}" ]]; then
      echo "[setup] editable install: ${repo}"
      uv pip install --python "$(command -v python)" --index-url "${PIP_INDEX_URL}" -e "${repo}" || echo "[setup][warn] editable install failed: ${repo}"
    else
      echo "[setup][warn] third-party repo missing, skipped: ${repo}"
    fi
  done
fi

cat <<EOF

[setup] done

Activate:
  conda activate ${ENV_NAME}

Runtime paths are configured outside this installer in config/*.yaml.
Launch the Gradio interface with:
  ./launch_gradio_app.sh

For another config file:
  PIPELINE_CONFIG=/path/to/main.yaml ./launch_gradio_app.sh

This script does not download model weights.
EOF
