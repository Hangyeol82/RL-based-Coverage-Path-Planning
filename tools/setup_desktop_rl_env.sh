#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/setup_desktop_rl_env.sh [env_name]
#
# Notes for GTX 1060 6GB:
# - Uses CUDA 11.8 PyTorch wheels.
# - Falls back to CPU wheels when USE_GPU=0.

ENV_NAME="${1:-rlcpp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
USE_GPU="${USE_GPU:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] conda env '${ENV_NAME}' already exists."
else
  echo "[INFO] creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

if [[ "${USE_GPU}" == "1" ]]; then
  echo "[INFO] installing PyTorch CUDA 11.8 wheels"
  python -m pip install \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
else
  echo "[INFO] installing PyTorch CPU wheels"
  python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
fi

# Keep SB3 in a version range tested with this project.
python -m pip install \
  stable-baselines3==2.3.2 \
  gymnasium==0.29.1 \
  numpy==1.26.4 \
  matplotlib==3.8.4 \
  pandas==2.2.2

python - <<'PY'
import torch
print("[OK] torch:", torch.__version__)
print("[OK] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[OK] cuda device:", torch.cuda.get_device_name(0))
PY

echo "[DONE] environment '${ENV_NAME}' is ready."
