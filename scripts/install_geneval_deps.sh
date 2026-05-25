#!/usr/bin/env bash
# scripts/install_geneval_deps.sh
# ─────────────────────────────────────────────────────────────────────────────
# Install GenEval reward model dependencies (mmcv, mmdetection, open_clip, etc.)
#
# Requirements:
#   - Python 3.10 (mmcv 1.x does NOT support Python 3.11+)
#   - PyTorch >= 2.0 with CUDA (for MMCV_WITH_OPS)
#   - CUDA toolkit matching your PyTorch build
#
# Usage:
#   bash scripts/install_geneval_deps.sh [VENV_PATH]
#
# Arguments:
#   VENV_PATH   Path to the virtual environment (default: .venv)
#
# This script will:
#   1. Install openmim + mmengine
#   2. Clone and install mmcv 1.x with CUDA ops
#   3. Clone and install mmdetection 2.x
#   4. Install open_clip_torch and clip_benchmark
#   5. Download the Mask2Former COCO checkpoint
#
# Note: This script is designed for the GenEval integration in Flow-Factory.
#       It assumes you are running from the Flow-Factory repository root.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${1:-.venv}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

# Check if venv exists
if [[ -d "${VENV_PATH}" ]]; then
    info "Using existing virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
else
    error "Virtual environment not found at ${VENV_PATH}"
    error "Create one first with: python3.10 -m venv ${VENV_PATH}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "${PYTHON_VERSION}" != "3.10" ]]; then
    warn "Python ${PYTHON_VERSION} detected. GenEval (mmcv 1.x) requires Python 3.10."
    warn "Proceeding anyway, but compilation may fail."
fi

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    error "PyTorch CUDA is not available. GenEval requires GPU support."
    error "Install PyTorch with CUDA first."
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
info "PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install openmim + mmengine
# ─────────────────────────────────────────────────────────────────────────────
info "Step 1/5: Installing openmim and mmengine..."
pip install -U openmim
mim install mmengine

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Install mmcv 1.x with CUDA ops
# ─────────────────────────────────────────────────────────────────────────────
info "Step 2/5: Installing mmcv 1.x (this may take a while, compiling CUDA ops)..."

GENEVAL_BUILD_DIR="${REPO_ROOT}/.geneval_build"
mkdir -p "${GENEVAL_BUILD_DIR}"

if [[ ! -d "${GENEVAL_BUILD_DIR}/mmcv" ]]; then
    git clone https://github.com/open-mmlab/mmcv.git "${GENEVAL_BUILD_DIR}/mmcv"
fi
cd "${GENEVAL_BUILD_DIR}/mmcv"
git checkout 1.x
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e . -v

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Install mmdetection 2.x
# ─────────────────────────────────────────────────────────────────────────────
info "Step 3/5: Installing mmdetection 2.x..."

if [[ ! -d "${GENEVAL_BUILD_DIR}/mmdetection" ]]; then
    git clone https://github.com/open-mmlab/mmdetection.git "${GENEVAL_BUILD_DIR}/mmdetection"
fi
cd "${GENEVAL_BUILD_DIR}/mmdetection"
git checkout 2.x
pip install -e . -v

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Install open_clip and clip_benchmark
# ─────────────────────────────────────────────────────────────────────────────
info "Step 4/5: Installing open_clip_torch and clip_benchmark..."
pip install open_clip_torch clip_benchmark

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Download Mask2Former checkpoint
# ─────────────────────────────────────────────────────────────────────────────
info "Step 5/5: Downloading Mask2Former COCO checkpoint..."

CKPT_DIR="${REPO_ROOT}/reward_ckpts"
mkdir -p "${CKPT_DIR}"

CKPT_NAME="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
CKPT_URL="https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/${CKPT_NAME}"

if [[ -f "${CKPT_DIR}/${CKPT_NAME}" ]]; then
    info "Checkpoint already exists: ${CKPT_DIR}/${CKPT_NAME}"
else
    info "Downloading checkpoint to ${CKPT_DIR}/"
    wget -q --show-progress -P "${CKPT_DIR}" "${CKPT_URL}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────
info "Verifying installation..."

python -c "
import mmcv
import mmdet
import open_clip
from clip_benchmark.metrics import zeroshot_classification
print(f'  mmcv:       {mmcv.__version__}')
print(f'  mmdet:      {mmdet.__version__}')
print(f'  open_clip:  {open_clip.__version__}')
print('  clip_benchmark: OK')
" || {
    error "Verification failed. Check the errors above."
    exit 1
}

info ""
info "GenEval dependencies installed successfully!"
info ""
info "Build artifacts are in: ${GENEVAL_BUILD_DIR}/"
info "Checkpoint saved to:    ${CKPT_DIR}/${CKPT_NAME}"
info ""
info "Usage in training config:"
info "  rewards:"
info "    - name: \"geneval\""
info "      reward_model: \"geneval\""
info "      batch_size: 32"
info "      device: \"cuda\""
info "      ckpt_path: \"${CKPT_DIR}\""
info "      reward_type: \"score\"  # or 'strict'"
