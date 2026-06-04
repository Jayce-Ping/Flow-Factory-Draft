#!/usr/bin/env bash
# scripts/install_geneval_deps.sh
# ─────────────────────────────────────────────────────────────────────────────
# Install GenEval reward model dependencies (mmcv + mmdet + open_clip)
#
# Requirements:
#   - Python 3.10 or 3.12 (tested)
#   - PyTorch >= 2.0 with CUDA
#   - CUDA toolkit (nvcc) for mmcv CUDA ops compilation
#   - uv (recommended) or pip
#
# Usage:
#   bash scripts/install_geneval_deps.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Prefer uv for speed; fall back to pip
if command -v uv &>/dev/null; then
    PIP="uv pip"
else
    PIP="pip"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [[ "$PY_VERSION" != "3.10" && "$PY_VERSION" != "3.12" ]]; then
    warn "Python ${PY_VERSION} detected. This script has only been tested with Python 3.10 and 3.12."
    warn "Proceeding anyway..."
    echo ""
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    error "PyTorch with CUDA is required but not available."
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
info "Python ${PY_VERSION}, PyTorch ${TORCH_VERSION}, installer: ${PIP}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install mmengine
# ─────────────────────────────────────────────────────────────────────────────
info "Step 1/4: Installing mmengine..."
$PIP install mmengine

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Compile mmcv with CUDA ops
# ─────────────────────────────────────────────────────────────────────────────
info "Step 2/4: Compiling mmcv with CUDA ops (5-10 minutes)..."

MMCV_BUILD_DIR="${REPO_ROOT}/.geneval_build"
mkdir -p "${MMCV_BUILD_DIR}"

if [ ! -d "${MMCV_BUILD_DIR}/mmcv" ]; then
    git clone --depth 1 -b v2.1.0 https://github.com/open-mmlab/mmcv.git "${MMCV_BUILD_DIR}/mmcv"
fi

export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
info "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, MAX_JOBS=${MAX_JOBS}"

$PIP install "${MMCV_BUILD_DIR}/mmcv" --no-build-isolation

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Install mmdetection
# ─────────────────────────────────────────────────────────────────────────────
info "Step 3/4: Installing mmdetection..."

if [ ! -d "${MMCV_BUILD_DIR}/mmdetection" ]; then
    git clone --depth 1 -b v3.3.0 https://github.com/open-mmlab/mmdetection.git "${MMCV_BUILD_DIR}/mmdetection"
fi

$PIP install "${MMCV_BUILD_DIR}/mmdetection" --no-build-isolation

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Install open_clip_torch
# ─────────────────────────────────────────────────────────────────────────────
info "Step 4/4: Installing open_clip_torch..."
$PIP install open_clip_torch

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────
info "Verifying installation..."

python -c "
import mmcv, mmdet, mmengine, open_clip
print(f'  mmcv:      {mmcv.__version__}')
print(f'  mmdet:     {mmdet.__version__}')
print(f'  mmengine:  {mmengine.__version__}')
print(f'  open_clip: {open_clip.__version__}')
from mmcv.ops import nms
print('  CUDA ops:  OK')
" || {
    error "Verification failed."
    exit 1
}

info ""
info "GenEval dependencies installed successfully!"
info "Build artifacts: ${MMCV_BUILD_DIR}/"
info "Mask2Former checkpoint will be auto-downloaded on first use."
