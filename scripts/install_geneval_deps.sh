#!/usr/bin/env bash
# scripts/install_geneval_deps.sh
# ─────────────────────────────────────────────────────────────────────────────
# Install GenEval reward model dependencies (mmdet 3.x, open_clip)
#
# Requirements:
#   - Python >= 3.10 (3.12 supported)
#   - PyTorch >= 2.0 with CUDA
#   - uv (recommended) or pip
#
# Usage:
#   bash scripts/install_geneval_deps.sh
#
# This script installs mmdet/mmengine/mmcv WITHOUT openmim (which is broken
# on Python 3.12 due to pkg_resources deprecation). Instead it uses prebuilt
# mmcv wheels from OpenMMLab's CDN, selected by auto-detected torch+CUDA version.
#
# The Mask2Former checkpoint is auto-downloaded by mmdet on first use.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Prefer uv for speed; fall back to pip
if command -v uv &>/dev/null; then
    PIP="uv pip"
else
    PIP="pip"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python ${PYTHON_VERSION}, installer: ${PIP}"

if ! python -c "import torch" 2>/dev/null; then
    error "PyTorch is not installed."
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; v=torch.__version__.split('+')[0].rsplit('.',1)[0]; print(v)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu')")

info "Detected: torch${TORCH_VERSION}, cu${CUDA_VERSION}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install mmcv from prebuilt wheels (bypasses broken openmim on 3.12)
# ─────────────────────────────────────────────────────────────────────────────
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu${CUDA_VERSION}/torch${TORCH_VERSION}/index.html"
info "Step 1/3: Installing mmcv from ${MMCV_INDEX} ..."
$PIP install mmcv --find-links "${MMCV_INDEX}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Install mmengine + mmdet
# ─────────────────────────────────────────────────────────────────────────────
info "Step 2/3: Installing mmengine + mmdet..."
$PIP install mmengine mmdet

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Install open_clip_torch
# ─────────────────────────────────────────────────────────────────────────────
info "Step 3/3: Installing open_clip_torch..."
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
" || {
    error "Verification failed."
    exit 1
}

info ""
info "GenEval dependencies installed successfully!"
info "Mask2Former checkpoint will be auto-downloaded on first use."
