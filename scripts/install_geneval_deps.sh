#!/usr/bin/env bash
# scripts/install_geneval_deps.sh
# ─────────────────────────────────────────────────────────────────────────────
# Install GenEval reward model dependencies (mmdet 3.x, open_clip)
#
# Requirements:
#   - Python >= 3.10 (3.12 supported)
#   - PyTorch >= 2.0 with CUDA
#
# Usage:
#   bash scripts/install_geneval_deps.sh
#
# This script will:
#   1. Install mmdet 3.x + mmengine (via pip, no compilation needed)
#   2. Install open_clip_torch
#   3. Verify installation
#
# The Mask2Former checkpoint is auto-downloaded by mmdet on first use.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    error "PyTorch CUDA is not available. GenEval requires GPU support."
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python ${PYTHON_VERSION}, PyTorch ${TORCH_VERSION}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install mmdet 3.x + mmengine
# ─────────────────────────────────────────────────────────────────────────────
info "Step 1/2: Installing mmdet + mmengine..."
pip install -U openmim
mim install mmengine
mim install "mmdet>=3.3.0"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Install open_clip_torch
# ─────────────────────────────────────────────────────────────────────────────
info "Step 2/2: Installing open_clip_torch..."
pip install open_clip_torch

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────
info "Verifying installation..."

python -c "
import mmdet
import mmengine
import open_clip
print(f'  mmdet:     {mmdet.__version__}')
print(f'  mmengine:  {mmengine.__version__}')
print(f'  open_clip: {open_clip.__version__}')
" || {
    error "Verification failed."
    exit 1
}

info ""
info "GenEval dependencies installed successfully!"
info ""
info "Note: Mask2Former checkpoint will be auto-downloaded on first use."
info "No manual download needed (mmdet handles this via its model zoo)."
