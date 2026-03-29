#!/bin/bash
# Setup for GigaPath pipeline (backed by TRIDENT)
#
# Creates the gigapath conda env and installs TRIDENT with slide-encoder support.
# TRIDENT handles: tissue segmentation, patching, GigaPath patch + slide features.
#
# GigaPath requires specific dependencies: flash_attn==2.5.8, fairscale.
#
# Idempotent — safe to re-run.
set -e

echo "=== GigaPath Pipeline Setup (via TRIDENT) ==="

eval "$(conda shell.bash hook)"

# --- 1. Conda environment ---
if ! conda env list | grep -q "^gigapath "; then
    echo "Creating gigapath conda environment (Python 3.10)..."
    conda create -n gigapath python=3.10 -y
fi
conda activate gigapath

# --- 2. Install TRIDENT with slide-encoder extras ---
TRIDENT_DIR="${HOME}/.local/share/trident"

if [ ! -d "$TRIDENT_DIR" ]; then
    echo "Cloning TRIDENT..."
    git clone https://github.com/mahmoodlab/trident.git "$TRIDENT_DIR"
else
    echo "Updating TRIDENT..."
    git -C "$TRIDENT_DIR" pull --ff-only 2>/dev/null || echo "  (pull skipped — local changes or detached HEAD)"
fi

echo "Installing TRIDENT [slide-encoders]..."
pip install -e "$TRIDENT_DIR[slide-encoders]"

# Pin timm (TRIDENT requires 0.9.16 for GigaPath patch encoder)
pip install -q "timm==0.9.16"

# --- 3. Install GigaPath-specific dependencies ---
echo "Installing GigaPath-specific dependencies..."
pip install -q fairscale
pip install -q flash_attn==2.5.8 2>/dev/null || echo "  WARNING: flash_attn install failed — GigaPath slide encoder may not work. Try: pip install flash_attn==2.5.8"

# Install gigapath package
pip install -q "git+https://github.com/prov-gigapath/prov-gigapath.git" 2>/dev/null || echo "  WARNING: gigapath package install failed. Install manually."

# --- 4. Verify HuggingFace model access ---
echo ""
echo "Verifying HuggingFace model access..."
python -c "
from huggingface_hub import HfApi
api = HfApi()

try:
    user = api.whoami()
    print(f'  HuggingFace user: {user[\"name\"]}')
except Exception:
    print('  WARNING: Not logged in to HuggingFace.')
    print('  Run: huggingface-cli login')
    print('  Then re-run this setup script.')
    exit(1)

print('  Checking prov-gigapath/prov-gigapath access...')
try:
    api.model_info('prov-gigapath/prov-gigapath')
    print('    OK')
except Exception as e:
    print(f'    FAIL: {e}')
    print('    Request access: https://huggingface.co/prov-gigapath/prov-gigapath')
    exit(1)
"

# --- 5. Run TRIDENT doctor ---
echo ""
echo "Running TRIDENT preflight checks..."
trident-doctor --profile slide-encoders 2>/dev/null || echo "  (some checks may have warnings — see above)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo "  conda activate gigapath"
echo "  python run_gigapath.py --wsi_path /path/to/slide.svs --output_dir ./output"
echo "  python run_gigapath.py --wsi_dir /path/to/slides/ --output_dir ./output"
echo ""
echo "Supported WSI formats:"
echo "  OpenSlide: .svs .tif .tiff .ndpi .mrxs .scn .vms .vmu"
echo "  Image:     .png .jpg .jpeg"
echo "  CuCIM:     .svs .tif .tiff  (if cucim installed)"
echo "  SDPC:      .sdpc            (if sdpc installed)"
