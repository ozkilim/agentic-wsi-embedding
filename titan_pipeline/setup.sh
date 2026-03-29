#!/bin/bash
# Setup for TITAN pipeline (backed by TRIDENT)
#
# Creates the TITAN conda env and installs TRIDENT with slide-encoder support.
# TRIDENT handles: tissue segmentation, patching, CONCHv1.5 patch features,
# and TITAN slide embeddings.
#
# Idempotent — safe to re-run.
set -e

echo "=== TITAN Pipeline Setup (via TRIDENT) ==="

eval "$(conda shell.bash hook)"

# --- 1. Conda environment ---
if ! conda env list | grep -q "^TITAN "; then
    echo "Creating TITAN conda environment (Python 3.10)..."
    conda create -n TITAN python=3.10 -y
fi
conda activate TITAN

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

# Pin timm to avoid compatibility issues (TRIDENT FAQ)
pip install -q "timm==0.9.16"

# --- 3. Verify HuggingFace model access ---
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

ok = True

print('  Checking MahmoodLab/TITAN access...')
try:
    api.model_info('MahmoodLab/TITAN')
    print('    OK')
except Exception as e:
    print(f'    FAIL: {e}')
    print('    Request access: https://huggingface.co/MahmoodLab/TITAN')
    ok = False

print('  Checking MahmoodLab/conchv1_5 (CONCHv1.5) access...')
try:
    api.model_info('MahmoodLab/conchv1_5')
    print('    OK')
except Exception as e:
    print(f'    FAIL: {e}')
    print('    Request access: https://huggingface.co/MahmoodLab/conchv1_5')
    ok = False

if not ok:
    print()
    print('  Some model access checks failed. Request access on HuggingFace before running the pipeline.')
    exit(1)
"

# --- 4. Run TRIDENT doctor ---
echo ""
echo "Running TRIDENT preflight checks..."
trident-doctor --profile slide-encoders 2>/dev/null || echo "  (some checks may have warnings — see above)"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo "  conda activate TITAN"
echo "  python run_titan.py --wsi_path /path/to/slide.svs --output_dir ./output"
echo "  python run_titan.py --wsi_dir /path/to/slides/ --output_dir ./output"
echo ""
echo "Supported WSI formats:"
echo "  OpenSlide: .svs .tif .tiff .ndpi .mrxs .scn .vms .vmu"
echo "  Image:     .png .jpg .jpeg"
echo "  CuCIM:     .svs .tif .tiff  (if cucim installed)"
echo "  SDPC:      .sdpc            (if sdpc installed)"
