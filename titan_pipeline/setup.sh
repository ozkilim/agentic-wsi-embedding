#!/bin/bash
# Setup for TITAN pipeline
# Creates/activates the TITAN conda env and installs all dependencies.
# Idempotent — safe to re-run.
set -e

echo "=== TITAN Pipeline Setup ==="

eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^TITAN "; then
    echo "Creating TITAN conda environment..."
    conda create -n TITAN python=3.10 -y
fi

conda activate TITAN

echo "Installing dependencies..."
pip install -q huggingface_hub transformers h5py openslide-python tqdm pillow torch torchvision

echo ""
echo "Verifying HuggingFace model access..."
python -c "
from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f'  HuggingFace user: {user[\"name\"]}')
print('  Checking MahmoodLab/TITAN access...')
try:
    api.model_info('MahmoodLab/TITAN')
    print('  OK — TITAN model accessible')
except Exception as e:
    print(f'  FAIL — Cannot access TITAN model: {e}')
    print('  Request access at https://huggingface.co/MahmoodLab/TITAN')
    exit(1)
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo "  conda activate TITAN"
echo "  python run_titan.py --wsi_path /path/to/slide.tif"
echo "  python run_titan.py --wsi_dir /path/to/slides/ --output_dir ./embeddings"
