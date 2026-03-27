#!/bin/bash
# Setup for Prov-GigaPath pipeline
# Creates/activates the gigapath conda env and installs all dependencies.
# Idempotent — safe to re-run.
set -e

echo "=== Prov-GigaPath Pipeline Setup ==="

eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^gigapath "; then
    echo "Creating gigapath conda environment..."
    conda create -n gigapath python=3.10 -y
fi

conda activate gigapath

echo "Installing dependencies..."
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q "timm>=1.0.3"
pip install -q transformers huggingface_hub
pip install -q openslide-python h5py tqdm pillow numpy
pip install -q fairscale einops
pip install -q xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124

echo "Installing gigapath package from GitHub..."
pip install -q git+https://github.com/prov-gigapath/prov-gigapath.git

echo ""
echo "Verifying HuggingFace model access..."
python -c "
from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f'  HuggingFace user: {user[\"name\"]}')
print('  Checking prov-gigapath/prov-gigapath access...')
try:
    api.model_info('prov-gigapath/prov-gigapath')
    print('  OK — GigaPath model accessible')
except Exception as e:
    print(f'  FAIL — Cannot access GigaPath model: {e}')
    print('  Request access at https://huggingface.co/prov-gigapath/prov-gigapath')
    exit(1)
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo "  conda activate gigapath"
echo "  python run_gigapath.py --wsi_path /path/to/slide.tif"
echo "  python run_gigapath.py --wsi_dir /path/to/slides/ --output_dir ./embeddings"
