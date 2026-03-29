---
name: wsi-embedding
description: >-
  Embed whole-slide images (WSIs) into slide-level feature vectors using
  pathology foundation models. Supports TITAN (MahmoodLab) and Prov-GigaPath.
  Use when the user asks to embed WSIs, extract slide embeddings, run TITAN,
  run GigaPath, process pathology slides, or generate slide features.
---

# WSI Slide Embedding Pipelines

This repo contains two standalone pipelines that produce slide-level
embeddings from whole-slide images.

## Available Models

| Model | Env name | Embedding dim | Pipeline | Script |
|-------|----------|---------------|----------|--------|
| **TITAN** (MahmoodLab) | `TITAN` | 768 | TRIDENT (seg → patch → CONCHv1.5 → TITAN) | `titan_pipeline/run_titan.py` |
| **Prov-GigaPath** | `gigapath` | 768 | Custom (threshold → tile → GigaPath) | `gigapath_pipeline/run_gigapath.py` |

## TITAN Pipeline (Recommended)

The TITAN pipeline uses **TRIDENT** (https://github.com/mahmoodlab/TRIDENT)
for the full processing flow, matching the official MahmoodLab workflow:

1. **Tissue segmentation** — ML-based (HEST/GrandQC) or classical (Otsu)
2. **Patch coordinate extraction** — tissue-aware patching with H5 output
3. **CONCHv1.5 patch features** — required patch encoder for TITAN
4. **TITAN slide embedding** — multimodal whole-slide foundation model

### 1. First-time setup (run once)

```bash
cd titan_pipeline && bash setup.sh
```

Setup clones TRIDENT, installs it with slide-encoder support, and verifies
HuggingFace model access for both TITAN and CONCHv1.5.

### 2. Check if environment already exists

```bash
conda env list | grep TITAN
```

### 3. Embed a single WSI

```bash
conda activate TITAN
python titan_pipeline/run_titan.py --wsi_path /path/to/slide.svs --output_dir /path/to/output
```

### 4. Embed a directory of WSIs (batch mode)

```bash
conda activate TITAN
python titan_pipeline/run_titan.py --wsi_dir /path/to/slides/ --output_dir /path/to/output
```

### 5. Common arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--wsi_path` | — | Single WSI file (mutually exclusive with `--wsi_dir`) |
| `--wsi_dir` | — | Directory of WSIs (mutually exclusive with `--wsi_path`) |
| `--output_dir` | `./output` | Where to save all outputs |
| `--target_mag` | `20.0` | Target extraction magnification |
| `--patch_size` | `512` | Patch size at target mag (512 required by TITAN) |
| `--batch_size` | `32` | Batch size for GPU inference |
| `--segmenter` | `hest` | Tissue segmenter: `hest`, `grandqc`, or `otsu` |
| `--seg_conf_thresh` | `0.5` | Segmentation confidence threshold |
| `--remove_holes` | off | Exclude holes within tissue regions |
| `--gpu` | `0` | GPU index |
| `--skip_errors` | off | Skip errored slides in batch mode |
| `--overwrite` | off | Re-process slides that already have outputs |
| `--export_pt` | off | Also save slide embeddings as legacy .pt files |
| `--custom_mpp_keys` | — | Custom metadata keys for microns-per-pixel |

## Output Format

### TITAN — Full output structure

```
output_dir/
├── thumbnails/                         # WSI thumbnails
├── contours/                           # Segmentation overlay images
├── contours_geojson/                   # GeoJSON contours (editable in QuPath)
├── 20x_512px_0px_overlap/
│   ├── patches/                        # Patch coordinates (H5)
│   │   └── {slide_name}_patches.h5
│   ├── visualization/                  # Patch grid overlays
│   ├── features_conch_v15/             # CONCHv1.5 patch features (H5)
│   │   └── {slide_name}.h5            # shape: (n_patches, 768)
│   └── slide_features_titan/           # TITAN slide embeddings (H5)
│       └── {slide_name}.h5            # shape: (768,)
└── {slide_name}_titan_embedding.pt     # (only with --export_pt)
```

Load the slide embedding:

```python
# H5 format (default)
import h5py, torch
with h5py.File("output/20x_512px_0px_overlap/slide_features_titan/slide.h5", "r") as f:
    emb = torch.from_numpy(f["features"][:])  # shape (768,)

# Legacy .pt format (with --export_pt)
data = torch.load("output/slide_titan_embedding.pt")
emb = data["slide_embedding"]  # shape (1, 768)
```

### GigaPath — Output format

Each WSI produces one `.pt` file named `{slide_stem}_gigapath_embedding.pt`:

```python
{
    "slide_embedding": tensor,    # shape (1, 768)
    "embedding_dim": int,
    "wsi_path": str,
    "wsi_name": str,
    "num_patches": int,
    "patch_size_at_target_mag": int,
    "target_mag": float,
    "native_mag": float,
    "extraction_level": int,
}
```

## Supported WSI Formats

### TITAN (via TRIDENT)
- **OpenSlide**: `.svs`, `.tif`, `.tiff`, `.ndpi`, `.mrxs`, `.scn`, `.vms`, `.vmu`
- **PIL/Image**: `.png`, `.jpg`, `.jpeg`
- **CuCIM** (if installed): `.svs`, `.tif`, `.tiff`
- **SDPC** (if installed): `.sdpc`

### GigaPath
- `.tif`, `.svs`, `.ndpi`, `.mrxs`, `.scn`, `.bif`, `.vsi`

## Workflow for an Agent

When a user says "embed these slides with TITAN":

1. Check conda env exists: `conda env list | grep TITAN`
2. If missing, run setup: `cd titan_pipeline && bash setup.sh`
3. Run the pipeline with the user's WSI path or directory
4. Report the output directory, intermediate outputs, and embedding shapes

## Known Issues

- TITAN and CONCHv1.5 are gated models on HuggingFace — the user must
  request access for both before running the pipeline.
- GigaPath's `torchscale/architecture/config.py` may need `import numpy as np`.
- If `timm != 0.9.16`, some models may fail to load.
- WSIs without magnification metadata: use `--custom_mpp_keys` or provide
  a CSV with `wsi,mpp` columns via TRIDENT's `--custom_list_of_wsis`.
