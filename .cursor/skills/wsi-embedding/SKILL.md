---
name: wsi-embedding
description: >-
  Embed whole-slide images (WSIs) into slide-level feature vectors using
  pathology foundation models. Supports TITAN (MahmoodLab) and Prov-GigaPath.
  Use when the user asks to embed WSIs, extract slide embeddings, run TITAN,
  run GigaPath, process pathology slides, or generate slide features.
---

# WSI Slide Embedding Pipelines

This repo contains two standalone pipelines that produce a single slide-level
embedding from whole-slide images. Both are fully self-contained — each has a
conda environment, setup script, and run script.

## Available Models

| Model | Env name | Embedding dim | Patch size (at 20x) | Script |
|-------|----------|---------------|---------------------|--------|
| **TITAN** (MahmoodLab) | `TITAN` | 768 | 512 | `titan_pipeline/run_titan.py` |
| **Prov-GigaPath** | `gigapath` | 768 | 256 | `gigapath_pipeline/run_gigapath.py` |

Both extract at **20x magnification** from 40x native WSIs by default.

## Quick Reference

### 1. First-time setup (run once per model)

```bash
# TITAN
cd titan_pipeline && bash setup.sh

# GigaPath
cd gigapath_pipeline && bash setup.sh
```

Setup is idempotent. The scripts create conda envs, install deps, and verify
HuggingFace model access. If model access fails, the user needs to request
access on HuggingFace (links are printed).

### 2. Check if environment already exists

```bash
conda env list | grep -E "TITAN|gigapath"
```

If the env exists, skip setup.

### 3. Embed a single WSI

```bash
# TITAN
conda activate TITAN
python titan_pipeline/run_titan.py --wsi_path /path/to/slide.tif --output_dir /path/to/output

# GigaPath
conda activate gigapath
python gigapath_pipeline/run_gigapath.py --wsi_path /path/to/slide.tif --output_dir /path/to/output
```

### 4. Embed a directory of WSIs (batch mode)

```bash
# TITAN
conda activate TITAN
python titan_pipeline/run_titan.py --wsi_dir /path/to/slides/ --output_dir /path/to/output

# GigaPath
conda activate gigapath
python gigapath_pipeline/run_gigapath.py --wsi_dir /path/to/slides/ --output_dir /path/to/output
```

Models load once and are reused across all slides. Already-processed slides
are skipped (use `--overwrite` to force re-processing).

### 5. Common arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--wsi_path` | — | Single WSI file (mutually exclusive with `--wsi_dir`) |
| `--wsi_dir` | — | Directory of WSIs (mutually exclusive with `--wsi_path`) |
| `--output_dir` | `./output` | Where to save `.pt` embedding files |
| `--native_mag` | `40.0` | Native magnification of slides |
| `--target_mag` | `20.0` | Target extraction magnification |
| `--batch_size` | 48 (TITAN) / 128 (GigaPath) | Tile batch size for GPU inference |
| `--overwrite` | off | Re-embed slides that already have output files |

## Output Format

Each WSI produces one `.pt` file named `{slide_stem}_{model}_embedding.pt`
containing a dict:

```python
{
    "slide_embedding": tensor,    # shape (1, embed_dim)
    "embedding_dim": int,         # 768 for both models
    "wsi_path": str,
    "wsi_name": str,
    "num_patches": int,
    "patch_size_at_target_mag": int,
    "target_mag": float,
    "native_mag": float,
    "extraction_level": int,
}
```

Load with: `data = torch.load("slide_embedding.pt"); emb = data["slide_embedding"]`

## Supported WSI Formats

`.tif`, `.svs`, `.ndpi`, `.mrxs`, `.scn`, `.bif`, `.vsi`

## Typical Runtime

On a Quadro RTX 8000 (48 GB):

| Model | Per-slide (typical) | 100 slides |
|-------|-------------------|------------|
| TITAN | ~6 min | ~10 hours |
| GigaPath | ~13 min | ~21 hours |

## Workflow for an Agent

When a user says "embed these slides with TITAN" (or GigaPath):

1. Check conda env exists: `conda env list | grep TITAN`
2. If missing, run setup: `cd titan_pipeline && bash setup.sh`
3. Run the pipeline with the user's WSI path or directory
4. Report the output directory and embedding shapes

## Known Issues

- GigaPath's internal `torchscale/architecture/config.py` may be missing
  `import numpy as np`. If you see `NameError: name 'np' is not defined`,
  add that import to the installed package file.
- GigaPath requires `xformers==0.0.28.post3` pinned to the cu124 index
  (for compatibility with `torch==2.5.1` on RTX 8000 / compute capability 7.5).
- WSIs with corrupted resolution metadata default to 40x native — override
  with `--native_mag` if needed.
