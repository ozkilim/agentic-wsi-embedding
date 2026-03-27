#!/usr/bin/env python3
"""
Prov-GigaPath Pipeline: WSI → GigaPath tile features → GigaPath slide embedding

Supports single-WSI or batch directory mode. Models are loaded once and
reused across all slides.

Usage:
  conda activate gigapath
  python run_gigapath.py --wsi_path /path/to/slide.tif
  python run_gigapath.py --wsi_dir /path/to/slides/ --output_dir ./embeddings
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import openslide
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm

WSI_EXTENSIONS = (".tif", ".svs", ".ndpi", ".mrxs", ".scn", ".bif", ".vsi")


def is_tissue(tile_rgb, bg_thresh=220, tissue_pct=0.3):
    arr = np.array(tile_rgb)
    gray = arr.mean(axis=2)
    return (gray < bg_thresh).mean() > tissue_pct


def find_target_level(slide, native_mag, target_mag):
    target_ds = native_mag / target_mag
    best_level = 0
    best_diff = float("inf")
    for i, ds in enumerate(slide.level_downsamples):
        diff = abs(ds - target_ds)
        if diff < best_diff:
            best_diff = diff
            best_level = i
    return best_level


def extract_tiles(wsi_path, target_level, patch_size, bg_thresh, tissue_pct):
    slide = openslide.OpenSlide(wsi_path)
    w, h = slide.level_dimensions[target_level]
    ds = slide.level_downsamples[target_level]

    nx, ny = w // patch_size, h // patch_size
    total = nx * ny
    print(
        f"  Level {target_level}: {w}x{h} (ds={ds:.1f}), "
        f"patch {patch_size}x{patch_size}, grid {nx}x{ny} = {total:,} patches"
    )

    tiles, coords = [], []
    for yi in tqdm(range(ny), desc="  Tiling", unit="row"):
        for xi in range(nx):
            x0 = int(xi * patch_size * ds)
            y0 = int(yi * patch_size * ds)
            tile = slide.read_region(
                (x0, y0), target_level, (patch_size, patch_size)
            ).convert("RGB")
            if is_tissue(tile, bg_thresh, tissue_pct):
                tiles.append(tile)
                coords.append([x0, y0])

    slide.close()
    coords = (
        np.array(coords, dtype=np.int64)
        if coords
        else np.zeros((0, 2), dtype=np.int64)
    )
    print(f"  Kept {len(tiles):,} tissue tiles ({100*len(tiles)/max(total,1):.1f}%)")
    return tiles, coords


def collect_wsi_paths(wsi_path=None, wsi_dir=None):
    """Return list of WSI paths from either a single path or a directory."""
    if wsi_path:
        p = Path(wsi_path)
        if not p.exists():
            print(f"ERROR: WSI not found: {p}")
            sys.exit(1)
        return [str(p)]

    if wsi_dir:
        d = Path(wsi_dir)
        if not d.is_dir():
            print(f"ERROR: Directory not found: {d}")
            sys.exit(1)
        paths = sorted(
            str(f) for f in d.iterdir() if f.suffix.lower() in WSI_EXTENSIONS
        )
        if not paths:
            print(f"ERROR: No WSI files found in {d} (looked for {WSI_EXTENSIONS})")
            sys.exit(1)
        return paths

    print("ERROR: Provide either --wsi_path or --wsi_dir")
    sys.exit(1)


def process_single_wsi(
    wsi_path, args, tile_encoder, slide_encoder, tile_transform, device
):
    """Process one WSI and return (output_path, elapsed_seconds) or (None, elapsed)."""
    wsi_name = Path(wsi_path).stem
    out_path = os.path.join(args.output_dir, f"{wsi_name}_gigapath_embedding.pt")

    if not args.overwrite and os.path.exists(out_path):
        print(f"  SKIP (already exists): {out_path}")
        return out_path, 0.0

    t0 = time.time()

    slide = openslide.OpenSlide(wsi_path)
    target_level = find_target_level(slide, args.native_mag, args.target_mag)
    actual_ds = slide.level_downsamples[target_level]
    slide.close()

    tiles, coords = extract_tiles(
        wsi_path, target_level, args.patch_size, args.bg_thresh, args.tissue_pct
    )
    if len(tiles) == 0:
        print("  WARNING: No tissue tiles found, skipping.")
        return None, time.time() - t0

    all_feats = []
    for i in tqdm(range(0, len(tiles), args.batch_size), desc="  GigaPath tiles"):
        batch = tiles[i : i + args.batch_size]
        imgs = torch.stack([tile_transform(t) for t in batch]).to(device)
        with torch.inference_mode():
            feats = tile_encoder(imgs)
        all_feats.append(feats.cpu())
        del imgs, feats
        torch.cuda.empty_cache()

    all_feats = torch.cat(all_feats, dim=0)
    print(f"  Tile features: {all_feats.shape}")

    feat_in = all_feats.unsqueeze(0).to(device)
    coords_at_extraction = coords / actual_ds
    coord_in = (
        torch.from_numpy(coords_at_extraction).unsqueeze(0).float().to(device)
    )

    with torch.inference_mode():
        outcomes = slide_encoder(feat_in, coord_in)

    slide_emb = outcomes[0]

    torch.save(
        {
            "slide_embedding": slide_emb.cpu(),
            "embedding_dim": slide_emb.shape[-1],
            "wsi_path": os.path.abspath(wsi_path),
            "wsi_name": wsi_name,
            "num_patches": len(tiles),
            "patch_size_at_target_mag": args.patch_size,
            "target_mag": args.target_mag,
            "native_mag": args.native_mag,
            "extraction_level": target_level,
        },
        out_path,
    )
    elapsed = time.time() - t0
    print(f"  Saved → {out_path}  ({elapsed:.1f}s)")
    return out_path, elapsed


def main():
    parser = argparse.ArgumentParser(description="GigaPath: WSI → slide embedding")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wsi_path", type=str, help="Path to a single WSI file")
    group.add_argument("--wsi_dir", type=str, help="Directory of WSI files")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--native_mag", type=float, default=40.0)
    parser.add_argument("--target_mag", type=float, default=20.0)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--bg_thresh", type=float, default=220)
    parser.add_argument("--tissue_pct", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-process WSIs that already have embeddings"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wsi_paths = collect_wsi_paths(args.wsi_path, args.wsi_dir)
    print(f"Device: {device}")
    print(f"WSIs to process: {len(wsi_paths)}\n")

    # Load models once
    print("[Model] Loading GigaPath tile + slide encoders ...")
    import timm

    tile_encoder = timm.create_model(
        "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
    )
    tile_encoder = tile_encoder.to(device).eval()

    from gigapath.slide_encoder import create_model as create_slide_encoder

    slide_encoder = create_slide_encoder(
        "hf_hub:prov-gigapath/prov-gigapath",
        "gigapath_slide_enc12l768d",
        1536,
    )
    slide_encoder = slide_encoder.to(device).eval()

    tile_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    print("[Model] Ready.\n")

    results = {"success": [], "skipped": [], "failed": []}
    total_time = 0.0

    for idx, wsi_path in enumerate(wsi_paths, 1):
        print(f"{'='*60}")
        print(f"[{idx}/{len(wsi_paths)}] {Path(wsi_path).name}")
        print(f"{'='*60}")
        try:
            out, elapsed = process_single_wsi(
                wsi_path, args, tile_encoder, slide_encoder, tile_transform, device
            )
            total_time += elapsed
            if out and elapsed > 0:
                results["success"].append(wsi_path)
            elif out and elapsed == 0:
                results["skipped"].append(wsi_path)
            else:
                results["failed"].append(wsi_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            results["failed"].append(wsi_path)
        print()

    # Summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Processed:  {len(results['success'])}")
    print(f"  Skipped:    {len(results['skipped'])}")
    print(f"  Failed:     {len(results['failed'])}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    if results["success"]:
        avg = total_time / len(results["success"])
        print(f"  Avg/slide:  {avg:.1f}s")
    if results["failed"]:
        print(f"\n  Failed slides:")
        for p in results["failed"]:
            print(f"    - {p}")
    print(f"\nOutput dir: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
