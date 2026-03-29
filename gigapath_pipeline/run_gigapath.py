#!/usr/bin/env python3
"""
GigaPath Pipeline: WSI → Tissue Segmentation → Patching → GigaPath Tile → GigaPath Slide Embedding

Uses TRIDENT (https://github.com/mahmoodlab/TRIDENT) for robust WSI processing.
Each slide is processed end-to-end before moving to the next:
  1. Tissue segmentation (HEST / GrandQC / Otsu)
  2. Patch coordinate extraction + sample patch QC images
  3. GigaPath patch (tile) feature extraction
  4. GigaPath slide-level feature extraction

All intermediate outputs are saved for reproducibility and downstream analysis.

Usage:
  conda activate gigapath
  python run_gigapath.py --wsi_path /path/to/slide.svs --output_dir ./output
  python run_gigapath.py --wsi_dir /path/to/slides/ --output_dir ./output
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from pathlib import Path

WSI_EXTENSIONS = {
    ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu",
    ".png", ".jpg", ".jpeg", ".sdpc", ".bif", ".vsi",
}


def check_trident():
    """Verify TRIDENT is installed and importable."""
    try:
        import trident
        return True
    except ImportError:
        print(
            "ERROR: TRIDENT is not installed.\n"
            "Run the setup script first:\n"
            "  cd gigapath_pipeline && bash setup.sh\n\n"
            "Or install manually:\n"
            "  git clone https://github.com/mahmoodlab/trident.git\n"
            "  cd trident && pip install -e '.[slide-encoders]'"
        )
        sys.exit(1)


def export_legacy_pt(h5_path, wsi_path, wsi_name, output_dir, target_mag, patch_size):
    """Export TRIDENT H5 slide embedding to legacy .pt format for backward compatibility."""
    import h5py

    if not os.path.exists(h5_path):
        return None

    with h5py.File(h5_path, "r") as f:
        features = torch.from_numpy(np.array(f["features"]))

    pt_path = os.path.join(output_dir, f"{wsi_name}_gigapath_embedding.pt")
    torch.save(
        {
            "slide_embedding": features.unsqueeze(0) if features.dim() == 1 else features,
            "embedding_dim": int(features.shape[-1]),
            "wsi_path": os.path.abspath(wsi_path),
            "wsi_name": wsi_name,
            "target_mag": target_mag,
            "patch_size": patch_size,
            "source_h5": os.path.abspath(h5_path),
        },
        pt_path,
    )
    return pt_path


def save_sample_patches(slide, patches_h5_path, save_dir, n_samples=10):
    """
    Save a grid of sample patch images for magnification QC.

    Reads n_samples evenly-spaced patches from the coordinate H5 file,
    extracts them from the WSI at full resolution, and saves:
      - Individual patch PNGs
      - A combined grid image for quick visual inspection
    """
    import h5py
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)

    with h5py.File(patches_h5_path, "r") as f:
        coords = f["coords"][:]
        patch_size_lv0 = int(f["coords"].attrs.get("patch_size_level0", 256))

    n_patches = len(coords)
    if n_patches == 0:
        return None

    n_samples = min(n_samples, n_patches)
    indices = np.linspace(0, n_patches - 1, n_samples, dtype=int)
    selected_coords = coords[indices]

    patch_images = []
    for i, (x, y) in enumerate(selected_coords):
        patch = slide.read_region(
            location=(int(x), int(y)),
            level=0,
            size=(patch_size_lv0, patch_size_lv0),
            read_as="pil",
        )
        display_size = 512
        patch_resized = patch.resize((display_size, display_size), Image.LANCZOS)
        patch_resized.save(os.path.join(save_dir, f"patch_{i:02d}_x{x}_y{y}.png"))
        patch_images.append(patch_resized)

    ncols = min(5, n_samples)
    nrows = (n_samples + ncols - 1) // ncols
    display_size = 512
    grid = Image.new("RGB", (ncols * display_size, nrows * display_size), (255, 255, 255))
    for i, img in enumerate(patch_images):
        col = i % ncols
        row = i // ncols
        grid.paste(img, (col * display_size, row * display_size))

    grid_path = os.path.join(save_dir, "_sample_grid.jpg")
    grid.save(grid_path, quality=95)
    return grid_path


def collect_wsi_paths(wsi_dir):
    """Collect all WSI file paths from a directory."""
    d = Path(wsi_dir)
    if not d.is_dir():
        print(f"ERROR: Directory not found: {d}")
        sys.exit(1)
    paths = sorted(
        str(f) for f in d.iterdir() if f.suffix.lower() in WSI_EXTENSIONS
    )
    if not paths:
        print(f"ERROR: No WSI files found in {d}")
        sys.exit(1)
    return paths


def process_slide(slide_path, args, models):
    """
    Process one WSI end-to-end: seg → patch → sample QC → GigaPath tile → GigaPath slide.

    Models dict contains pre-loaded models to avoid reloading per slide:
      - seg_model, patch_encoder, slide_encoder
    """
    from trident import load_wsi

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    job_dir = args.output_dir
    mag = int(args.target_mag)
    patch_size = args.patch_size
    coords_subdir = f"{mag}x_{patch_size}px_0px_overlap"
    slide_feat_dir = os.path.join(job_dir, coords_subdir, "slide_features_gigapath")

    load_kwargs = dict(
        slide_path=slide_path,
        lazy_init=False,
        custom_mpp_keys=args.custom_mpp_keys,
    )
    if args.mpp is not None:
        load_kwargs["mpp"] = args.mpp

    with load_wsi(**load_kwargs) as slide:

        wsi_name = slide.name
        final_h5 = os.path.join(slide_feat_dir, f"{wsi_name}.h5")
        if not args.overwrite and os.path.exists(final_h5):
            print(f"  SKIP (already exists): {final_h5}")
            return final_h5, 0.0

        t0 = time.time()

        # --- Step 1: Tissue segmentation ---
        print("  [1/4] Tissue segmentation...")
        seg_device = "cpu" if args.segmenter == "otsu" else device
        slide.segment_tissue(
            segmentation_model=models["seg_model"],
            target_mag=models["seg_model"].target_mag,
            job_dir=job_dir,
            device=seg_device,
            holes_are_tissue=not args.remove_holes,
        )
        print(f"        -> contours_geojson/{wsi_name}.geojson")

        # --- Step 2: Patch coordinate extraction ---
        print(f"  [2/4] Extracting patch coordinates ({mag}x, {patch_size}px)...")
        save_coords = os.path.join(job_dir, coords_subdir)
        coords_path = slide.extract_tissue_coords(
            target_mag=mag,
            patch_size=patch_size,
            save_coords=save_coords,
        )
        slide.visualize_coords(
            coords_path=coords_path,
            save_patch_viz=os.path.join(save_coords, "visualization"),
        )

        patches_h5 = os.path.join(save_coords, "patches", f"{wsi_name}_patches.h5")

        import h5py
        with h5py.File(patches_h5, "r") as f:
            n_patches = len(f["coords"])
        print(f"        -> {n_patches} patches")

        # --- Step 2b: Save sample patch images for QC ---
        sample_dir = os.path.join(save_coords, "sample_patches", wsi_name)
        grid_path = save_sample_patches(
            slide, patches_h5, sample_dir, n_samples=args.n_sample_patches,
        )
        if grid_path:
            print(f"        -> {args.n_sample_patches} sample patches saved for QC")

        # --- Step 3: GigaPath patch (tile) feature extraction ---
        print("  [3/4] GigaPath patch features...")
        features_dir = os.path.join(save_coords, "features_gigapath")
        slide.extract_patch_features(
            patch_encoder=models["patch_encoder"],
            coords_path=patches_h5,
            save_features=features_dir,
            device=device,
            batch_limit=args.batch_size,
        )
        print(f"        -> features_gigapath/{wsi_name}.h5")

        # --- Step 4: GigaPath slide embedding ---
        print("  [4/4] GigaPath slide embedding...")
        os.makedirs(slide_feat_dir, exist_ok=True)
        patch_feats_h5 = os.path.join(features_dir, f"{wsi_name}.h5")
        slide.extract_slide_features(
            patch_features_path=patch_feats_h5,
            slide_encoder=models["slide_encoder"],
            device=device,
            save_features=slide_feat_dir,
        )
        print(f"        -> slide_features_gigapath/{wsi_name}.h5")

    # Export backward-compatible .pt file
    if args.export_pt:
        pt_path = export_legacy_pt(
            final_h5, slide_path, wsi_name, job_dir, args.target_mag, patch_size
        )
        if pt_path:
            print(f"        -> {wsi_name}_gigapath_embedding.pt")

    elapsed = time.time() - t0
    return final_h5, elapsed


def load_models(args):
    """Load all models once, return a dict. Models are shared across slides."""
    from trident.segmentation_models import segmentation_model_factory
    from trident.patch_encoder_models import encoder_factory as patch_encoder_factory
    from trident.slide_encoder_models import encoder_factory as slide_encoder_factory

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    print("[Models] Loading segmentation model...")
    seg_model = segmentation_model_factory(
        model_name=args.segmenter,
        confidence_thresh=args.seg_conf_thresh,
    )

    print("[Models] Loading GigaPath patch (tile) encoder...")
    patch_encoder = patch_encoder_factory("gigapath")
    patch_encoder.eval()
    patch_encoder.to(device)

    print("[Models] Loading GigaPath slide encoder...")
    slide_encoder = slide_encoder_factory("gigapath")
    slide_encoder.eval()
    slide_encoder.to(device)

    print("[Models] All models loaded.\n")
    return {
        "seg_model": seg_model,
        "patch_encoder": patch_encoder,
        "slide_encoder": slide_encoder,
    }


def main():
    parser = argparse.ArgumentParser(
        description="GigaPath: WSI → slide embedding via TRIDENT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Output structure:
  output_dir/
  ├── thumbnails/                        WSI thumbnails
  ├── contours/                          Segmentation visualizations
  ├── contours_geojson/                  GeoJSON contours (editable in QuPath)
  └── {mag}x_{patch_size}px_0px_overlap/
      ├── patches/                       Patch coordinates (H5)
      ├── visualization/                 Patch border visualizations
      ├── sample_patches/{slide}/        Sample patch PNGs + grid for QC
      ├── features_gigapath/             GigaPath patch features (H5)
      └── slide_features_gigapath/       GigaPath slide embeddings (H5)

Examples:
  python run_gigapath.py --wsi_path /path/to/slide.svs --output_dir ./output
  python run_gigapath.py --wsi_dir /path/to/slides/ --output_dir ./output
  python run_gigapath.py --wsi_dir ./slides --output_dir ./output --segmenter grandqc --export_pt
""",
    )

    # --- Input ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wsi_path", type=str, help="Path to a single WSI file")
    group.add_argument("--wsi_dir", type=str, help="Directory of WSI files")

    # --- Output ---
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process WSIs that already have outputs",
    )
    parser.add_argument(
        "--export_pt",
        action="store_true",
        help="Also export slide embeddings in legacy .pt format",
    )

    # --- Magnification & patching ---
    parser.add_argument(
        "--target_mag",
        type=float,
        default=20.0,
        help="Target magnification for extraction (default: 20x)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size at target mag (default: 256, required by GigaPath)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for feature extraction (default: 64)",
    )

    # --- Tissue segmentation ---
    parser.add_argument(
        "--segmenter",
        type=str,
        default="hest",
        choices=["hest", "grandqc", "otsu"],
        help="Tissue segmentation model (default: hest)",
    )
    parser.add_argument(
        "--seg_conf_thresh",
        type=float,
        default=0.5,
        help="Segmentation confidence threshold (default: 0.5, try 0.4 for more tissue)",
    )
    parser.add_argument(
        "--remove_holes",
        action="store_true",
        help="Exclude holes within tissue regions",
    )

    # --- QC ---
    parser.add_argument(
        "--n_sample_patches",
        type=int,
        default=10,
        help="Number of sample patch images to save per slide for QC (default: 10)",
    )

    # --- Hardware & error handling ---
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index (default: 0)"
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Skip slides that cause errors and continue (batch mode)",
    )
    parser.add_argument(
        "--custom_mpp_keys",
        type=str,
        nargs="+",
        default=None,
        help="Custom WSI metadata keys for microns-per-pixel resolution",
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=None,
        help="Override microns-per-pixel (e.g. 0.25 for 40x, 0.5 for 20x). "
             "Use when WSI metadata is missing or incorrect.",
    )

    args = parser.parse_args()

    check_trident()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect slide paths
    if args.wsi_path:
        p = Path(args.wsi_path)
        if not p.exists():
            print(f"ERROR: WSI not found: {p}")
            sys.exit(1)
        wsi_paths = [str(p)]
    else:
        wsi_paths = collect_wsi_paths(args.wsi_dir)

    print(f"Output dir: {args.output_dir}")
    print(f"Slides:     {len(wsi_paths)}")
    print()

    # Load all models once
    models = load_models(args)

    # Process each slide end-to-end
    results = {"success": [], "skipped": [], "failed": []}
    total_time = 0.0

    for idx, wsi_path in enumerate(wsi_paths, 1):
        print("=" * 60)
        print(f"[{idx}/{len(wsi_paths)}] {Path(wsi_path).name}")
        print("=" * 60)
        try:
            out, elapsed = process_slide(wsi_path, args, models)
            total_time += elapsed
            if elapsed > 0:
                results["success"].append(wsi_path)
                print(f"  Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
            else:
                results["skipped"].append(wsi_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            results["failed"].append(wsi_path)
            if not args.skip_errors:
                raise
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Processed: {len(results['success'])}")
    print(f"  Skipped:   {len(results['skipped'])}")
    print(f"  Failed:    {len(results['failed'])}")
    print(f"  Total:     {total_time:.1f}s ({total_time / 60:.1f} min)")
    if results["success"]:
        avg = total_time / len(results["success"])
        print(f"  Avg/slide: {avg:.1f}s ({avg / 60:.1f} min)")
    if results["failed"]:
        print(f"\n  Failed slides:")
        for p in results["failed"]:
            print(f"    - {p}")

    mag = int(args.target_mag)
    coords_subdir = f"{mag}x_{args.patch_size}px_0px_overlap"
    print()
    print("  Output structure:")
    print(f"    {args.output_dir}/")
    print(f"    ├── thumbnails/")
    print(f"    ├── contours/")
    print(f"    ├── contours_geojson/")
    print(f"    └── {coords_subdir}/")
    print(f"        ├── patches/")
    print(f"        ├── visualization/")
    print(f"        ├── sample_patches/              # QC patch images")
    print(f"        ├── features_gigapath/")
    print(f"        └── slide_features_gigapath/")


if __name__ == "__main__":
    main()
