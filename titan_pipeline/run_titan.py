#!/usr/bin/env python3
"""
TITAN Pipeline: WSI → Tissue Segmentation → Patching → CONCHv1.5 → TITAN Slide Embedding

Uses TRIDENT (https://github.com/mahmoodlab/TRIDENT) for robust WSI processing.
The pipeline follows the official MahmoodLab workflow:
  1. Tissue segmentation (HEST / GrandQC / Otsu)
  2. Patch coordinate extraction with tissue-aware filtering
  3. CONCHv1.5 patch feature extraction (required by TITAN)
  4. TITAN slide-level feature extraction

All intermediate outputs are saved for reproducibility and downstream analysis.

Usage:
  conda activate TITAN
  python run_titan.py --wsi_path /path/to/slide.svs --output_dir ./output
  python run_titan.py --wsi_dir /path/to/slides/ --output_dir ./output
"""

import os
import sys
import time
import argparse
import torch
from pathlib import Path


def check_trident():
    """Verify TRIDENT is installed and importable."""
    try:
        import trident
        return True
    except ImportError:
        print(
            "ERROR: TRIDENT is not installed.\n"
            "Run the setup script first:\n"
            "  cd titan_pipeline && bash setup.sh\n\n"
            "Or install manually:\n"
            "  git clone https://github.com/mahmoodlab/trident.git\n"
            "  cd trident && pip install -e '.[slide-encoders]'"
        )
        sys.exit(1)


def export_legacy_pt(h5_path, wsi_path, wsi_name, output_dir, target_mag, patch_size):
    """Export TRIDENT H5 slide embedding to legacy .pt format for backward compatibility."""
    import h5py
    import numpy as np

    if not os.path.exists(h5_path):
        return None

    with h5py.File(h5_path, "r") as f:
        features = torch.from_numpy(np.array(f["features"]))

    pt_path = os.path.join(output_dir, f"{wsi_name}_titan_embedding.pt")
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


def process_single(args):
    """Process a single WSI through the full TITAN pipeline using TRIDENT's Slide API."""
    from trident import load_wsi
    from trident.segmentation_models import segmentation_model_factory
    from trident.patch_encoder_models import encoder_factory as patch_encoder_factory
    from trident.slide_encoder_models import encoder_factory as slide_encoder_factory

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    job_dir = args.output_dir
    mag = int(args.target_mag)
    patch_size = args.patch_size
    coords_subdir = f"{mag}x_{patch_size}px_0px_overlap"

    slide_feat_dir = os.path.join(job_dir, coords_subdir, "slide_features_titan")

    t0 = time.time()
    print(f"Processing: {args.wsi_path}")
    print(f"Output dir: {job_dir}")
    print(f"Device:     {device}")
    print()

    load_kwargs = dict(
        slide_path=args.wsi_path,
        lazy_init=False,
        custom_mpp_keys=args.custom_mpp_keys,
    )
    if args.mpp is not None:
        load_kwargs["mpp"] = args.mpp

    with load_wsi(**load_kwargs) as slide:

        wsi_name = slide.name
        final_h5 = os.path.join(slide_feat_dir, f"{wsi_name}.h5")
        if not args.overwrite and os.path.exists(final_h5):
            print(f"SKIP (already exists): {final_h5}")
            return final_h5

        # --- Step 1: Tissue segmentation ---
        print("[1/4] Tissue segmentation...")
        seg_device = "cpu" if args.segmenter == "otsu" else device
        seg_model = segmentation_model_factory(
            model_name=args.segmenter,
            confidence_thresh=args.seg_conf_thresh,
        )
        slide.segment_tissue(
            segmentation_model=seg_model,
            target_mag=seg_model.target_mag,
            job_dir=job_dir,
            device=seg_device,
            holes_are_tissue=not args.remove_holes,
        )
        del seg_model
        if "cuda" in seg_device:
            torch.cuda.empty_cache()
        print(f"  -> Contours (GeoJSON): {os.path.join(job_dir, 'contours_geojson')}")
        print(f"  -> Thumbnails:         {os.path.join(job_dir, 'contours')}")
        print()

        # --- Step 2: Patch coordinate extraction ---
        print(f"[2/4] Extracting patch coordinates (mag={mag}x, size={patch_size}px)...")
        save_coords = os.path.join(job_dir, coords_subdir)
        coords_path = slide.extract_tissue_coords(
            target_mag=mag,
            patch_size=patch_size,
            save_coords=save_coords,
        )
        print(f"  -> Coordinates: {coords_path}")

        viz_path = slide.visualize_coords(
            coords_path=coords_path,
            save_patch_viz=os.path.join(save_coords, "visualization"),
        )
        print(f"  -> Visualization: {viz_path}")
        print()

        # --- Step 3: CONCHv1.5 patch feature extraction ---
        print("[3/4] Extracting CONCHv1.5 patch features...")
        patch_encoder = patch_encoder_factory("conch_v15")
        patch_encoder.eval()
        patch_encoder.to(device)
        features_dir = os.path.join(save_coords, "features_conch_v15")
        patches_h5 = os.path.join(save_coords, "patches", f"{wsi_name}_patches.h5")
        slide.extract_patch_features(
            patch_encoder=patch_encoder,
            coords_path=patches_h5,
            save_features=features_dir,
            device=device,
            batch_limit=args.batch_size,
        )
        del patch_encoder
        torch.cuda.empty_cache()
        print(f"  -> Patch features: {features_dir}")
        print()

        # --- Step 4: TITAN slide embedding ---
        print("[4/4] Extracting TITAN slide embedding...")
        slide_encoder = slide_encoder_factory("titan")
        slide_encoder.eval()
        slide_encoder.to(device)
        os.makedirs(slide_feat_dir, exist_ok=True)
        patch_feats_h5 = os.path.join(features_dir, f"{wsi_name}.h5")
        slide.extract_slide_features(
            patch_features_path=patch_feats_h5,
            slide_encoder=slide_encoder,
            device=device,
            save_features=slide_feat_dir,
        )
        del slide_encoder
        torch.cuda.empty_cache()
        print(f"  -> Slide embedding (H5): {slide_feat_dir}")

    # Export backward-compatible .pt file
    if args.export_pt:
        pt_path = export_legacy_pt(
            final_h5, args.wsi_path, wsi_name, job_dir, args.target_mag, patch_size
        )
        if pt_path:
            print(f"  -> Legacy .pt: {pt_path}")

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"All outputs in: {os.path.abspath(job_dir)}")
    return final_h5


def process_batch(args):
    """Process a directory of WSIs through the full TITAN pipeline using TRIDENT's Processor."""
    from trident import Processor
    from trident.segmentation_models import segmentation_model_factory
    from trident.slide_encoder_models import encoder_factory as slide_encoder_factory

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    job_dir = args.output_dir
    mag = int(args.target_mag)
    patch_size = args.patch_size
    coords_subdir = f"{mag}x_{patch_size}px_0px_overlap"

    t0 = time.time()
    print(f"Batch mode: {args.wsi_dir}")
    print(f"Output dir: {job_dir}")
    print(f"Device:     {device}")
    print()

    processor = Processor(
        job_dir=job_dir,
        wsi_source=args.wsi_dir,
        skip_errors=args.skip_errors,
        custom_mpp_keys=args.custom_mpp_keys,
    )

    # --- Step 1: Tissue segmentation ---
    print("=" * 60)
    print("[1/3] TISSUE SEGMENTATION")
    print("=" * 60)
    seg_device = "cpu" if args.segmenter == "otsu" else device
    seg_model = segmentation_model_factory(
        model_name=args.segmenter,
        confidence_thresh=args.seg_conf_thresh,
    )
    processor.run_segmentation_job(
        segmentation_model=seg_model,
        seg_mag=seg_model.target_mag,
        holes_are_tissue=not args.remove_holes,
        batch_size=args.batch_size,
        device=seg_device,
    )
    del seg_model
    if "cuda" in seg_device:
        torch.cuda.empty_cache()
    print()

    # --- Step 2: Patch coordinate extraction ---
    print("=" * 60)
    print("[2/3] PATCH COORDINATE EXTRACTION")
    print("=" * 60)
    processor.run_patching_job(
        target_magnification=mag,
        patch_size=patch_size,
    )
    print()

    # --- Step 3: Slide feature extraction ---
    # TRIDENT automatically extracts CONCHv1.5 patch features if missing,
    # then runs the TITAN slide encoder on top.
    print("=" * 60)
    print("[3/3] TITAN SLIDE EMBEDDING (+ CONCHv1.5 patch features)")
    print("=" * 60)
    slide_encoder = slide_encoder_factory("titan")
    processor.run_slide_feature_extraction_job(
        slide_encoder=slide_encoder,
        coords_dir=coords_subdir,
        device=device,
        batch_limit=args.batch_size,
    )
    del slide_encoder
    torch.cuda.empty_cache()

    # Export backward-compatible .pt files
    if args.export_pt:
        slide_feat_dir = os.path.join(job_dir, coords_subdir, "slide_features_titan")
        if os.path.isdir(slide_feat_dir):
            print()
            print("Exporting legacy .pt files...")
            for h5_name in sorted(os.listdir(slide_feat_dir)):
                if not h5_name.endswith(".h5"):
                    continue
                wsi_name = h5_name[: -len(".h5")]
                h5_path = os.path.join(slide_feat_dir, h5_name)
                wsi_path = ""
                for slide in processor.wsis:
                    if slide.name == wsi_name:
                        wsi_path = slide.slide_path if hasattr(slide, "slide_path") else ""
                        break
                pt_path = export_legacy_pt(
                    h5_path, wsi_path, wsi_name, job_dir, args.target_mag, patch_size
                )
                if pt_path:
                    print(f"  {pt_path}")

    processor.release()

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Output dir: {os.path.abspath(job_dir)}")
    print()
    print("  Output structure:")
    print(f"    {job_dir}/")
    print(f"    ├── thumbnails/")
    print(f"    ├── contours/                        # segmentation overlays")
    print(f"    ├── contours_geojson/                # editable in QuPath")
    print(f"    └── {coords_subdir}/")
    print(f"        ├── patches/                     # patch coordinates (H5)")
    print(f"        ├── visualization/               # patch grid overlays")
    print(f"        ├── features_conch_v15/          # CONCHv1.5 patch features (H5)")
    print(f"        └── slide_features_titan/        # TITAN slide embeddings (H5)")


def main():
    parser = argparse.ArgumentParser(
        description="TITAN: WSI → slide embedding via TRIDENT",
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
      ├── features_conch_v15/            CONCHv1.5 patch features (H5)
      └── slide_features_titan/          TITAN slide embeddings (H5)

Examples:
  python run_titan.py --wsi_path /path/to/slide.svs --output_dir ./output
  python run_titan.py --wsi_dir /path/to/slides/ --output_dir ./output
  python run_titan.py --wsi_dir ./slides --output_dir ./output --segmenter grandqc --export_pt
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
        default=512,
        help="Patch size at target mag (default: 512, required by TITAN/CONCHv1.5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction (default: 32)",
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

    if args.wsi_path:
        process_single(args)
    else:
        process_batch(args)


if __name__ == "__main__":
    main()
