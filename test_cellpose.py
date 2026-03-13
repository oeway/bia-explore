"""Test Cellpose segmentation on a Daphnia image with multiple models/settings."""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import torch

# Paths
IMAGE_PATH = "/data/wei/workspace/bia-explore/data/20260210-DM-MPLX-MIX-24h_J21_w1.TIF"
RESULTS_DIR = "/data/wei/workspace/bia-explore/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check GPU
use_gpu = torch.cuda.is_available()
print(f"GPU available: {use_gpu}")
if use_gpu:
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Load image
img_raw = np.array(Image.open(IMAGE_PATH))
print(f"\nImage shape: {img_raw.shape}, dtype: {img_raw.dtype}")
print(f"Image range: [{img_raw.min()}, {img_raw.max()}], mean: {img_raw.mean():.1f}")

# Normalize to 0-255 uint8 for cellpose
img_norm = ((img_raw.astype(np.float64) - img_raw.min()) / (img_raw.max() - img_raw.min()) * 255).astype(np.uint8)

from cellpose.models import CellposeModel

def create_overlay(img_gray, masks, title=""):
    """Create a colored overlay of segmentation masks on the grayscale image."""
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    n_objects = masks.max()
    if n_objects > 0:
        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(n_objects + 1, 3), dtype=np.uint8)
        for i in range(1, n_objects + 1):
            mask_i = masks == i
            color = colors[i].tolist()
            overlay[mask_i] = (overlay[mask_i] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        # Draw contours
        for i in range(1, n_objects + 1):
            mask_i = (masks == i).astype(np.uint8)
            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    # Add title text
    if title:
        cv2.putText(overlay, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return overlay

def report_masks(masks, label):
    """Print statistics about detected masks."""
    n_objects = masks.max()
    print(f"    Objects found: {n_objects}")
    if n_objects > 0:
        sizes = []
        for i in range(1, n_objects + 1):
            area = int(np.sum(masks == i))
            sizes.append(area)
        sizes_sorted = sorted(sizes, reverse=True)
        print(f"    Object sizes (pixels, largest first): {sizes_sorted[:20]}")
        if len(sizes_sorted) > 20:
            print(f"    ... and {len(sizes_sorted) - 20} more objects")
        total = sum(sizes)
        pct = 100 * total / (img_raw.shape[0] * img_raw.shape[1])
        print(f"    Total coverage: {total} px ({pct:.1f}% of image)")
    return n_objects

# ---- Try legacy model names (cyto, cyto2, cyto3, nuclei, TN1) ----
legacy_models = ["cyto", "cyto2", "cyto3", "nuclei", "TN1"]
for mname in legacy_models:
    print(f"\n{'='*60}")
    print(f"Attempting legacy model: {mname}")
    print(f"{'='*60}")
    try:
        model = CellposeModel(model_type=mname, gpu=use_gpu)
        for diam in [None, 100, 200]:
            diam_label = "auto" if diam is None else f"d{diam}"
            print(f"\n  Diameter: {diam_label}")
            try:
                masks, flows, styles = model.eval(
                    img_norm,
                    diameter=diam,
                    channels=[0, 0],
                    flow_threshold=0.4,
                    cellprob_threshold=0.0,
                )
                n = report_masks(masks, f"{mname}_{diam_label}")
                overlay = create_overlay(img_norm, masks, f"{mname} diam={diam_label} n={n}")
                out_path = os.path.join(RESULTS_DIR, f"cellpose_{mname}_{diam_label}.png")
                cv2.imwrite(out_path, overlay)
                print(f"    Saved: {out_path}")
            except Exception as e:
                print(f"    Eval error: {e}")
    except Exception as e:
        print(f"  SKIPPED - could not load: {e}")

# ---- Try CPSAM (the default cellpose v4 model) ----
print(f"\n{'='*60}")
print(f"Model: cpsam (CellposeSAM v4 - default)")
print(f"{'='*60}")
try:
    model = CellposeModel(pretrained_model="cpsam", gpu=use_gpu)
    for diam in [None, 100, 200, 300]:
        diam_label = "auto" if diam is None else f"d{diam}"
        print(f"\n  Diameter: {diam_label}")
        try:
            masks, flows, styles = model.eval(
                img_norm,
                diameter=diam,
                channels=[0, 0],
                flow_threshold=0.4,
                cellprob_threshold=0.0,
            )
            n = report_masks(masks, f"cpsam_{diam_label}")
            overlay = create_overlay(img_norm, masks, f"cpsam diam={diam_label} n={n}")
            out_path = os.path.join(RESULTS_DIR, f"cellpose_cpsam_{diam_label}.png")
            cv2.imwrite(out_path, overlay)
            print(f"    Saved: {out_path}")
        except Exception as e:
            print(f"    Eval error: {e}")
except Exception as e:
    print(f"  SKIPPED - could not load: {e}")

# ---- Try CPSAM with different cellprob thresholds ----
print(f"\n{'='*60}")
print(f"Model: cpsam with varied cellprob_threshold")
print(f"{'='*60}")
try:
    model = CellposeModel(pretrained_model="cpsam", gpu=use_gpu)
    for cp_thresh in [-2.0, -4.0, -6.0]:
        print(f"\n  cellprob_threshold: {cp_thresh}, diameter: auto")
        try:
            masks, flows, styles = model.eval(
                img_norm,
                diameter=None,
                channels=[0, 0],
                flow_threshold=0.4,
                cellprob_threshold=cp_thresh,
            )
            n = report_masks(masks, f"cpsam_cp{cp_thresh}")
            overlay = create_overlay(img_norm, masks, f"cpsam cp={cp_thresh} n={n}")
            out_path = os.path.join(RESULTS_DIR, f"cellpose_cpsam_cp{cp_thresh}.png")
            cv2.imwrite(out_path, overlay)
            print(f"    Saved: {out_path}")
        except Exception as e:
            print(f"    Eval error: {e}")
except Exception as e:
    print(f"  SKIPPED: {e}")

print("\n\nDone! All results saved to:", RESULTS_DIR)
