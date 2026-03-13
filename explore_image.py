"""
Explore a 16-bit Daphnia bright-field TIF image.
Generates analysis PNGs in ./results/ and prints stats.
"""

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

# ---------------------------------------------------------------------------
# 1. Load image
# ---------------------------------------------------------------------------
IMG_PATH = "/data/wei/workspace/bia-explore/data/20260210-DM-MPLX-MIX-24h_J21_w1.TIF"
OUT_DIR = "/data/wei/workspace/bia-explore/results"

img = Image.open(IMG_PATH)
raw = np.array(img, dtype=np.float64)

print("=" * 60)
print("IMAGE PROPERTIES")
print("=" * 60)
print(f"  File       : {IMG_PATH}")
print(f"  Mode       : {img.mode}")
print(f"  Size (WxH) : {img.size}")
print(f"  Dtype      : uint16")
print(f"  Min        : {raw.min():.0f}")
print(f"  Max        : {raw.max():.0f}")
print(f"  Mean       : {raw.mean():.1f}")
print(f"  Std        : {raw.std():.1f}")
print(f"  Median     : {np.median(raw):.1f}")

# Percentiles
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  P{p:02d}        : {np.percentile(raw, p):.0f}")

# ---------------------------------------------------------------------------
# Helper: normalize 16-bit -> 8-bit using percentile stretch
# ---------------------------------------------------------------------------
def to_uint8(arr, low_pct=0.5, high_pct=99.5):
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    stretched = (arr - lo) / (hi - lo)
    stretched = np.clip(stretched, 0, 1)
    return (stretched * 255).astype(np.uint8)

norm8 = to_uint8(raw)

# ---------------------------------------------------------------------------
# 2a. Original (normalized to 8-bit)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(norm8, cmap="gray")
ax.set_title("Original (contrast-stretched to 8-bit)")
ax.axis("off")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/original.png", dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_DIR}/original.png")

# ---------------------------------------------------------------------------
# 2b. Histogram
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full 16-bit histogram
axes[0].hist(raw.ravel(), bins=256, color="steelblue", edgecolor="none")
axes[0].set_title("Intensity histogram (16-bit)")
axes[0].set_xlabel("Pixel intensity")
axes[0].set_ylabel("Count")
axes[0].set_yscale("log")

# Zoomed histogram excluding saturated pixels
mask_nonsat = raw < 65535
if mask_nonsat.any():
    axes[1].hist(raw[mask_nonsat].ravel(), bins=256, color="darkorange", edgecolor="none")
    axes[1].set_title("Histogram (excluding saturated pixels)")
    axes[1].set_xlabel("Pixel intensity")
    axes[1].set_ylabel("Count")
    axes[1].set_yscale("log")

n_saturated = np.sum(raw >= 65535)
pct_saturated = 100.0 * n_saturated / raw.size
print(f"\n  Saturated pixels (==65535): {n_saturated}  ({pct_saturated:.2f}%)")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/histogram.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/histogram.png")

# ---------------------------------------------------------------------------
# 2c. Otsu thresholding (manual implementation)
# ---------------------------------------------------------------------------
def otsu_threshold(image, nbins=256):
    """Compute Otsu threshold on a flattened image array."""
    pixel_counts, bin_edges = np.histogram(image.ravel(), bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = pixel_counts.sum()
    weight1 = np.cumsum(pixel_counts)
    weight2 = total - weight1

    mean1 = np.cumsum(pixel_counts * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.sum(pixel_counts * bin_centers) - np.cumsum(pixel_counts * bin_centers)) / np.maximum(weight2, 1)

    variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
    idx = np.argmax(variance_between)
    return bin_centers[idx]

threshold = otsu_threshold(raw, nbins=512)
print(f"\n  Otsu threshold : {threshold:.0f}")

binary = (raw < threshold).astype(np.uint8)  # bright-field: object is darker
# Also try the inverse
binary_inv = (raw >= threshold).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(norm8, cmap="gray")
axes[0].set_title("Original (8-bit)")
axes[0].axis("off")

axes[1].imshow(binary, cmap="gray")
axes[1].set_title(f"Thresholded (dark objects, T={threshold:.0f})")
axes[1].axis("off")

axes[2].imshow(binary_inv, cmap="gray")
axes[2].set_title(f"Thresholded (bright objects, T={threshold:.0f})")
axes[2].axis("off")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/thresholded.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/thresholded.png")

# ---------------------------------------------------------------------------
# 2d. Edge detection (Canny-like using scipy)
# ---------------------------------------------------------------------------
# Smooth first
smoothed = ndimage.gaussian_filter(raw, sigma=2.0)

# Sobel gradients
sx = ndimage.sobel(smoothed, axis=1)
sy = ndimage.sobel(smoothed, axis=0)
magnitude = np.hypot(sx, sy)

# Threshold edges at a percentile
edge_thresh = np.percentile(magnitude, 90)
edges = (magnitude > edge_thresh).astype(np.uint8)

print(f"\n  Edge magnitude range: {magnitude.min():.1f} – {magnitude.max():.1f}")
print(f"  Edge threshold (P90): {edge_thresh:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(to_uint8(magnitude, 0, 99), cmap="inferno")
axes[0].set_title("Edge magnitude (Sobel)")
axes[0].axis("off")

axes[1].imshow(edges, cmap="gray")
axes[1].set_title(f"Edges (threshold at P90 = {edge_thresh:.0f})")
axes[1].axis("off")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/edges.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/edges.png")

# ---------------------------------------------------------------------------
# 2e. Inverted image (for bright-field)
# ---------------------------------------------------------------------------
inverted = raw.max() - raw
inv8 = to_uint8(inverted)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(norm8, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(inv8, cmap="gray")
axes[1].set_title("Inverted (bright-field correction)")
axes[1].axis("off")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/inverted.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/inverted.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Image is 16-bit grayscale, {img.size[0]}x{img.size[1]} pixels.")
print(f"  Intensity range: {raw.min():.0f} – {raw.max():.0f}")
print(f"  {pct_saturated:.2f}% pixels are saturated (65535).")
print(f"  Otsu threshold at ~{threshold:.0f} separates foreground/background.")
print(f"  All outputs saved to {OUT_DIR}/")
print("=" * 60)
