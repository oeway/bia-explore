"""Generate synthetic Daphnia-like test images to validate segmentation robustness.

Creates images with:
- Circular well (bright background)
- Elliptical body (semi-transparent, darker than background)
- Dark circular eye
- Moderately dark brain region near eye
- Thin appendages (antennae, legs)
- Varying orientation, size, position, and contrast
"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def make_daphnia_image(
    size=921,
    well_radius=420,
    body_center=(0.0, 0.0),  # relative to well center, in fraction of radius
    body_size=(0.35, 0.55),  # semi-axes as fraction of well radius
    body_angle=30,  # degrees
    eye_offset=(0.15, 0.25),  # from body center, fraction of well radius
    eye_radius_frac=0.04,
    brain_offset=(-0.02, 0.18),
    brain_size=(0.08, 0.06),
    bg_intensity=48000,
    body_darkness=0.7,  # body intensity as fraction of background
    eye_darkness=0.05,
    brain_darkness=0.35,
    noise_std=2000,
    seed=42,
):
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size), dtype=np.float64)

    cy, cx = size // 2, size // 2
    y, x = np.mgrid[0:size, 0:size]

    # Well: bright circular area
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    well = dist_from_center < well_radius
    # Smooth well edge
    well_smooth = np.clip(1.0 - (dist_from_center - well_radius + 15) / 30, 0, 1)
    img = bg_intensity * well_smooth

    # Outside well: very dark
    img[~well] = rng.normal(500, 200, (~well).sum()).clip(0, 2000)

    # Body center in absolute coords
    body_cy = cy + body_center[0] * well_radius
    body_cx = cx + body_center[1] * well_radius
    a, b = body_size[0] * well_radius, body_size[1] * well_radius

    # Rotated ellipse for body
    angle_rad = np.radians(body_angle)
    dx = x - body_cx
    dy = y - body_cy
    rx = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
    ry = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    body_dist = (rx / a) ** 2 + (ry / b) ** 2
    body_mask = body_dist < 1.0

    # Body is semi-transparent: slightly darker than background
    body_factor = np.ones_like(img)
    body_smooth = np.clip(1.0 - (body_dist - 0.85) / 0.3, 0, 1)
    body_factor = 1.0 - (1.0 - body_darkness) * body_smooth
    img *= body_factor

    # Internal texture: random dark patches simulating organs
    organ_noise = gaussian_filter(rng.randn(size, size), sigma=20) * 5000
    img[body_mask] += organ_noise[body_mask] * (1 - body_dist[body_mask])

    # Eye: dark circle
    eye_cy = body_cy + eye_offset[0] * well_radius
    eye_cx = body_cx + eye_offset[1] * well_radius
    eye_r = eye_radius_frac * well_radius
    eye_dist = np.sqrt((y - eye_cy) ** 2 + (x - eye_cx) ** 2)
    eye_mask = eye_dist < eye_r
    eye_smooth = np.clip(1.0 - (eye_dist - eye_r + 3) / 6, 0, 1)
    img *= 1.0 - (1.0 - eye_darkness) * eye_smooth

    # Brain: elliptical dark region near eye
    brain_cy = eye_cy + brain_offset[0] * well_radius
    brain_cx = eye_cx + brain_offset[1] * well_radius
    ba, bb = brain_size[0] * well_radius, brain_size[1] * well_radius
    brain_rx = (dx_b := x - brain_cx) * np.cos(angle_rad) + (dy_b := y - brain_cy) * np.sin(angle_rad)
    brain_ry = -dx_b * np.sin(angle_rad) + dy_b * np.cos(angle_rad)
    brain_dist = (brain_rx / ba) ** 2 + (brain_ry / bb) ** 2
    brain_smooth = np.clip(1.0 - (brain_dist - 0.7) / 0.6, 0, 1)
    img *= 1.0 - (1.0 - brain_darkness) * brain_smooth * body_smooth

    # Appendages: thin dark lines
    for _ in range(6):
        # Random line from body edge
        angle = rng.uniform(0, 2 * np.pi)
        start_y = body_cy + b * 0.8 * np.sin(angle)
        start_x = body_cx + a * 0.8 * np.cos(angle)
        length = rng.uniform(50, 150)
        end_y = start_y + length * np.sin(angle + rng.uniform(-0.3, 0.3))
        end_x = start_x + length * np.cos(angle + rng.uniform(-0.3, 0.3))

        # Distance from line segment
        line_len = np.sqrt((end_y - start_y) ** 2 + (end_x - start_x) ** 2)
        if line_len < 1:
            continue
        t = np.clip(
            ((y - start_y) * (end_y - start_y) + (x - start_x) * (end_x - start_x)) / line_len ** 2,
            0, 1,
        )
        proj_y = start_y + t * (end_y - start_y)
        proj_x = start_x + t * (end_x - start_x)
        dist_to_line = np.sqrt((y - proj_y) ** 2 + (x - proj_x) ** 2)
        appendage = np.clip(1.0 - dist_to_line / 3, 0, 1)
        img *= 1.0 - 0.4 * appendage * well_smooth

    # Add noise
    img += rng.normal(0, noise_std, img.shape)
    img = np.clip(img, 0, 65535).astype(np.uint16)

    return img


# Generate test images with different configurations
configs = [
    {
        "name": "test_centered",
        "body_center": (-0.1, 0.0),
        "body_angle": 30,
        "body_size": (0.3, 0.5),
        "seed": 42,
    },
    {
        "name": "test_rotated",
        "body_center": (0.05, -0.1),
        "body_angle": -45,
        "body_size": (0.3, 0.5),
        "seed": 123,
    },
    {
        "name": "test_small",
        "body_center": (0.15, 0.15),
        "body_angle": 60,
        "body_size": (0.2, 0.35),
        "eye_radius_frac": 0.03,
        "seed": 456,
    },
    {
        "name": "test_large",
        "body_center": (-0.05, -0.05),
        "body_angle": 10,
        "body_size": (0.4, 0.6),
        "eye_radius_frac": 0.05,
        "brain_size": (0.1, 0.08),
        "seed": 789,
    },
    {
        "name": "test_low_contrast",
        "body_center": (0.0, 0.1),
        "body_angle": -20,
        "body_darkness": 0.82,
        "eye_darkness": 0.1,
        "brain_darkness": 0.5,
        "seed": 321,
    },
]

for cfg in configs:
    name = cfg.pop("name")
    img = make_daphnia_image(**cfg)
    path = DATA_DIR / f"{name}.TIF"
    Image.fromarray(img).save(str(path))
    print(f"Generated {path}: shape={img.shape}, range=[{img.min()}, {img.max()}]")

print(f"\nGenerated {len(configs)} test images in {DATA_DIR}/")
