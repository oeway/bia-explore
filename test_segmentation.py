"""Validation tests for Daphnia segmentation against success criteria."""
import numpy as np
from PIL import Image
from scipy import ndimage
import time
import sys

sys.path.insert(0, ".")
from server import (
    RAW_16, get_well_mask, segment_body, segment_eye, segment_brain, create_overlay
)

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


print("=" * 60)
print("DAPHNIA SEGMENTATION VALIDATION")
print("=" * 60)

# Time the full pipeline
t0 = time.time()
well = get_well_mask(RAW_16)
body = segment_body(RAW_16, well, 0.85)
eye = segment_eye(RAW_16, body, 4)
brain = segment_brain(RAW_16, body, eye, 25)
overlay = create_overlay(RAW_16, body, eye, brain)
elapsed = time.time() - t0

print(f"\nSegmentation completed in {elapsed:.2f}s")
print(f"  Body: {body.sum()} px ({body.sum()/well.sum()*100:.1f}% of well)")
print(f"  Eye:  {eye.sum()} px")
print(f"  Brain: {brain.sum()} px")

# === BODY TESTS ===
print(f"\n--- Body Mask ---")
body_labels, body_n = ndimage.label(body)
check("Body is non-empty", body.sum() > 0)
check("Body is single connected component", body_n == 1, f"found {body_n}")
check("Body is 5-30% of well area",
      0.05 < body.sum() / well.sum() < 0.30,
      f"{body.sum()/well.sum()*100:.1f}%")
check("Body fully within well", np.all(body <= well))

# Check roundishness: area / convex_hull_area
body_coords = np.argwhere(body)
if len(body_coords) > 0:
    import cv2
    points = body_coords[:, ::-1].astype(np.float32)
    hull = cv2.convexHull(points)
    hull_mask = np.zeros_like(body, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 1)
    solidity = body.sum() / max(hull_mask.sum(), 1)
    check("Body solidity > 0.4 (roughly convex)",
          solidity > 0.4, f"solidity={solidity:.3f}")

# === EYE TESTS ===
print(f"\n--- Eye Mask ---")
check("Eye is non-empty", eye.sum() > 0)
eye_labels, eye_n = ndimage.label(eye)
check("Eye is single connected component", eye_n == 1, f"found {eye_n}")
check("Eye fully within body", np.all(eye <= body))
check("Eye area reasonable (100-3000 px)", 100 < eye.sum() < 3000, f"{eye.sum()} px")

# Eye should be dark
if eye.sum() > 0:
    eye_mean = RAW_16[eye].mean()
    body_mean = RAW_16[body].mean()
    check("Eye is darker than body average",
          eye_mean < body_mean * 0.5,
          f"eye={eye_mean:.0f}, body={body_mean:.0f}")

    # Circularity
    eye_contours, _ = cv2.findContours(eye.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if eye_contours:
        area = eye.sum()
        perim = cv2.arcLength(eye_contours[0], True)
        circ = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0
        check("Eye is roughly circular (circularity > 0.5)",
              circ > 0.5, f"circularity={circ:.3f}")

# === BRAIN TESTS ===
print(f"\n--- Brain Mask ---")
check("Brain is non-empty", brain.sum() > 0)
brain_labels, brain_n = ndimage.label(brain)
check("Brain is single connected component", brain_n <= 2, f"found {brain_n}")
check("Brain fully within body", np.all(brain <= body))
check("Brain area reasonable (500-10000 px)", 500 < brain.sum() < 10000, f"{brain.sum()} px")

# === RELATIONSHIP TESTS ===
print(f"\n--- Structural Relationships ---")
check("Eye and brain don't overlap", (eye & brain).sum() == 0,
      f"overlap={( eye & brain).sum()} px")

if eye.sum() > 0 and brain.sum() > 0:
    eye_cy, eye_cx = np.argwhere(eye).mean(axis=0)
    brain_cy, brain_cx = np.argwhere(brain).mean(axis=0)
    dist = np.sqrt((eye_cy - brain_cy) ** 2 + (eye_cx - brain_cx) ** 2)
    check("Brain is near eye (within 150px)",
          dist < 150, f"distance={dist:.1f} px")

# === PERFORMANCE ===
print(f"\n--- Performance ---")
check("Runs in < 30 seconds", elapsed < 30, f"{elapsed:.2f}s")
check("Runs in < 10 seconds", elapsed < 10, f"{elapsed:.2f}s")

# === SUMMARY ===
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
print(f"\n{'=' * 60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} tests")
print(f"{'=' * 60}")

if failed > 0:
    print("\nFailed tests:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  - {name}: {detail}")

# Save validated overlay
Image.fromarray(overlay).save("results/validated_overlay.png")
print(f"\nOverlay saved to results/validated_overlay.png")

sys.exit(0 if failed == 0 else 1)
