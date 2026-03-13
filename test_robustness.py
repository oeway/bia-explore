"""Test segmentation robustness across multiple images."""
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
import time
import sys
from pathlib import Path

sys.path.insert(0, ".")
from server import get_well_mask, segment_body, segment_eye, segment_brain, create_overlay, to_uint8

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

image_files = sorted(DATA_DIR.glob("*.TIF"))
print(f"Found {len(image_files)} images to test\n")

all_results = []

for img_path in image_files:
    name = img_path.stem
    print(f"{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    img = np.array(Image.open(img_path))
    print(f"  Shape: {img.shape}, range: [{img.min()}, {img.max()}]")

    t0 = time.time()
    try:
        well = get_well_mask(img)
        body = segment_body(img, well, 0.85)
        eye = segment_eye(img, body, 4)
        brain = segment_brain(img, body, eye, 18)
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results.append({"name": name, "error": str(e)})
        continue

    # Collect metrics
    results = {"name": name, "time": elapsed, "passed": 0, "failed": 0, "tests": []}

    def check(test_name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        results["tests"].append((test_name, status, detail))
        if condition:
            results["passed"] += 1
        else:
            results["failed"] += 1
        print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))

    well_area = well.sum()
    body_area = body.sum()
    eye_area = eye.sum()
    brain_area = brain.sum()

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Well: {well_area} px, Body: {body_area} px, Eye: {eye_area} px, Brain: {brain_area} px")

    # Body tests
    check("Body non-empty", body_area > 0)
    body_labels, body_n = ndimage.label(body)
    check("Body single component", body_n == 1, f"{body_n} components")
    check("Body 5-30% of well", 0.05 < body_area / well_area < 0.30,
          f"{body_area/well_area*100:.1f}%")
    check("Body within well", np.all(body <= well))

    # Eye tests
    check("Eye non-empty", eye_area > 0)
    check("Eye within body", np.all(eye <= body))
    check("Eye area 50-3000 px", 50 < eye_area < 3000, f"{eye_area} px")
    if eye_area > 0:
        eye_mean = img[eye].mean()
        body_mean = img[body].mean()
        check("Eye darker than body", eye_mean < body_mean * 0.6,
              f"eye={eye_mean:.0f} vs body={body_mean:.0f}")
        # Circularity
        cc, _ = cv2.findContours(eye.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cc:
            perim = cv2.arcLength(cc[0], True)
            circ = 4 * np.pi * eye_area / (perim ** 2) if perim > 0 else 0
            check("Eye circular (>0.5)", circ > 0.5, f"{circ:.3f}")

    # Brain tests
    check("Brain non-empty", brain_area > 0)
    check("Brain within body", np.all(brain <= body))
    check("Brain area 200-15000 px", 200 < brain_area < 15000, f"{brain_area} px")

    # Relationships
    check("No eye-brain overlap", (eye & brain).sum() == 0)
    if eye_area > 0 and brain_area > 0:
        ey, ex = np.argwhere(eye).mean(axis=0)
        by, bx = np.argwhere(brain).mean(axis=0)
        dist = np.sqrt((ey - by) ** 2 + (ex - bx) ** 2)
        check("Brain near eye (<200px)", dist < 200, f"{dist:.1f} px")

    check("Speed < 30s", elapsed < 30, f"{elapsed:.2f}s")

    # Save overlay
    overlay = create_overlay(img, body, eye, brain)
    Image.fromarray(overlay).save(str(RESULTS_DIR / f"robustness_{name}.png"))

    all_results.append(results)
    print()

# Summary
print("=" * 60)
print("ROBUSTNESS SUMMARY")
print("=" * 60)
total_pass = sum(r.get("passed", 0) for r in all_results)
total_fail = sum(r.get("failed", 0) for r in all_results)
total_error = sum(1 for r in all_results if "error" in r)

for r in all_results:
    if "error" in r:
        print(f"  {r['name']}: ERROR — {r['error']}")
    else:
        status = "ALL PASS" if r["failed"] == 0 else f"{r['failed']} FAILED"
        print(f"  {r['name']}: {r['passed']}/{r['passed']+r['failed']} ({status}, {r['time']:.2f}s)")

print(f"\nTotal: {total_pass} passed, {total_fail} failed, {total_error} errors")
print(f"Images: {len(all_results) - total_error}/{len(all_results)} segmented successfully")

# Detailed failures
if total_fail > 0:
    print(f"\nFailed tests:")
    for r in all_results:
        for test_name, status, detail in r.get("tests", []):
            if status == "FAIL":
                print(f"  [{r['name']}] {test_name}: {detail}")

sys.exit(0 if total_fail == 0 and total_error == 0 else 1)
