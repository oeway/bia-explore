# Daphnia Segmentation Project

## Project Vision

Segment bright field microscopy images of Daphnia (water fleas) to extract three key structures:

1. **Body** — the overall roundish body outline of the Daphnia
2. **Eye** — the compound eye (typically a dark, well-defined circular structure)
3. **Brain** — the brain region located near/behind the eye in the head

Input data: 16-bit bright field TIF images (e.g., 921x921 px).

## Segmentation Goals

- Produce per-pixel masks for each structure (body, eye, brain)
- Handle the bright field contrast characteristics (specimen darker than background, eye darkest feature)
- Robust to variation in orientation, size, and imaging conditions

## Methods to Explore

1. **Classical thresholding + morphology** — Otsu/adaptive threshold, fill holes, watershed for separating structures. Good baseline, fast, no training data needed.
2. **Edge-based / active contour** — Use gradient information to find body boundary; may work well for the roundish body shape.
3. **Pre-trained foundation models (e.g., SAM, Cellpose)** — Segment Anything or Cellpose may generalize to Daphnia without fine-tuning. Prompt-based or auto-segmentation.
4. **U-Net / deep learning** — If labeled data becomes available, train a small U-Net for multi-class segmentation.

## Validation & Success Criteria

### How we know the segmentation works

- **Visual inspection**: overlay masks on original images; body mask should tightly follow the organism boundary, eye and brain should be correctly localized with no confusion between structures.
- **Body mask**: should be a single connected component, roughly convex/roundish, covering the full organism and no background.
- **Eye mask**: should be a single compact region corresponding to the dark compound eye.
- **Brain mask**: should sit in the head region, near/behind the eye, and not extend into the body cavity or outside the body mask.
- **Containment**: eye and brain masks must be fully contained within the body mask.
- **No overlap**: eye and brain masks should not overlap each other.

### Quantitative metrics (when ground truth is available)

- IoU (Intersection over Union) per class >= 0.8
- Dice coefficient per class >= 0.85
- False positive rate for eye/brain outside body = 0

### Practical acceptance

- Method runs on a single image in < 30 seconds
- Works on at least 3 representative Daphnia images without manual parameter tuning
- Results are reproducible (deterministic or low variance across runs)

## Data

- `20260210-DM-MPLX-MIX-24h_J21_w1.TIF` — 921x921 px, 16-bit, bright field image of Daphnia

## Project Structure

- `CLAUDE.md` — this file (project description and goals)
- `segment_daphnia.py` — main segmentation script (to be created)
- `results/` — output masks and overlays (to be created)
