# Daphnia Segmentation — Exploration Plan

## Phase 1: Data Exploration (done)
- [x] Load and inspect the TIF image (921x921, 16-bit bright field)
- [x] Generate intensity histogram — found bimodal distribution (dark organism + bright well background)
- [x] Test Otsu thresholding — segments body but merges with dark well border
- [x] Run edge detection — Sobel edges clearly reveal body outline, eye, internal structures
- [x] Generate inverted image — makes structures easier to distinguish

### Key Observations
- Image is a Daphnia inside a circular well; outside the well is dark
- The **eye** is the darkest compact feature inside the body
- The **brain** is a moderately dark region near/behind the eye in the head
- Otsu alone won't work — need to first isolate the well, then segment within it
- Edge map is very informative; could guide contour-based approaches

## Phase 2: Interactive Viewer (done)
- [x] FastAPI server serving original image as PNG (TIF doesn't work in browsers)
- [x] Compare mode with slider overlay (original vs segmented)
- [x] Adjustable parameters (body threshold, eye dark %, brain intensity %)
- [x] Individual mask viewing (body, eye, brain)
- [x] Validation stats panel (areas, containment checks, overlap)

## Phase 3: Classical Segmentation Baseline (current)
Strategy:
1. **Well detection** — threshold + largest connected component to isolate the circular well
2. **Body segmentation** — percentile-based threshold within the well, morphological cleanup, largest component
3. **Eye segmentation** — find the darkest compact region within the body (compactness scoring)
4. **Brain segmentation** — moderately dark region in a neighborhood around the eye (head region)

## Phase 4: Try Cellpose (next)
- Cellpose is trained on cells, so likely won't work on whole organisms
- Worth a quick test with the `cyto2` or `TN1` model to confirm
- If it segments anything useful (e.g., the eye), we can use it as a component

## Phase 5: Advanced Methods (if needed)
- **SAM (Segment Anything)** — general-purpose, prompt with eye/brain points
- **Active contours** — fit to body boundary using edge information
- **Watershed** — separate touching structures within the body
- **U-Net** — only if labeled training data becomes available

## Validation Checklist
- [ ] Body mask is single connected component, roundish, fully within well
- [ ] Eye mask is compact, darkest structure, fully within body
- [ ] Brain mask near eye, within body, no overlap with eye
- [ ] Visual overlay looks correct on the sample image
- [ ] Runs in < 30 seconds
