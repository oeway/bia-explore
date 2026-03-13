"""FastAPI server for Daphnia segmentation viewer."""
import io
import json
import base64
import numpy as np
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from scipy import ndimage

app = FastAPI(title="Daphnia Segmentation Viewer")

DATA_DIR = Path(__file__).parent
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load image once at startup
IMG_PATH = DATA_DIR / "data" / "20260210-DM-MPLX-MIX-24h_J21_w1.TIF"
RAW_16 = np.array(Image.open(IMG_PATH))


def to_uint8(arr):
    """Normalize array to 0-255 uint8."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def arr_to_png_bytes(arr, colormap=None):
    """Convert numpy array to PNG bytes."""
    if arr.dtype != np.uint8:
        arr = to_uint8(arr)
    if colormap == "mask_overlay":
        # arr is expected to be an RGB array already
        img = Image.fromarray(arr)
    elif len(arr.shape) == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_well_mask(img):
    """Detect the circular well to restrict analysis."""
    # The well is the bright circular area; outside is dark
    threshold = np.percentile(img, 30)
    bright = img > threshold
    # Fill holes and find largest connected component
    bright = ndimage.binary_fill_holes(bright)
    labeled, n = ndimage.label(bright)
    if n == 0:
        return np.ones_like(img, dtype=bool)
    sizes = ndimage.sum(bright, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    well = labeled == largest
    # Erode slightly to avoid well border
    well = ndimage.binary_erosion(well, iterations=10)
    return well


def segment_body(img, well_mask, threshold_pct=35):
    """Segment the Daphnia body using thresholding within the well."""
    # Work within well only
    well_pixels = img[well_mask]
    threshold = np.percentile(well_pixels, threshold_pct)
    body = (img < threshold) & well_mask
    # Morphological cleanup
    body = ndimage.binary_fill_holes(body)
    body = ndimage.binary_opening(body, iterations=3)
    body = ndimage.binary_closing(body, iterations=5)
    body = ndimage.binary_fill_holes(body)
    # Keep largest connected component
    labeled, n = ndimage.label(body)
    if n == 0:
        return body
    sizes = ndimage.sum(body, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    body = labeled == largest
    return body


def segment_eye(img, body_mask, dark_pct=2):
    """Segment the eye — the darkest compact region within the body."""
    body_pixels = img[body_mask]
    threshold = np.percentile(body_pixels, dark_pct)
    eye = (img < threshold) & body_mask
    # Cleanup: close to merge nearby dark pixels, then keep compact component
    eye = ndimage.binary_closing(eye, iterations=3)
    eye = ndimage.binary_fill_holes(eye)
    labeled, n = ndimage.label(eye)
    if n == 0:
        return eye
    # Pick the most compact (roundest) component of reasonable size
    best_label = 1
    best_score = 0
    for i in range(1, n + 1):
        comp = labeled == i
        area = comp.sum()
        if area < 20:
            continue
        # Compactness: area / (perimeter^2) — approximate perimeter by erosion
        eroded = ndimage.binary_erosion(comp)
        perimeter = (comp & ~eroded).sum()
        if perimeter == 0:
            continue
        compactness = area / (perimeter ** 2)
        # Prefer darker average intensity
        mean_intensity = img[comp].mean()
        score = compactness * area / (mean_intensity + 1)
        if score > best_score:
            best_score = score
            best_label = i
    eye = labeled == best_label
    eye = ndimage.binary_fill_holes(eye)
    return eye


def segment_brain(img, body_mask, eye_mask, brain_intensity_pct=15):
    """Segment the brain — darker region near the eye, in the head."""
    # Find eye centroid to locate head region
    eye_coords = np.argwhere(eye_mask)
    if len(eye_coords) == 0:
        return np.zeros_like(img, dtype=bool)
    eye_center = eye_coords.mean(axis=0)

    # Create a search region around the eye (head area)
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    dist_from_eye = np.sqrt((y - eye_center[0]) ** 2 + (x - eye_center[1]) ** 2)
    eye_radius = max(np.sqrt(eye_mask.sum() / np.pi), 10)
    head_region = (dist_from_eye < eye_radius * 6) & body_mask & ~eye_mask

    if head_region.sum() == 0:
        return np.zeros_like(img, dtype=bool)

    # Brain: moderately dark region in the head area
    head_pixels = img[head_region]
    threshold = np.percentile(head_pixels, brain_intensity_pct)
    brain = (img < threshold) & head_region
    brain = ndimage.binary_closing(brain, iterations=3)
    brain = ndimage.binary_fill_holes(brain)
    brain = ndimage.binary_opening(brain, iterations=2)
    # Keep largest component
    labeled, n = ndimage.label(brain)
    if n == 0:
        return brain
    sizes = ndimage.sum(brain, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    brain = labeled == largest
    return brain


def create_overlay(img, body_mask, eye_mask, brain_mask, alpha=0.4):
    """Create RGB overlay of masks on the original image."""
    img8 = to_uint8(img)
    rgb = np.stack([img8, img8, img8], axis=-1).astype(np.float32)
    # Body: green outline
    body_border = body_mask & ~ndimage.binary_erosion(body_mask, iterations=2)
    rgb[body_border] = rgb[body_border] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    # Eye: red fill
    rgb[eye_mask] = rgb[eye_mask] * (1 - 0.5) + np.array([255, 50, 50]) * 0.5
    # Brain: blue fill
    rgb[brain_mask] = rgb[brain_mask] * (1 - 0.5) + np.array([50, 100, 255]) * 0.5
    return rgb.astype(np.uint8)


# Cache for current segmentation results
cache = {}


def run_segmentation(body_thresh=35, eye_dark_pct=2, brain_pct=15):
    key = (body_thresh, eye_dark_pct, brain_pct)
    if key in cache:
        return cache[key]
    well = get_well_mask(RAW_16)
    body = segment_body(RAW_16, well, body_thresh)
    eye = segment_eye(RAW_16, body, eye_dark_pct)
    brain = segment_brain(RAW_16, body, eye, brain_pct)
    overlay = create_overlay(RAW_16, body, eye, brain)
    result = {"body": body, "eye": eye, "brain": brain, "overlay": overlay, "well": well}
    cache[key] = result
    return result


@app.get("/api/original.png")
def get_original():
    return Response(content=arr_to_png_bytes(RAW_16), media_type="image/png")


@app.get("/api/segment.png")
def get_segmentation(
    body_thresh: float = Query(35, ge=5, le=80),
    eye_dark_pct: float = Query(2, ge=0.5, le=15),
    brain_pct: float = Query(15, ge=3, le=40),
):
    result = run_segmentation(body_thresh, eye_dark_pct, brain_pct)
    return Response(content=arr_to_png_bytes(result["overlay"], colormap="mask_overlay"), media_type="image/png")


@app.get("/api/mask/{name}.png")
def get_mask(
    name: str,
    body_thresh: float = Query(35),
    eye_dark_pct: float = Query(2),
    brain_pct: float = Query(15),
):
    result = run_segmentation(body_thresh, eye_dark_pct, brain_pct)
    if name not in result:
        return Response(content=b"Not found", status_code=404)
    mask = result[name]
    if mask.dtype == bool:
        return Response(content=arr_to_png_bytes(mask.astype(np.uint8) * 255), media_type="image/png")
    return Response(content=arr_to_png_bytes(mask), media_type="image/png")


@app.get("/api/stats")
def get_stats(
    body_thresh: float = Query(35),
    eye_dark_pct: float = Query(2),
    brain_pct: float = Query(15),
):
    result = run_segmentation(body_thresh, eye_dark_pct, brain_pct)
    return {
        "body_area_px": int(result["body"].sum()),
        "eye_area_px": int(result["eye"].sum()),
        "brain_area_px": int(result["brain"].sum()),
        "eye_in_body": bool(np.all(result["eye"] <= result["body"])),
        "brain_in_body": bool(np.all(result["brain"] <= result["body"])),
        "eye_brain_overlap": int((result["eye"] & result["brain"]).sum()),
        "image_shape": list(RAW_16.shape),
    }


@app.get("/", response_class=HTMLResponse)
def viewer():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daphnia Segmentation Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }
.header { background: #16213e; padding: 12px 24px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #0f3460; }
.header h1 { font-size: 18px; color: #e94560; }
.header .status { font-size: 13px; color: #888; margin-left: auto; }
.main { display: flex; height: calc(100vh - 49px); }
.sidebar { width: 300px; background: #16213e; padding: 16px; overflow-y: auto; border-right: 1px solid #0f3460; flex-shrink: 0; }
.sidebar h3 { font-size: 13px; text-transform: uppercase; color: #e94560; margin-bottom: 10px; letter-spacing: 1px; }
.control { margin-bottom: 16px; }
.control label { display: block; font-size: 12px; color: #aaa; margin-bottom: 4px; }
.control input[type=range] { width: 100%; accent-color: #e94560; }
.control .value { font-size: 12px; color: #e94560; float: right; }
.btn { background: #e94560; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 13px; width: 100%; margin-bottom: 8px; transition: background 0.2s; }
.btn:hover { background: #c73652; }
.btn.secondary { background: #0f3460; }
.btn.secondary:hover { background: #1a4a8a; }
.btn.active { outline: 2px solid #e94560; outline-offset: 2px; }
.stats { background: #0f3460; border-radius: 6px; padding: 12px; margin-top: 12px; font-size: 12px; }
.stats .row { display: flex; justify-content: space-between; margin-bottom: 4px; }
.stats .label { color: #888; }
.stats .ok { color: #4caf50; }
.stats .bad { color: #f44336; }
.viewer { flex: 1; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; }
.viewer canvas { max-width: 100%; max-height: 100%; }
.compare-container { position: relative; display: inline-block; }
.compare-container img { display: block; max-width: 100%; max-height: calc(100vh - 49px); }
.compare-slider { position: absolute; top: 0; bottom: 0; width: 3px; background: #e94560; cursor: ew-resize; z-index: 10; }
.compare-slider::after { content: ''; position: absolute; top: 50%; left: -10px; width: 23px; height: 40px; background: #e94560; border-radius: 4px; transform: translateY(-50%); }
.compare-right { position: absolute; top: 0; right: 0; bottom: 0; overflow: hidden; }
.compare-right img { position: absolute; top: 0; right: 0; max-height: calc(100vh - 49px); }
.legend { position: absolute; bottom: 16px; right: 16px; background: rgba(22,33,62,0.9); padding: 10px 14px; border-radius: 6px; font-size: 12px; }
.legend div { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.legend .dot { width: 12px; height: 12px; border-radius: 50%; }
.mode-tabs { display: flex; gap: 4px; margin-bottom: 16px; }
.mode-tab { flex: 1; padding: 6px; text-align: center; background: #0f3460; border: none; color: #aaa; cursor: pointer; font-size: 12px; border-radius: 4px; }
.mode-tab.active { background: #e94560; color: white; }
.loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #e94560; font-size: 14px; }
</style>
</head>
<body>
<div class="header">
  <h1>Daphnia Segmentation</h1>
  <span class="status" id="statusText">Ready</span>
</div>
<div class="main">
  <div class="sidebar">
    <h3>View Mode</h3>
    <div class="mode-tabs">
      <button class="mode-tab active" onclick="setMode('compare')" id="tab-compare">Compare</button>
      <button class="mode-tab" onclick="setMode('original')" id="tab-original">Original</button>
      <button class="mode-tab" onclick="setMode('overlay')" id="tab-overlay">Overlay</button>
    </div>

    <h3>Parameters</h3>
    <div class="control">
      <label>Body threshold percentile <span class="value" id="val-body">35</span></label>
      <input type="range" id="param-body" min="5" max="80" value="35" step="1">
    </div>
    <div class="control">
      <label>Eye dark percentile <span class="value" id="val-eye">2</span></label>
      <input type="range" id="param-eye" min="0.5" max="15" value="2" step="0.5">
    </div>
    <div class="control">
      <label>Brain intensity percentile <span class="value" id="val-brain">15</span></label>
      <input type="range" id="param-brain" min="3" max="40" value="15" step="1">
    </div>

    <button class="btn" onclick="runSegmentation()">Run Segmentation</button>
    <button class="btn secondary" onclick="resetParams()">Reset Defaults</button>

    <h3 style="margin-top:16px">Individual Masks</h3>
    <button class="btn secondary" onclick="showMask('body')">Show Body Mask</button>
    <button class="btn secondary" onclick="showMask('eye')">Show Eye Mask</button>
    <button class="btn secondary" onclick="showMask('brain')">Show Brain Mask</button>

    <div class="stats" id="stats">
      <div class="row"><span class="label">Click "Run Segmentation" to start</span></div>
    </div>
  </div>

  <div class="viewer" id="viewer">
    <div class="compare-container" id="compareContainer" style="display:none;">
      <img id="imgLeft" src="" draggable="false">
      <div class="compare-slider" id="slider"></div>
      <div class="compare-right" id="clipRight">
        <img id="imgRight" src="" draggable="false">
      </div>
    </div>
    <img id="singleImg" src="" style="max-width:100%;max-height:calc(100vh - 49px);display:none;" draggable="false">
    <div class="loading" id="loading">Loading image...</div>
    <div class="legend" id="legend" style="display:none;">
      <div><span class="dot" style="background:#00ff00;"></span> Body outline</div>
      <div><span class="dot" style="background:#ff3232;"></span> Eye</div>
      <div><span class="dot" style="background:#3264ff;"></span> Brain</div>
    </div>
  </div>
</div>

<script>
let mode = 'compare';
let overlayUrl = null;
let currentParams = {};

function getParams() {
  return {
    body_thresh: document.getElementById('param-body').value,
    eye_dark_pct: document.getElementById('param-eye').value,
    brain_pct: document.getElementById('param-brain').value
  };
}

function paramString() {
  const p = getParams();
  return `body_thresh=${p.body_thresh}&eye_dark_pct=${p.eye_dark_pct}&brain_pct=${p.brain_pct}`;
}

// Update value displays
['body','eye','brain'].forEach(name => {
  const el = document.getElementById('param-'+name);
  el.addEventListener('input', () => {
    document.getElementById('val-'+name).textContent = el.value;
  });
});

function setMode(m) {
  mode = m;
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-'+m).classList.add('active');
  render();
}

function setStatus(text) {
  document.getElementById('statusText').textContent = text;
}

function render() {
  const compare = document.getElementById('compareContainer');
  const single = document.getElementById('singleImg');
  const legend = document.getElementById('legend');
  const loading = document.getElementById('loading');

  compare.style.display = 'none';
  single.style.display = 'none';
  legend.style.display = 'none';

  if (mode === 'original') {
    single.style.display = 'block';
    single.src = '/api/original.png';
    loading.style.display = 'none';
  } else if (mode === 'overlay' && overlayUrl) {
    single.style.display = 'block';
    single.src = overlayUrl;
    legend.style.display = 'block';
    loading.style.display = 'none';
  } else if (mode === 'compare' && overlayUrl) {
    compare.style.display = 'block';
    document.getElementById('imgLeft').src = '/api/original.png';
    document.getElementById('imgRight').src = overlayUrl;
    legend.style.display = 'block';
    loading.style.display = 'none';
    requestAnimationFrame(() => initSlider());
  } else if (mode === 'compare' || mode === 'overlay') {
    loading.style.display = 'block';
    loading.textContent = 'Run segmentation first';
  }
}

function showMask(name) {
  const single = document.getElementById('singleImg');
  const compare = document.getElementById('compareContainer');
  compare.style.display = 'none';
  single.style.display = 'block';
  single.src = `/api/mask/${name}.png?${paramString()}`;
  document.getElementById('loading').style.display = 'none';
  document.getElementById('legend').style.display = 'none';
  // Deselect mode tabs
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
}

async function runSegmentation() {
  setStatus('Running segmentation...');
  document.getElementById('loading').style.display = 'block';
  document.getElementById('loading').textContent = 'Segmenting...';

  const ps = paramString();
  overlayUrl = `/api/segment.png?${ps}`;

  // Preload the overlay image
  const img = new window.Image();
  img.onload = async () => {
    setStatus('Done');
    render();
    // Fetch stats
    const resp = await fetch(`/api/stats?${ps}`);
    const data = await resp.json();
    updateStats(data);
  };
  img.onerror = () => setStatus('Error loading segmentation');
  img.src = overlayUrl;
}

function updateStats(data) {
  const el = document.getElementById('stats');
  el.innerHTML = `
    <div class="row"><span class="label">Body area</span><span>${data.body_area_px.toLocaleString()} px</span></div>
    <div class="row"><span class="label">Eye area</span><span>${data.eye_area_px.toLocaleString()} px</span></div>
    <div class="row"><span class="label">Brain area</span><span>${data.brain_area_px.toLocaleString()} px</span></div>
    <div class="row"><span class="label">Eye in body?</span><span class="${data.eye_in_body?'ok':'bad'}">${data.eye_in_body?'Yes':'No'}</span></div>
    <div class="row"><span class="label">Brain in body?</span><span class="${data.brain_in_body?'ok':'bad'}">${data.brain_in_body?'Yes':'No'}</span></div>
    <div class="row"><span class="label">Eye-brain overlap</span><span class="${data.eye_brain_overlap===0?'ok':'bad'}">${data.eye_brain_overlap} px</span></div>
  `;
}

function resetParams() {
  document.getElementById('param-body').value = 35;
  document.getElementById('param-eye').value = 2;
  document.getElementById('param-brain').value = 15;
  ['body','eye','brain'].forEach(name => {
    document.getElementById('val-'+name).textContent = document.getElementById('param-'+name).value;
  });
}

// Slider for compare mode
function initSlider() {
  const container = document.getElementById('compareContainer');
  const slider = document.getElementById('slider');
  const clipRight = document.getElementById('clipRight');
  const w = container.offsetWidth;

  let pos = w / 2;
  updateSliderPos(pos, w);

  let dragging = false;
  slider.addEventListener('mousedown', () => dragging = true);
  document.addEventListener('mouseup', () => dragging = false);
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    pos = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    updateSliderPos(pos, rect.width);
  });
  // Touch support
  slider.addEventListener('touchstart', () => dragging = true);
  document.addEventListener('touchend', () => dragging = false);
  document.addEventListener('touchmove', e => {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    pos = Math.max(0, Math.min(e.touches[0].clientX - rect.left, rect.width));
    updateSliderPos(pos, rect.width);
  });
}

function updateSliderPos(pos, totalWidth) {
  const slider = document.getElementById('slider');
  const clipRight = document.getElementById('clipRight');
  slider.style.left = pos + 'px';
  clipRight.style.left = pos + 'px';
  clipRight.style.width = (totalWidth - pos) + 'px';
}

// Load original on start
window.addEventListener('load', () => {
  const img = document.getElementById('singleImg');
  img.onload = () => document.getElementById('loading').style.display = 'none';
  img.style.display = 'block';
  img.src = '/api/original.png';
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
