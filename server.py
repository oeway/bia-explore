"""FastAPI server for Daphnia segmentation viewer."""
import io
import numpy as np
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import cv2

app = FastAPI(title="Daphnia Segmentation Viewer")

DATA_DIR = Path(__file__).parent
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load image once at startup
IMG_PATH = DATA_DIR / "data" / "20260210-DM-MPLX-MIX-24h_J21_w1.TIF"
RAW_16 = np.array(Image.open(IMG_PATH))


def to_uint8(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def arr_to_png_bytes(arr):
    if arr.dtype != np.uint8:
        arr = to_uint8(arr)
    if len(arr.shape) == 3:
        img = Image.fromarray(arr)
    else:
        img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_well_mask(img):
    """Detect the circular well by fitting a circle to the bright region."""
    thresh = np.percentile(img, 30)
    bright = (img > thresh).astype(np.uint8)
    bright = ndimage.binary_fill_holes(bright).astype(np.uint8)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(img, dtype=bool)
    c = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(c)
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    well = ((x - cx) ** 2 + (y - cy) ** 2) < (radius - 30) ** 2
    return well


def segment_body(img, well, body_ratio_thresh=0.85):
    """Segment Daphnia body using local contrast ratio within the well."""
    local_bg = gaussian_filter(img.astype(float), sigma=50)
    ratio = img.astype(float) / (local_bg + 1)
    body_raw = (ratio < body_ratio_thresh) & well
    body_raw = ndimage.binary_closing(body_raw, iterations=5)
    body_raw = ndimage.binary_opening(body_raw, iterations=2)
    # Keep largest connected component
    lab, n = ndimage.label(body_raw)
    if n > 0:
        sz = [ndimage.sum(body_raw, lab, i) for i in range(1, n + 1)]
        body_raw = lab == (np.argmax(sz) + 1)
    body = ndimage.binary_fill_holes(body_raw)
    body = ndimage.binary_closing(body, iterations=8)
    body = ndimage.binary_opening(body, iterations=3)
    body = ndimage.binary_fill_holes(body)
    body = body & well
    # Ensure single connected component
    lab, n = ndimage.label(body)
    if n > 1:
        sz = [ndimage.sum(body, lab, i) for i in range(1, n + 1)]
        body = lab == (np.argmax(sz) + 1)
    return body


def segment_eye(img, body, eye_pct=4):
    """Segment the eye — the most circular dark region within the body."""
    if body.sum() == 0:
        return np.zeros_like(img, dtype=bool)
    body_pixels = img[body]
    eye_thresh = np.percentile(body_pixels, eye_pct)
    eye_mask = (img < eye_thresh) & body
    eye_mask = ndimage.binary_closing(eye_mask, iterations=4)
    eye_mask = ndimage.binary_fill_holes(eye_mask)
    lab, n = ndimage.label(eye_mask)
    if n == 0:
        return np.zeros_like(img, dtype=bool)
    # Score by circularity * sqrt(area) / intensity
    best_label, best_score = 1, 0
    for i in range(1, n + 1):
        comp = lab == i
        area = comp.sum()
        if area < 50:
            continue
        cc, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cc:
            continue
        perim = cv2.arcLength(cc[0], True)
        if perim == 0:
            continue
        circ = 4 * np.pi * area / (perim ** 2)
        score = circ * np.sqrt(area) / (img[comp].mean() + 1)
        if score > best_score:
            best_score = score
            best_label = i
    eye = ndimage.binary_fill_holes(lab == best_label)
    # Dilate slightly to capture full eye extent
    eye = ndimage.binary_dilation(eye, iterations=2)
    return eye & body


def segment_brain(img, body, eye, brain_pct=18):
    """Segment the brain — darker region near the eye in the head."""
    eye_coords = np.argwhere(eye)
    if len(eye_coords) == 0:
        return np.zeros_like(img, dtype=bool)
    ey, ex = eye_coords.mean(axis=0)
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    dist = np.sqrt((y - ey) ** 2 + (x - ex) ** 2)
    eye_r = max(np.sqrt(eye.sum() / np.pi), 15)
    # Tighter search radius: brain should be close to the eye
    head_region = (dist < eye_r * 4) & body & ~eye
    if head_region.sum() == 0:
        return np.zeros_like(img, dtype=bool)
    head_pixels = img[head_region]
    brain_thresh = np.percentile(head_pixels, brain_pct)
    brain = (img < brain_thresh) & head_region
    brain = ndimage.binary_closing(brain, iterations=3)
    brain = ndimage.binary_fill_holes(brain)
    brain = ndimage.binary_opening(brain, iterations=2)
    lab, n = ndimage.label(brain)
    if n > 0:
        sz = [ndimage.sum(brain, lab, i) for i in range(1, n + 1)]
        brain = lab == (np.argmax(sz) + 1)
    # Cap brain size: should not exceed ~5x eye area
    max_brain = max(eye.sum() * 5, 2000)
    if brain.sum() > max_brain:
        # Erode to reduce size
        while brain.sum() > max_brain:
            brain = ndimage.binary_erosion(brain)
            if brain.sum() == 0:
                break
    return brain & body & ~eye


def create_overlay(img, body, eye, brain):
    img8 = to_uint8(img)
    rgb = np.stack([img8, img8, img8], axis=-1).astype(np.float32)
    # Body: green border
    border = body & ~ndimage.binary_erosion(body, iterations=2)
    rgb[border] = rgb[border] * 0.4 + np.array([0, 255, 0]) * 0.6
    # Eye: red fill
    rgb[eye] = rgb[eye] * 0.4 + np.array([255, 50, 50]) * 0.6
    # Brain: blue fill
    rgb[brain] = rgb[brain] * 0.4 + np.array([50, 100, 255]) * 0.6
    return rgb.astype(np.uint8)


cache = {}


def run_segmentation(body_ratio=0.85, eye_pct=4, brain_pct=18):
    key = (body_ratio, eye_pct, brain_pct)
    if key in cache:
        return cache[key]
    well = get_well_mask(RAW_16)
    body = segment_body(RAW_16, well, body_ratio)
    eye = segment_eye(RAW_16, body, eye_pct)
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
    body_ratio: float = Query(0.85, ge=0.5, le=0.98),
    eye_pct: float = Query(4, ge=1, le=15),
    brain_pct: float = Query(18, ge=3, le=40),
):
    result = run_segmentation(body_ratio, eye_pct, brain_pct)
    return Response(content=arr_to_png_bytes(result["overlay"]), media_type="image/png")


@app.get("/api/mask/{name}.png")
def get_mask(
    name: str,
    body_ratio: float = Query(0.85),
    eye_pct: float = Query(4),
    brain_pct: float = Query(18),
):
    result = run_segmentation(body_ratio, eye_pct, brain_pct)
    if name not in result:
        return Response(content=b"Not found", status_code=404)
    mask = result[name]
    if mask.dtype == bool:
        return Response(content=arr_to_png_bytes(mask.astype(np.uint8) * 255), media_type="image/png")
    return Response(content=arr_to_png_bytes(mask), media_type="image/png")


@app.get("/api/stats")
def get_stats(
    body_ratio: float = Query(0.85),
    eye_pct: float = Query(4),
    brain_pct: float = Query(18),
):
    result = run_segmentation(body_ratio, eye_pct, brain_pct)
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
.sidebar { width: 320px; background: #16213e; padding: 16px; overflow-y: auto; border-right: 1px solid #0f3460; flex-shrink: 0; }
.sidebar h3 { font-size: 13px; text-transform: uppercase; color: #e94560; margin-bottom: 10px; letter-spacing: 1px; }
.control { margin-bottom: 16px; }
.control label { display: block; font-size: 12px; color: #aaa; margin-bottom: 4px; }
.control input[type=range] { width: 100%; accent-color: #e94560; }
.control .value { font-size: 12px; color: #e94560; float: right; }
.control .desc { font-size: 10px; color: #666; margin-top: 2px; clear: both; }
.btn { background: #e94560; color: white; border: none; padding: 10px 16px; border-radius: 4px; cursor: pointer; font-size: 13px; width: 100%; margin-bottom: 8px; transition: background 0.2s; }
.btn:hover { background: #c73652; }
.btn.secondary { background: #0f3460; }
.btn.secondary:hover { background: #1a4a8a; }
.stats { background: #0f3460; border-radius: 6px; padding: 12px; margin-top: 12px; font-size: 12px; }
.stats .row { display: flex; justify-content: space-between; margin-bottom: 4px; }
.stats .label { color: #888; }
.stats .ok { color: #4caf50; }
.stats .bad { color: #f44336; }
.viewer { flex: 1; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #111; }
.compare-container { position: relative; display: inline-block; user-select: none; }
.compare-container img { display: block; max-width: 100%; max-height: calc(100vh - 49px); }
.compare-slider { position: absolute; top: 0; bottom: 0; width: 3px; background: #e94560; cursor: ew-resize; z-index: 10; }
.compare-slider::after { content: '\\2194'; position: absolute; top: 50%; left: -12px; width: 27px; height: 27px; background: #e94560; border-radius: 50%; transform: translateY(-50%); display: flex; align-items: center; justify-content: center; font-size: 14px; color: white; line-height: 27px; text-align: center; }
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
      <button class="mode-tab" onclick="setMode('original')" id="tab-original">Original</button>
      <button class="mode-tab active" onclick="setMode('overlay')" id="tab-overlay">Overlay</button>
      <button class="mode-tab" onclick="setMode('compare')" id="tab-compare">Compare</button>
    </div>

    <h3>Segmentation Parameters</h3>
    <div class="control">
      <label>Body contrast ratio <span class="value" id="val-body">0.85</span></label>
      <input type="range" id="param-body" min="0.5" max="0.98" value="0.85" step="0.01">
      <div class="desc">Lower = tighter body mask (only darkest regions)</div>
    </div>
    <div class="control">
      <label>Eye dark percentile <span class="value" id="val-eye">4</span></label>
      <input type="range" id="param-eye" min="1" max="15" value="4" step="0.5">
      <div class="desc">Lower = stricter (only darkest pixels as eye)</div>
    </div>
    <div class="control">
      <label>Brain intensity percentile <span class="value" id="val-brain">18</span></label>
      <input type="range" id="param-brain" min="3" max="40" value="18" step="1">
      <div class="desc">Higher = larger brain region</div>
    </div>

    <button class="btn" onclick="runSegmentation()">Run Segmentation</button>
    <button class="btn secondary" onclick="resetParams()">Reset Defaults</button>

    <h3 style="margin-top:16px">Individual Masks</h3>
    <button class="btn secondary" onclick="showMask('body')">Body Mask</button>
    <button class="btn secondary" onclick="showMask('eye')">Eye Mask</button>
    <button class="btn secondary" onclick="showMask('brain')">Brain Mask</button>

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
const BASE = '';
let mode = 'overlay';
let overlayUrl = null;

function getParams() {
  return {
    body_ratio: document.getElementById('param-body').value,
    eye_pct: document.getElementById('param-eye').value,
    brain_pct: document.getElementById('param-brain').value
  };
}

function paramString() {
  const p = getParams();
  return `body_ratio=${p.body_ratio}&eye_pct=${p.eye_pct}&brain_pct=${p.brain_pct}`;
}

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
    single.src = 'api/original.png';
    loading.style.display = 'none';
  } else if (mode === 'overlay' && overlayUrl) {
    single.style.display = 'block';
    single.src = overlayUrl;
    legend.style.display = 'block';
    loading.style.display = 'none';
  } else if (mode === 'compare' && overlayUrl) {
    compare.style.display = 'block';
    const leftImg = document.getElementById('imgLeft');
    const rightImg = document.getElementById('imgRight');
    leftImg.src = 'api/original.png';
    rightImg.src = overlayUrl;
    legend.style.display = 'block';
    loading.style.display = 'none';
    leftImg.onload = () => initSlider();
  } else {
    loading.style.display = 'block';
    loading.textContent = overlayUrl ? 'Loading...' : 'Click "Run Segmentation" to start';
  }
}

function showMask(name) {
  document.getElementById('compareContainer').style.display = 'none';
  const single = document.getElementById('singleImg');
  single.style.display = 'block';
  single.src = `api/mask/${name}.png?${paramString()}`;
  document.getElementById('loading').style.display = 'none';
  document.getElementById('legend').style.display = 'none';
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
}

async function runSegmentation() {
  setStatus('Running segmentation...');
  document.getElementById('loading').style.display = 'block';
  document.getElementById('loading').textContent = 'Segmenting...';
  const ps = paramString();
  overlayUrl = `api/segment.png?${ps}`;
  const img = new window.Image();
  img.onload = async () => {
    setStatus('Done');
    render();
    try {
      const resp = await fetch(`api/stats?${ps}`);
      const data = await resp.json();
      updateStats(data);
    } catch(e) { console.error(e); }
  };
  img.onerror = () => { setStatus('Error'); document.getElementById('loading').textContent = 'Error loading segmentation'; };
  img.src = overlayUrl;
}

function updateStats(data) {
  document.getElementById('stats').innerHTML = `
    <div class="row"><span class="label">Body area</span><span>${data.body_area_px.toLocaleString()} px</span></div>
    <div class="row"><span class="label">Eye area</span><span>${data.eye_area_px.toLocaleString()} px</span></div>
    <div class="row"><span class="label">Brain area</span><span>${data.brain_area_px.toLocaleString()} px</span></div>
    <hr style="border-color:#1a1a2e;margin:6px 0">
    <div class="row"><span class="label">Eye in body?</span><span class="${data.eye_in_body?'ok':'bad'}">${data.eye_in_body?'Yes':'No'}</span></div>
    <div class="row"><span class="label">Brain in body?</span><span class="${data.brain_in_body?'ok':'bad'}">${data.brain_in_body?'Yes':'No'}</span></div>
    <div class="row"><span class="label">Eye-brain overlap</span><span class="${data.eye_brain_overlap===0?'ok':'bad'}">${data.eye_brain_overlap} px</span></div>
  `;
}

function resetParams() {
  document.getElementById('param-body').value = 0.85;
  document.getElementById('param-eye').value = 4;
  document.getElementById('param-brain').value = 18;
  document.getElementById('val-body').textContent = '0.85';
  document.getElementById('val-eye').textContent = '4';
  document.getElementById('val-brain').textContent = '18';
}

function initSlider() {
  const container = document.getElementById('compareContainer');
  const slider = document.getElementById('slider');
  const clipRight = document.getElementById('clipRight');
  const w = container.offsetWidth;
  let pos = w / 2;
  updateSliderPos(pos, w);
  let dragging = false;
  const onMove = e => {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    pos = Math.max(0, Math.min(clientX - rect.left, rect.width));
    updateSliderPos(pos, rect.width);
  };
  slider.onmousedown = slider.ontouchstart = () => dragging = true;
  document.onmouseup = document.ontouchend = () => dragging = false;
  document.onmousemove = onMove;
  document.ontouchmove = onMove;
}

function updateSliderPos(pos, totalWidth) {
  document.getElementById('slider').style.left = pos + 'px';
  const clip = document.getElementById('clipRight');
  clip.style.left = pos + 'px';
  clip.style.width = (totalWidth - pos) + 'px';
}

// Auto-load original on start
window.addEventListener('load', () => {
  const img = document.getElementById('singleImg');
  img.onload = () => document.getElementById('loading').style.display = 'none';
  img.style.display = 'block';
  img.src = 'api/original.png';
  // Auto-run segmentation with defaults
  setTimeout(runSegmentation, 500);
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
