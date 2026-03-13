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

# Load all TIF images from data directory
IMAGE_DIR = DATA_DIR / "data"
IMAGES = {}
for tif in sorted(IMAGE_DIR.glob("*.TIF")):
    IMAGES[tif.stem] = np.array(Image.open(tif))
# Default image key
DEFAULT_IMAGE = list(IMAGES.keys())[0] if IMAGES else None
RAW_16 = IMAGES[DEFAULT_IMAGE] if DEFAULT_IMAGE else None


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
    """Segment Daphnia body using local contrast ratio within the well.

    Adaptive: if initial threshold yields a very small body (<2% of well),
    gradually relax the threshold to handle low-contrast images.
    """
    local_bg = gaussian_filter(img.astype(float), sigma=50)
    ratio = img.astype(float) / (local_bg + 1)

    # Try the given threshold first; adapt if body is too small
    thresh = body_ratio_thresh
    well_area = well.sum()
    for _ in range(10):
        body_raw = (ratio < thresh) & well
        body_raw = ndimage.binary_closing(body_raw, iterations=5)
        body_raw = ndimage.binary_opening(body_raw, iterations=2)
        lab, n = ndimage.label(body_raw)
        if n > 0:
            sz = [ndimage.sum(body_raw, lab, i) for i in range(1, n + 1)]
            body_raw = lab == (np.argmax(sz) + 1)
        if body_raw.sum() > well_area * 0.02:
            break
        thresh += 0.02  # relax threshold for low contrast

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


def segment_brain(img, body, eye, brain_pct=25):
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


def run_segmentation(body_ratio=0.85, eye_pct=4, brain_pct=25, image_key=None):
    if image_key is None:
        image_key = DEFAULT_IMAGE
    if image_key not in IMAGES:
        raise ValueError(f"Unknown image: {image_key}")
    img = IMAGES[image_key]
    key = (image_key, body_ratio, eye_pct, brain_pct)
    if key in cache:
        return cache[key]
    well = get_well_mask(img)
    body = segment_body(img, well, body_ratio)
    eye = segment_eye(img, body, eye_pct)
    brain = segment_brain(img, body, eye, brain_pct)
    overlay = create_overlay(img, body, eye, brain)
    result = {"body": body, "eye": eye, "brain": brain, "overlay": overlay, "well": well}
    cache[key] = result
    return result


@app.get("/api/images")
def list_images():
    return {"images": list(IMAGES.keys()), "default": DEFAULT_IMAGE}


@app.get("/api/original.png")
def get_original(image: str = Query(None)):
    img_key = image or DEFAULT_IMAGE
    if img_key not in IMAGES:
        return Response(content=b"Not found", status_code=404)
    return Response(content=arr_to_png_bytes(IMAGES[img_key]), media_type="image/png")


@app.get("/api/segment.png")
def get_segmentation(
    body_ratio: float = Query(0.85, ge=0.5, le=0.98),
    eye_pct: float = Query(4, ge=1, le=15),
    brain_pct: float = Query(25, ge=3, le=40),
    image: str = Query(None),
):
    result = run_segmentation(body_ratio, eye_pct, brain_pct, image)
    return Response(content=arr_to_png_bytes(result["overlay"]), media_type="image/png")


@app.get("/api/mask/{name}.png")
def get_mask(
    name: str,
    body_ratio: float = Query(0.85),
    eye_pct: float = Query(4),
    brain_pct: float = Query(25),
    image: str = Query(None),
):
    result = run_segmentation(body_ratio, eye_pct, brain_pct, image)
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
    brain_pct: float = Query(25),
    image: str = Query(None),
):
    result = run_segmentation(body_ratio, eye_pct, brain_pct, image)
    img_key = image or DEFAULT_IMAGE
    return {
        "body_area_px": int(result["body"].sum()),
        "eye_area_px": int(result["eye"].sum()),
        "brain_area_px": int(result["brain"].sum()),
        "eye_in_body": bool(np.all(result["eye"] <= result["body"])),
        "brain_in_body": bool(np.all(result["brain"] <= result["body"])),
        "eye_brain_overlap": int((result["eye"] & result["brain"]).sum()),
        "image_shape": list(IMAGES[img_key].shape),
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
.header { background: #16213e; padding: 10px 16px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #0f3460; }
.header h1 { font-size: 16px; color: #e94560; white-space: nowrap; }
.header .status { font-size: 12px; color: #888; margin-left: auto; white-space: nowrap; }
.toggle-btn { display: none; background: #0f3460; border: none; color: #e94560; font-size: 20px; padding: 4px 10px; border-radius: 4px; cursor: pointer; }
.main { display: flex; height: calc(100vh - 45px); }
.sidebar { width: 300px; background: #16213e; padding: 14px; overflow-y: auto; border-right: 1px solid #0f3460; flex-shrink: 0; }
.sidebar h3 { font-size: 12px; text-transform: uppercase; color: #e94560; margin-bottom: 8px; letter-spacing: 1px; }
.control { margin-bottom: 14px; }
.control label { display: block; font-size: 12px; color: #aaa; margin-bottom: 4px; }
.control input[type=range] { width: 100%; accent-color: #e94560; height: 28px; }
.control .value { font-size: 12px; color: #e94560; float: right; }
.control .desc { font-size: 10px; color: #666; margin-top: 2px; clear: both; }
.btn { background: #e94560; color: white; border: none; padding: 10px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; width: 100%; margin-bottom: 8px; transition: background 0.2s; -webkit-tap-highlight-color: transparent; }
.btn:hover { background: #c73652; }
.btn:active { background: #b02e48; }
.btn.secondary { background: #0f3460; }
.btn.secondary:hover { background: #1a4a8a; }
.mask-btns { display: flex; gap: 6px; }
.mask-btns .btn { flex: 1; padding: 8px 4px; font-size: 12px; }
.stats { background: #0f3460; border-radius: 6px; padding: 10px; margin-top: 10px; font-size: 12px; }
.stats .row { display: flex; justify-content: space-between; margin-bottom: 3px; }
.stats .label { color: #888; }
.stats .ok { color: #4caf50; }
.stats .bad { color: #f44336; }
.viewer { flex: 1; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #111; }
.compare-container { position: relative; display: inline-block; user-select: none; touch-action: none; }
.compare-container img { display: block; max-width: 100%; max-height: calc(100vh - 45px); }
.compare-slider { position: absolute; top: 0; bottom: 0; width: 3px; background: #e94560; cursor: ew-resize; z-index: 10; touch-action: none; }
.compare-slider::after { content: '\\2194'; position: absolute; top: 50%; left: -14px; width: 31px; height: 31px; background: #e94560; border-radius: 50%; transform: translateY(-50%); display: flex; align-items: center; justify-content: center; font-size: 16px; color: white; line-height: 31px; text-align: center; }
.legend { position: absolute; bottom: 12px; right: 12px; background: rgba(22,33,62,0.9); padding: 8px 12px; border-radius: 6px; font-size: 11px; }
.legend div { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
.legend .dot { width: 10px; height: 10px; border-radius: 50%; }
.mode-tabs { display: flex; gap: 4px; margin-bottom: 14px; }
.mode-tab { flex: 1; padding: 7px 4px; text-align: center; background: #0f3460; border: none; color: #aaa; cursor: pointer; font-size: 12px; border-radius: 4px; -webkit-tap-highlight-color: transparent; }
.mode-tab.active { background: #e94560; color: white; }
.loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #e94560; font-size: 14px; }

/* Mobile: stack vertically, collapsible sidebar */
@media (max-width: 768px) {
  .toggle-btn { display: block; }
  .main { flex-direction: column; height: auto; min-height: calc(100vh - 45px); }
  .sidebar { width: 100%; max-height: 50vh; border-right: none; border-bottom: 1px solid #0f3460; padding: 12px; }
  .sidebar.collapsed { display: none; }
  .viewer { min-height: 50vh; height: 50vh; }
  .viewer img, .compare-container img, .compare-right img { max-height: 50vh !important; }
  #singleImg { max-height: 50vh !important; }
  .legend { bottom: 8px; right: 8px; font-size: 10px; padding: 6px 8px; }
}
@media (max-width: 480px) {
  .header h1 { font-size: 14px; }
  .sidebar { padding: 10px; }
  .btn { padding: 10px 12px; font-size: 13px; }
  .control input[type=range] { height: 32px; }
}
</style>
</head>
<body>
<div class="header">
  <button class="toggle-btn" id="toggleSidebar" onclick="toggleSidebar()">&#9776;</button>
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

    <h3>Image</h3>
    <div class="control">
      <select id="image-select" style="width:100%;padding:6px;background:#0f3460;color:#e0e0e0;border:1px solid #1a4a8a;border-radius:4px;font-size:12px;">
      </select>
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
      <label>Brain intensity percentile <span class="value" id="val-brain">25</span></label>
      <input type="range" id="param-brain" min="3" max="40" value="25" step="1">
      <div class="desc">Higher = larger brain region</div>
    </div>

    <button class="btn" onclick="runSegmentation()">Run Segmentation</button>
    <button class="btn secondary" onclick="resetParams()">Reset Defaults</button>

    <h3 style="margin-top:12px">Individual Masks</h3>
    <div class="mask-btns">
      <button class="btn secondary" onclick="showMask('body')">Body</button>
      <button class="btn secondary" onclick="showMask('eye')">Eye</button>
      <button class="btn secondary" onclick="showMask('brain')">Brain</button>
    </div>

    <div class="stats" id="stats">
      <div class="row"><span class="label">Click "Run Segmentation" to start</span></div>
    </div>
  </div>

  <div class="viewer" id="viewer">
    <div class="compare-container" id="compareContainer" style="display:none;">
      <img id="imgLeft" src="" draggable="false">
      <img id="imgRight" src="" draggable="false" style="position:absolute;top:0;left:0;width:100%;height:100%;">
      <div class="compare-slider" id="slider"></div>
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
let currentImage = null;

function toggleSidebar() {
  const sb = document.querySelector('.sidebar');
  sb.classList.toggle('collapsed');
}

function getParams() {
  return {
    body_ratio: document.getElementById('param-body').value,
    eye_pct: document.getElementById('param-eye').value,
    brain_pct: document.getElementById('param-brain').value
  };
}

function paramString() {
  const p = getParams();
  let ps = `body_ratio=${p.body_ratio}&eye_pct=${p.eye_pct}&brain_pct=${p.brain_pct}`;
  if (currentImage) ps += `&image=${encodeURIComponent(currentImage)}`;
  return ps;
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
    single.src = 'api/original.png?' + (currentImage ? `image=${encodeURIComponent(currentImage)}` : '');
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
    leftImg.src = 'api/original.png?' + (currentImage ? `image=${encodeURIComponent(currentImage)}` : '');
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
  document.getElementById('param-brain').value = 25;
  document.getElementById('val-body').textContent = '0.85';
  document.getElementById('val-eye').textContent = '4';
  document.getElementById('val-brain').textContent = '25';
}

function initSlider() {
  const container = document.getElementById('compareContainer');
  const slider = document.getElementById('slider');
  const w = container.offsetWidth;
  let pos = w / 2;
  updateSliderPos(pos, w);
  let dragging = false;
  const onMove = e => {
    if (!dragging) return;
    e.preventDefault();
    const rect = container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    pos = Math.max(0, Math.min(clientX - rect.left, rect.width));
    updateSliderPos(pos, rect.width);
  };
  slider.onmousedown = slider.ontouchstart = (e) => { dragging = true; e.preventDefault(); };
  document.onmouseup = document.ontouchend = () => dragging = false;
  document.onmousemove = onMove;
  document.ontouchmove = onMove;
}

function updateSliderPos(pos, totalWidth) {
  document.getElementById('slider').style.left = pos + 'px';
  // Clip the overlay (right) image: show only from slider position to the right
  const pct = (pos / totalWidth) * 100;
  document.getElementById('imgRight').style.clipPath = `inset(0 0 0 ${pct}%)`;
}

// Load image list and auto-run
window.addEventListener('load', async () => {
  try {
    const resp = await fetch('api/images');
    const data = await resp.json();
    const sel = document.getElementById('image-select');
    data.images.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (name === data.default) opt.selected = true;
      sel.appendChild(opt);
    });
    currentImage = data.default;
    sel.addEventListener('change', () => {
      currentImage = sel.value;
      overlayUrl = null;
      render();
    });
  } catch(e) { console.error('Failed to load images:', e); }
  const img = document.getElementById('singleImg');
  img.onload = () => document.getElementById('loading').style.display = 'none';
  img.style.display = 'block';
  img.src = 'api/original.png';
  setTimeout(runSegmentation, 500);
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
