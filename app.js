import {
  ObjectDetector,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const startBtn = document.getElementById('startBtn');
const flipBtn = document.getElementById('flipBtn');
const modelRadios = document.querySelectorAll('input[name="model"]');

const COLORS = [
  '#37c2b5', '#f2c14e', '#ef6f6c', '#7b9acc',
  '#c77dff', '#9bd67e', '#ffa552', '#6fd1f6',
];
const colorFor = (label) => {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return COLORS[h % COLORS.length];
};

let stream = null;
let facingMode = 'environment';
let rafId = null;
let running = false;
let lastFrameTime = 0;
let fpsAvg = 0;

const MEDIAPIPE_VERSION = '0.10.14';
const MEDIAPIPE_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const EFFICIENTDET_LITE0_URL =
  'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite';

const detectors = {
  'coco-ssd': {
    label: 'COCO-SSD',
    instance: null,
    async load() {
      if (this.instance) return;
      this.instance = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
    },
    async detect(video) {
      const raw = await this.instance.detect(video);
      return raw.map((p) => ({ class: p.class, score: p.score, bbox: p.bbox }));
    },
  },
  'efficientdet-lite0': {
    label: 'EfficientDet-Lite0',
    instance: null,
    async load() {
      if (this.instance) return;
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
      this.instance = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: EFFICIENTDET_LITE0_URL,
          delegate: 'GPU',
        },
        scoreThreshold: 0.3,
        runningMode: 'VIDEO',
      });
    },
    async detect(video) {
      const result = this.instance.detectForVideo(video, performance.now());
      return result.detections.map((d) => {
        const cat = d.categories[0] || { categoryName: 'object', score: 0 };
        return {
          class: cat.categoryName,
          score: cat.score,
          bbox: [d.boundingBox.originX, d.boundingBox.originY, d.boundingBox.width, d.boundingBox.height],
        };
      });
    },
  },
};

let activeDetectorKey = 'coco-ssd';
// Bumped on every model switch so an in-flight detect() from the previous
// model can be discarded instead of polluting the tracker.
let detectorGen = 0;

// Temporal smoothing: treat each detection frame as an update to a tracked
// object rather than an independent draw, so boxes stop flickering.
const IOU_MATCH_THRESHOLD = 0.3;
const BBOX_ALPHA = 0.35;         // EMA weight for new detection on bbox
const SCORE_ALPHA = 0.4;         // EMA weight for new detection on score
const MIN_NEW_SCORE = 0.5;       // gate for creating a new track
const MIN_DISPLAY_SCORE = 0.4;   // hide smoothed tracks below this
const MAX_MISSES = 6;            // keep a track alive across short dropouts

const tracks = new Map();
let nextTrackId = 1;

const setStatus = (msg, visible = true) => {
  statusEl.innerHTML = msg;
  statusEl.classList.toggle('hidden', !visible);
};

async function ensureActiveDetector() {
  const d = detectors[activeDetectorKey];
  if (!d.instance) {
    setStatus(`Loading ${d.label}…`);
    await d.load();
  }
  return d;
}

async function startCamera() {
  stopCamera();
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await new Promise((resolve) => {
    if (video.readyState >= 2) return resolve();
    video.onloadedmetadata = () => resolve();
  });
  await video.play();
  resizeOverlay();
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(rect.width * dpr);
  overlay.height = Math.round(rect.height * dpr);
  overlay.style.width = rect.width + 'px';
  overlay.style.height = rect.height + 'px';
}

function videoDrawRect() {
  // Compute where the video content is actually painted inside the element
  // given object-fit: cover, so bounding boxes line up precisely.
  const elW = overlay.width;
  const elH = overlay.height;
  const vW = video.videoWidth;
  const vH = video.videoHeight;
  if (!vW || !vH) return { sx: 0, sy: 0, scale: 1, sW: elW, sH: elH };
  const scale = Math.max(elW / vW, elH / vH);
  const sW = vW * scale;
  const sH = vH * scale;
  const sx = (elW - sW) / 2;
  const sy = (elH - sH) / 2;
  return { sx, sy, scale, sW, sH };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]);
  const y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

function updateTracks(predictions) {
  const matched = new Set();
  for (const track of tracks.values()) {
    let bestIdx = -1;
    let bestIou = IOU_MATCH_THRESHOLD;
    for (let i = 0; i < predictions.length; i++) {
      if (matched.has(i)) continue;
      const p = predictions[i];
      if (p.class !== track.class) continue;
      const score = iou(track.bbox, p.bbox);
      if (score > bestIou) { bestIou = score; bestIdx = i; }
    }
    if (bestIdx >= 0) {
      const p = predictions[bestIdx];
      matched.add(bestIdx);
      for (let k = 0; k < 4; k++) {
        track.bbox[k] += (p.bbox[k] - track.bbox[k]) * BBOX_ALPHA;
      }
      track.score += (p.score - track.score) * SCORE_ALPHA;
      track.misses = 0;
    } else {
      track.misses += 1;
    }
  }

  for (let i = 0; i < predictions.length; i++) {
    if (matched.has(i)) continue;
    const p = predictions[i];
    if (p.score < MIN_NEW_SCORE) continue;
    tracks.set(nextTrackId++, {
      class: p.class,
      bbox: p.bbox.slice(),
      score: p.score,
      misses: 0,
    });
  }

  for (const [id, track] of tracks) {
    if (track.misses > MAX_MISSES) tracks.delete(id);
  }
}

function drawTracks() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const { sx, sy, scale } = videoDrawRect();
  const dpr = window.devicePixelRatio || 1;
  const lineWidth = Math.max(2, 3 * dpr);
  const fontSize = Math.max(12, 14 * dpr);
  ctx.lineWidth = lineWidth;
  ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif`;
  ctx.textBaseline = 'top';

  for (const track of tracks.values()) {
    if (track.score < MIN_DISPLAY_SCORE) continue;
    const [x, y, w, h] = track.bbox;
    const bx = sx + x * scale;
    const by = sy + y * scale;
    const bw = w * scale;
    const bh = h * scale;
    const color = colorFor(track.class);

    ctx.strokeStyle = color;
    ctx.strokeRect(bx, by, bw, bh);

    const label = `${track.class} ${Math.round(track.score * 100)}%`;
    const padX = 6 * dpr;
    const padY = 4 * dpr;
    const textW = ctx.measureText(label).width;
    const labelH = fontSize + padY * 2;
    const labelY = by - labelH >= 0 ? by - labelH : by;

    ctx.fillStyle = color;
    ctx.fillRect(bx, labelY, textW + padX * 2, labelH);
    ctx.fillStyle = '#06201d';
    ctx.fillText(label, bx + padX, labelY + padY);
  }
}

async function detectLoop() {
  if (!running) return;
  try {
    const gen = detectorGen;
    const predictions = await detectors[activeDetectorKey].detect(video);
    if (gen !== detectorGen) {
      rafId = requestAnimationFrame(detectLoop);
      return;
    }
    updateTracks(predictions);
    drawTracks();

    const now = performance.now();
    if (lastFrameTime) {
      const dt = now - lastFrameTime;
      const fps = 1000 / dt;
      fpsAvg = fpsAvg ? fpsAvg * 0.9 + fps * 0.1 : fps;
      fpsEl.textContent = `${fpsAvg.toFixed(1)} FPS`;
    }
    lastFrameTime = now;
  } catch (err) {
    console.error(err);
  }
  rafId = requestAnimationFrame(detectLoop);
}

async function start() {
  startBtn.disabled = true;
  try {
    await ensureActiveDetector();
    setStatus('Starting camera…');
    await startCamera();
    setStatus('', false);
    fpsEl.classList.add('visible');
    flipBtn.disabled = false;
    startBtn.textContent = 'Stop';
    running = true;
    lastFrameTime = 0;
    detectLoop();
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message || err}`);
    flipBtn.disabled = true;
    startBtn.textContent = 'Start';
  } finally {
    startBtn.disabled = false;
  }
}

function stop() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  stopCamera();
  tracks.clear();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  fpsEl.classList.remove('visible');
  flipBtn.disabled = true;
  startBtn.textContent = 'Start';
  setStatus('Stopped. Tap <b>Start</b> to begin again');
}

startBtn.addEventListener('click', () => (running ? stop() : start()));

flipBtn.addEventListener('click', async () => {
  if (!running) return;
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  flipBtn.disabled = true;
  try {
    tracks.clear();
    await startCamera();
  } catch (err) {
    setStatus(`Error: ${err.message || err}`);
  } finally {
    flipBtn.disabled = false;
  }
});

async function switchDetector(key) {
  if (key === activeDetectorKey) return;
  const prev = activeDetectorKey;
  modelRadios.forEach((r) => (r.disabled = true));
  const prevStatusVisible = !statusEl.classList.contains('hidden');
  try {
    setStatus(`Loading ${detectors[key].label}…`);
    await detectors[key].load();
    activeDetectorKey = key;
    detectorGen += 1;
    tracks.clear();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!prevStatusVisible) setStatus('', false);
    else setStatus(`${detectors[key].label} ready`);
  } catch (err) {
    console.error(err);
    setStatus(`Error loading ${detectors[key].label}: ${err.message || err}`);
    const revert = document.querySelector(`input[name="model"][value="${prev}"]`);
    if (revert) revert.checked = true;
  } finally {
    modelRadios.forEach((r) => (r.disabled = false));
  }
}

modelRadios.forEach((r) => {
  r.addEventListener('change', (e) => {
    if (e.target.checked) switchDetector(e.target.value);
  });
});

window.addEventListener('resize', resizeOverlay);
window.addEventListener('orientationchange', () => setTimeout(resizeOverlay, 150));
document.addEventListener('visibilitychange', () => {
  if (document.hidden && running) stop();
});

if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  setStatus('Camera API not supported in this browser.');
  startBtn.disabled = true;
}
