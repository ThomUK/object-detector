const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const startBtn = document.getElementById('startBtn');
const flipBtn = document.getElementById('flipBtn');

const COLORS = [
  '#37c2b5', '#f2c14e', '#ef6f6c', '#7b9acc',
  '#c77dff', '#9bd67e', '#ffa552', '#6fd1f6',
];
const colorFor = (label) => {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return COLORS[h % COLORS.length];
};

let model = null;
let stream = null;
let facingMode = 'environment';
let rafId = null;
let running = false;
let lastFrameTime = 0;
let fpsAvg = 0;

const setStatus = (msg, visible = true) => {
  statusEl.innerHTML = msg;
  statusEl.classList.toggle('hidden', !visible);
};

async function ensureModel() {
  if (model) return model;
  setStatus('Loading model…');
  model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
  return model;
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

function drawPredictions(predictions) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const { sx, sy, scale } = videoDrawRect();
  const dpr = window.devicePixelRatio || 1;
  const lineWidth = Math.max(2, 3 * dpr);
  const fontSize = Math.max(12, 14 * dpr);
  ctx.lineWidth = lineWidth;
  ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif`;
  ctx.textBaseline = 'top';

  for (const p of predictions) {
    const [x, y, w, h] = p.bbox;
    const bx = sx + x * scale;
    const by = sy + y * scale;
    const bw = w * scale;
    const bh = h * scale;
    const color = colorFor(p.class);

    ctx.strokeStyle = color;
    ctx.strokeRect(bx, by, bw, bh);

    const label = `${p.class} ${Math.round(p.score * 100)}%`;
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
    const predictions = await model.detect(video);
    drawPredictions(predictions);

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
    await ensureModel();
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
    await startCamera();
  } catch (err) {
    setStatus(`Error: ${err.message || err}`);
  } finally {
    flipBtn.disabled = false;
  }
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
