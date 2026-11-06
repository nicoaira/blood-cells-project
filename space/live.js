// Live camera tab: detect blood cells in webcam video, in the browser.
//
// space/app.py runs this file as the js_on_load code of the tab's gr.HTML component, with `element` (the component's
// root), `props.live` (settings from model_info.json), `trigger` and `upload` in scope.
//
// The Space's ZeroGPU lends a GPU per request, which takes seconds, so video cannot go through it. Instead a web
// worker runs the ONNX export of the model with ONNX Runtime Web: on the GPU through WebGPU where the browser has
// it, otherwise on the CPU through WebAssembly, much slower. Red cell size is calibrated at several scales, then
// frames are analyzed in overlapping 640-pixel tiles at the fine-tuning cell size. Frames stay in the browser:
// only "Capture frame" uploads one, to the Image tab, where the Space's GPU pipeline adds the table and the CSV.
// In the local app, frames instead go to the loopback Python API and use the same CUDA model as the Image tab.
//
// The zoom slider is a digital zoom: the video is shown magnified about its center, and the detector and "Capture
// frame" get only that center crop, at the camera's own resolution, so the model sees the cells larger.

const cfg = props.live;
const $ = (selector) => element.querySelector(selector);
const ui = {
  start: $(".live-start"),
  stop: $(".live-stop"),
  recalibrate: $(".live-recalibrate"),
  capture: $(".live-capture"),
  camera: $(".live-camera"),
  detector: $(".live-detector"),
  stage: $(".live-stage"),
  video: $(".live-video"),
  overlay: $(".live-overlay"),
  placeholder: $(".live-placeholder"),
  counts: $(".live-counts"),
  conf: $(".live-conf"),
  confValue: $(".live-conf-value"),
  tags: $(".live-tags"),
  subtypes: $(".live-subtypes"),
  subtypeConf: $(".live-subtype-conf"),
  subtypeConfValue: $(".live-subtype-conf-value"),
  stable: $(".live-stable"),
  zoom: $(".live-zoom"),
  zoomValue: $(".live-zoom-value"),
  status: $(".live-status"),
};
const BACKENDS = { webgpu: "WebGPU (GPU)", wasm: "WebAssembly (CPU)", local: cfg.deviceName };

let worker = null; // the detector, once its model is loaded
let backend = "";
let loadedDetector = null;
let stream = null; // the camera
let running = false; // whether the detection loop runs
let generation = 0; // the current loop's number: a loop from before a stop and restart ends on its next frame
let pending = null; // callbacks of the frame the worker is processing
let rate = { since: 0, frames: 0, fps: 0 };
let localCalibration = null;
let localCalibrationVersion = 0;
const tracker = new LiveTracker();
let trackingRevision = 0;

function resetTracking() {
  tracker.reset();
  trackingRevision++;
  ui.overlay.getContext("2d").clearRect(0, 0, ui.overlay.width, ui.overlay.height);
  ui.counts.innerHTML = '<div class="counts-empty">Waiting for fresh detections…</div>';
}

// ---- The worker. Its source is this function's text, so it must not use anything outside it. ----

function detectorWorker() {
  let ort = null;
  let session = null;
  let cfg = null;
  let context = null;
  let calibration = null;
  let calibrationVersion = 0;

  async function fetchModel(url) {
    // From the browser's Cache Storage when an earlier visit downloaded it; the file name changes with the model.
    let cache = null;
    try {
      cache = await caches.open("blood-cell-detector");
      const allowed = cfg.modelUrls || [url];
      for (const request of await cache.keys()) if (!allowed.includes(request.url)) await cache.delete(request);
      const hit = await cache.match(url);
      if (hit) return new Uint8Array(await hit.arrayBuffer());
    } catch (error) {
      cache = null; // no Cache Storage here: download every time
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`model download failed (HTTP ${response.status})`);
    const total = Number(response.headers.get("content-length")) || cfg.modelBytes;
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    let reported = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (performance.now() - reported > 250) {
        reported = performance.now();
        self.postMessage({ type: "progress", loaded, total });
      }
    }
    const model = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      model.set(chunk, offset);
      offset += chunk.length;
    }
    if (cache) await cache.put(url, new Response(model)).catch(() => {}); // full storage: download next time
    return model;
  }

  async function load(settings) {
    cfg = settings;
    importScripts(cfg.ortScript);
    ort = self.ort;
    ort.env.wasm.wasmPaths = cfg.ortBase;
    ort.env.logLevel = "error";
    const model = await fetchModel(cfg.modelUrl);
    self.postMessage({ type: "progress", compiling: true });
    let backend = "wasm";
    try {
      if (self.navigator.gpu && (await self.navigator.gpu.requestAdapter())) {
        session = await ort.InferenceSession.create(model, { executionProviders: ["webgpu"] });
        backend = "webgpu";
      }
    } catch (error) {
      session = null; // WebGPU is there but unusable: use the CPU
    }
    if (!session) session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
    context = new OffscreenCanvas(cfg.size, cfg.size).getContext("2d", { willReadFrequently: true });
    // A first run compiles the GPU shaders, so that the first camera frame is not the slow one.
    const blank = new Float32Array(3 * cfg.size * cfg.size).fill(114 / 255);
    await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", blank, [1, 3, cfg.size, cfg.size]) });
    return backend;
  }

  function iou(a, b) {
    const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
    const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    if (w <= 0 || h <= 0) return 0;
    const inter = w * h;
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
  }

  function suppress(out, n, nc, conf) {
    // Ultralytics' non_max_suppression for one image, as model.predict runs it: the best class of each anchor, a
    // score above conf, greedy NMS within each class at IoU cfg.iou, and at most cfg.maxDet boxes. `out` holds, for
    // each of the n anchors, the box center, width and height and the nc class scores, one row per quantity.
    const candidates = [];
    for (let i = 0; i < n; i++) {
      let score = out[4 * n + i];
      let cls = 0;
      for (let c = 1; c < nc; c++) {
        const s = out[(4 + c) * n + i];
        if (s > score) {
          score = s;
          cls = c;
        }
      }
      if (score > conf) {
        const cx = out[i], cy = out[n + i], hw = out[2 * n + i] / 2, hh = out[3 * n + i] / 2;
        candidates.push([cx - hw, cy - hh, cx + hw, cy + hh, score, cls]);
      }
    }
    candidates.sort((a, b) => b[4] - a[4]);
    const kept = [];
    for (const box of candidates) {
      if (kept.length === cfg.maxDet) break;
      if (!kept.some((k) => k[5] === box[5] && iou(k, box) > cfg.iou)) kept.push(box);
    }
    return kept;
  }

  async function predictRegion(bitmap, sx, sy, sw, sh, nw, nh, conf) {
    // A tile is already at the training scale. Only its incomplete right/bottom edges receive gray padding.
    const { size } = cfg;
    context.fillStyle = "rgb(114, 114, 114)";
    context.fillRect(0, 0, size, size);
    context.drawImage(bitmap, sx, sy, sw, sh, 0, 0, nw, nh);
    const pixels = context.getImageData(0, 0, size, size).data;
    const area = size * size;
    const input = new Float32Array(3 * area);
    for (let i = 0, j = 0; i < area; i++, j += 4) {
      input[i] = pixels[j] / 255;
      input[area + i] = pixels[j + 1] / 255;
      input[2 * area + i] = pixels[j + 2] / 255;
    }
    const tensor = new ort.Tensor("float32", input, [1, 3, size, size]);
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    const output = outputs[session.outputNames[0]];
    const found = suppress(output.data, output.dims[2], output.dims[1] - 4, conf);
    tensor.dispose();
    for (const value of Object.values(outputs)) value.dispose();
    return found;
  }

  function fieldMask(bitmap) {
    // A small, smoothed approximation of smear.field_mask: bright, pale, largest connected field, holes filled.
    const ratio = Math.min(1, 192 / Math.max(bitmap.width, bitmap.height));
    const w = Math.max(1, Math.round(bitmap.width * ratio)), h = Math.max(1, Math.round(bitmap.height * ratio));
    const ctx = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, w, h);
    const pixels = ctx.getImageData(0, 0, w, h).data;
    const gray = [], hues = [], sats = [];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0, count = 0;
      for (let j = Math.max(0, y - 2); j <= Math.min(h - 1, y + 2); j++) {
        for (let i = Math.max(0, x - 2); i <= Math.min(w - 1, x + 2); i++) {
          const k = 4 * (j * w + i);
          r += pixels[k]; g += pixels[k + 1]; b += pixels[k + 2]; count++;
        }
      }
      r /= count; g /= count; b /= count;
      gray.push(0.299 * r + 0.587 * g + 0.114 * b);
      const max = Math.max(r, g, b), min = Math.min(r, g, b), delta = max - min;
      let hue = !delta ? 0 : max === r ? ((g - b) / delta + 6) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
      hues.push(hue * 30); // OpenCV hue range: 0..180
      sats.push(max ? (delta / max) * 255 : 0);
    }
    const median = (arr) => { const s = [...arr].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] || 0; };
    const p99 = [...gray].sort((a, b) => a - b)[Math.floor(0.99 * (gray.length - 1))];
    const ch = Math.max(1, Math.floor(h / 10)), cw = Math.max(1, Math.floor(w / 10));
    const box = (y0, y1, x0, x1, src) => {
      const vals = [];
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) vals.push(src[y * w + x]);
      return median(vals);
    };
    const cornerMin = Math.min(
      box(0, ch, 0, cw, gray), box(0, ch, w - cw, w, gray),
      box(h - ch, h, 0, cw, gray), box(h - ch, h, w - cw, w, gray),
    );
    if (cornerMin >= 0.3 * p99) {
      return { mask: new Uint8Array(w * h).fill(1), w, h, cx: bitmap.width / 2, cy: bitmap.height / 2 };
    }
    const s0 = box(Math.floor(h / 4), Math.max(1, Math.floor((3 * h) / 4)), Math.floor(w / 4), Math.max(1, Math.floor((3 * w) / 4)), sats);
    const cutoff = p99 * 0.5;
    const raw = gray.map((v, i) => v > cutoff && !((hues[i] < 15 || hues[i] > 150) && sats[i] > s0 + 0.2 * 255));
    const seen = new Uint8Array(w * h);
    const neighbors = (k) => [k % w ? k - 1 : -1, k % w < w - 1 ? k + 1 : -1, k - w, k + w];
    let largest = [];
    for (let i = 0; i < raw.length; i++) {
      if (!raw[i] || seen[i]) continue;
      const group = [i]; seen[i] = 1;
      for (let j = 0; j < group.length; j++) for (const k of neighbors(group[j])) {
        if (k >= 0 && k < raw.length && raw[k] && !seen[k]) { seen[k] = 1; group.push(k); }
      }
      if (group.length > largest.length) largest = group;
    }
    const mask = new Uint8Array(w * h);
    for (const k of largest) mask[k] = 1;
    const outside = new Uint8Array(w * h), queue = [];
    for (let k = 0; k < mask.length; k++) {
      if ((k < w || k >= w * (h - 1) || k % w === 0 || k % w === w - 1) && !mask[k]) {
        outside[k] = 1; queue.push(k);
      }
    }
    for (let j = 0; j < queue.length; j++) for (const k of neighbors(queue[j])) {
      if (k >= 0 && k < mask.length && !mask[k] && !outside[k]) { outside[k] = 1; queue.push(k); }
    }
    let cx = 0, cy = 0, count = 0;
    for (let k = 0; k < mask.length; k++) {
      mask[k] = outside[k] ? 0 : 1;
      if (mask[k]) { cx += k % w; cy += Math.floor(k / w); count++; }
    }
    return { mask, w, h, cx: count ? (cx / count + 0.5) * bitmap.width / w : bitmap.width / 2,
      cy: count ? (cy / count + 0.5) * bitmap.height / h : bitmap.height / 2 };
  }

  async function redCellSize(bitmap, field) {
    const candidates = [];
    const rbc = cfg.classNames.indexOf("RBC");
    for (const scale of [0.25, 0.5, 1, 2, 3]) {
      const half = cfg.size / (2 * scale);
      const sx = Math.max(0, Math.round(field.cx - half)), sy = Math.max(0, Math.round(field.cy - half));
      const sw = Math.min(bitmap.width - sx, Math.round(2 * half));
      const sh = Math.min(bitmap.height - sy, Math.round(2 * half));
      const boxes = await predictRegion(bitmap, sx, sy, sw, sh, Math.min(cfg.size, sw * scale),
        Math.min(cfg.size, sh * scale), 0.4);
      const sizes = boxes.filter((b) => b[5] === rbc).map((b) => Math.sqrt((b[2] - b[0]) * (b[3] - b[1])));
      if (sizes.length >= 8) {
        sizes.sort((a, b) => a - b);
        const mid = sizes.length >> 1;
        const diameter = sizes.length % 2 ? sizes[mid] : (sizes[mid - 1] + sizes[mid]) / 2;
        candidates.push({ count: sizes.length, diameter, native: diameter / scale });
      }
    }
    const trusted = candidates.filter((c) => c.diameter >= 50 && c.diameter <= 180);
    return (trusted.length ? trusted : candidates).sort((a, b) => b.count - a.count)[0]?.native || null;
  }

  function origins(length) {
    const tile = cfg.smear.tile;
    if (length <= tile) return [0];
    const n = Math.ceil((length - tile) / (tile - cfg.smear.overlap));
    return Array.from({ length: n + 1 }, (_, i) => Math.round(i * (length - tile) / n));
  }

  async function detect(bitmap, conf) {
    const start = performance.now();
    try {
      const w = bitmap.width, h = bitmap.height;
      const field = fieldMask(bitmap);
      let measured = calibration;
      if (!measured || measured.w !== w || measured.h !== h) {
        const version = calibrationVersion;
        self.postMessage({ type: "calibrating" });
        measured = { w, h, diameter: await redCellSize(bitmap, field) };
        if (version === calibrationVersion) calibration = measured;
      }
      const scale = measured.diameter ? cfg.smear.red_cell_px / measured.diameter : 1;
      const W = Math.max(1, Math.round(w * scale)), H = Math.max(1, Math.round(h * scale));
      const xs = origins(W), ys = origins(H), tile = cfg.smear.tile;
      const inside = (x, y) => field.mask[Math.min(field.h - 1, Math.floor(y * field.h / H)) * field.w +
        Math.min(field.w - 1, Math.floor(x * field.w / W))];
      const found = [];
      let tiles = 0;
      for (let i = 0; i < ys.length; i++) for (let j = 0; j < xs.length; j++) {
        const x = xs[j], y = ys[i], tw = Math.min(tile, W - x), th = Math.min(tile, H - y);
        let bright = 0;
        for (let a = 0; a < 8; a++) for (let b = 0; b < 8; b++) bright += inside(x + (a + 0.5) * tw / 8, y + (b + 0.5) * th / 8);
        if (bright / 64 < cfg.smear.min_field) continue;
        const boxes = await predictRegion(bitmap, x * w / W, y * h / H, tw * w / W, th * h / H, tw, th, conf);
        tiles++;
        const x0 = j ? (xs[j - 1] + tile + x) / 2 : -Infinity;
        const x1 = j + 1 < xs.length ? (x + tile + xs[j + 1]) / 2 : Infinity;
        const y0 = i ? (ys[i - 1] + tile + y) / 2 : -Infinity;
        const y1 = i + 1 < ys.length ? (y + tile + ys[i + 1]) / 2 : Infinity;
        for (const b of boxes) {
          const mx = (b[0] + b[2]) / 2 + x, my = (b[1] + b[3]) / 2 + y;
          if (mx >= x0 && mx < x1 && my >= y0 && my < y1) {
            found.push([b[0] + x, b[1] + y, b[2] + x, b[3] + y, b[4], b[5]]);
          }
        }
      }
      // Original photo coordinates, field filtering, and cross-class duplicate removal as in smear.py.
      const clamp = (v, hi) => Math.min(Math.max(v, 0), hi);
      const boxes = [];
      found.sort((a, b) => b[4] - a[4]);
      for (const [x1, y1, x2, y2, score, cls] of found) {
        const b = [clamp(x1, W), clamp(y1, H), clamp(x2, W), clamp(y2, H), score, cls];
        if (b[2] <= b[0] || b[3] <= b[1] || !inside((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) continue;
        if (!boxes.some((k) => iou(k, b) > cfg.smear.duplicate_iou)) boxes.push(b);
      }
      return { boxes: boxes.map((b) => b.map((v, i) => i < 4 ? v / scale : v)),
        ms: performance.now() - start, width: w, height: h, tiles, redCellPx: measured.diameter };
    } finally {
      bitmap.close();
    }
  }

  self.onmessage = async ({ data }) => {
    try {
      if (data.type === "load") self.postMessage({ type: "loaded", backend: await load(data.cfg) });
      else if (data.type === "frame") self.postMessage({ type: "result", ...(await detect(data.bitmap, data.conf)) });
      else if (data.type === "recalibrate") { calibration = null; calibrationVersion++; }
    } catch (error) {
      self.postMessage({ type: "error", message: String((error && error.message) || error) });
    }
  };
}

// ---- Page side ----

function setStatus(text, error = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle("live-error", error);
}

function setState(state) {
  // idle: camera off; starting: camera or detector loading; running: detecting.
  ui.start.disabled = state !== "idle";
  ui.stop.disabled = state === "idle";
  ui.capture.disabled = state !== "running";
  ui.recalibrate.disabled = state !== "running";
  ui.placeholder.hidden = state !== "idle";
}

function megabytes(bytes) {
  return `${Math.round(bytes / 1e6)} MB`;
}

function loadDetector(detectorId) {
  return new Promise((resolve, reject) => {
    worker?.terminate();
    worker = null;
    const source = `(${detectorWorker.toString()})();`;
    const candidate = new Worker(URL.createObjectURL(new Blob([source], { type: "text/javascript" })));
    candidate.onmessage = ({ data }) => {
      if (data.type === "progress" && data.compiling) {
        setStatus("Preparing the detector…");
      } else if (data.type === "progress") {
        const share = data.total ? ` (${Math.round((100 * data.loaded) / data.total)}%)` : "";
        setStatus(`Downloading the model: ${megabytes(data.loaded)} of ${megabytes(data.total)}${share}…`);
      } else if (data.type === "loaded") {
        worker = candidate;
        loadedDetector = detectorId;
        backend = data.backend;
        resolve();
      } else if (data.type === "calibrating") {
        setStatus("Measuring red cell size for this magnification…");
      } else if (data.type === "result" && pending) {
        pending.resolve(data);
        pending = null;
      } else if (data.type === "error") {
        if (pending) {
          pending.reject(new Error(data.message));
          pending = null;
        } else {
          candidate.terminate();
          reject(new Error(data.message));
        }
      }
    };
    candidate.onerror = (event) => {
      candidate.terminate();
      reject(new Error(event.message || "the detector did not start"));
    };
    const detector = cfg.detectors[detectorId];
    candidate.postMessage({ type: "load", cfg: { ...cfg, ...detector,
      modelUrl: new URL(detector.modelUrl, location.href).href,
      modelUrls: Object.values(cfg.detectors).map(d => new URL(d.modelUrl, location.href).href),
    } });
  });
}

async function ensureBrowserDetector(detectorId) {
  if (worker && loadedDetector === detectorId) return;
  setStatus(`Loading ${cfg.detectors[detectorId].label}…`);
  await loadDetector(detectorId);
}

function recalibrate() {
  resetTracking();
  localCalibration = null;
  localCalibrationVersion++;
  worker?.postMessage({ type: "recalibrate" });
}

async function detectLocalFrame(bitmap, conf, detectorId) {
  const w = bitmap.width, h = bitmap.height;
  const version = localCalibrationVersion;
  const measurement = localCalibration?.w === w && localCalibration?.h === h && localCalibration?.detector === detectorId
    ? localCalibration.redCellPx : null;
  if (!measurement) setStatus(`Measuring red cell size on ${cfg.deviceName}…`);
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  bitmap.close();
  const response = await fetch("/gradio_api/call/detect_live", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: [canvas.toDataURL("image/jpeg", 0.95), conf, measurement, detectorId] }),
  });
  if (!response.ok) throw new Error(`local inference failed (HTTP ${response.status})`);
  const { event_id } = await response.json();
  const result = await new Promise((resolve, reject) => {
    const events = new EventSource(`/gradio_api/call/detect_live/${event_id}`);
    events.addEventListener("complete", (event) => {
      events.close();
      try { resolve(JSON.parse(event.data)[0]); } catch (error) { reject(error); }
    });
    events.addEventListener("error", (event) => {
      events.close();
      reject(new Error(event.data || "The local inference connection closed. Check that the app is running."));
    });
  });
  if (version === localCalibrationVersion) localCalibration = { w, h, detector: detectorId, redCellPx: result.redCellPx };
  return result;
}

function detectFrame(bitmap, conf, detectorId) {
  if (cfg.backend === "local") return detectLocalFrame(bitmap, conf, detectorId);
  return new Promise((resolve, reject) => {
    pending = { resolve, reject };
    worker.postMessage({ type: "frame", bitmap, conf }, [bitmap]);
  });
}

function subtypeForBox(box) {
  if (!cfg.subtypesEnabled || box[5] !== cfg.wbcClass) return null;
  const probs = box[6];
  if (!probs) return { name: "Uncertain", score: null, index: -1 };
  const score = Math.max(...probs), index = probs.indexOf(score);
  const conf = Number(ui.subtypeConf?.value ?? cfg.defaultSubtypeConf);
  return { name: score >= conf ? cfg.subtypeNames[index] : "Uncertain", score,
    index: score >= conf ? index : -1 };
}

function draw({ boxes, width, height }) {
  const canvas = ui.overlay;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  const side = Math.max(width, height);
  context.lineWidth = Math.max(2, Math.round(side / 320));
  const ordered = [...boxes].sort((a, b) => cfg.drawRank[a[5]] - cfg.drawRank[b[5]]);
  for (const [x1, y1, x2, y2, , cls] of ordered) {
    context.strokeStyle = cfg.classColors[cls];
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  if (!ui.tags.checked && !ui.subtypes?.checked) return;
  const fontSize = Math.max(11, Math.round(side / 50));
  const pad = Math.max(1, Math.floor(context.lineWidth / 2));
  context.font = `${fontSize}px sans-serif`;
  context.textBaseline = "top";
  for (const box of ordered) {
    const [x1, y1, , , score, cls] = box;
    const subtype = subtypeForBox(box);
    let text;
    const approx = ui.stable.checked ? "≈" : "";
    if (subtype && ui.subtypes?.checked) {
      text = `WBC / ${subtype.name}` + (subtype.score === null ? "" : ` ${approx}${subtype.score.toFixed(2)}`);
      if (ui.tags.checked) text += ` (det ${approx}${score.toFixed(2)})`;
    } else if (ui.tags.checked) text = `${cfg.classNames[cls]} ${approx}${score.toFixed(2)}`;
    else continue;
    const tw = context.measureText(text).width + 2 * pad, th = fontSize + 2 * pad;
    const tx = Math.min(x1, width - tw), ty = y1 >= th ? y1 - th : y1;
    context.fillStyle = cfg.classColors[cls];
    context.fillRect(tx, ty, tw, th);
    context.fillStyle = "#ffffff";
    context.fillText(text, tx + pad, ty + pad);
  }
}

function showCounts(boxes) {
  const counts = cfg.classNames.map((_, i) => boxes.filter((b) => b[5] === i).length);
  const tiles = cfg.classNames.map(
    (name, i) =>
      `<div class="tile"><div class="name"><span class="swatch swatch-${i}"></span>${name}</div>` +
      `<div class="value">${counts[i]}</div><div class="note">${cfg.classNotes[i]}</div></div>`,
  );
  tiles.push(
    `<div class="tile total"><div class="name">Total</div><div class="value">${boxes.length}</div>` +
      `<div class="note">cells</div></div>`,
  );
  ui.counts.innerHTML = `<div class="counts">${tiles.join("")}</div>` +
    `<p class="counts-note">${ui.stable.checked ? "Stabilized live counts" : "Raw frame counts"}</p>`;
  if (cfg.subtypesEnabled) {
    const subcounts = new Array(cfg.subtypeNames.length + 1).fill(0);
    for (const box of boxes) {
      const subtype = subtypeForBox(box);
      if (subtype) subcounts[subtype.index < 0 ? cfg.subtypeNames.length : subtype.index]++;
    }
    const items = [...cfg.subtypeNames, "Uncertain"].map((name, i) =>
      `<span class="subtype-count">${name}: <b>${subcounts[i]}</b></span>`).join("");
    ui.counts.innerHTML += `<div class="subtype-summary"><strong>WBC subtypes</strong><div class="subtype-counts">${items}</div></div>`;
  }
}

function showRate({ ms, tiles, redCellPx }) {
  const now = performance.now();
  rate.frames += 1;
  if (now - rate.since >= 1000) {
    rate.fps = (1000 * rate.frames) / (now - rate.since);
    rate = { since: now, frames: 0, fps: rate.fps };
  }
  const fps = rate.fps ? `${rate.fps.toFixed(rate.fps < 10 ? 1 : 0)} frames/s · ` : "";
  const size = redCellPx ? `red cell ≈ ${Math.round(redCellPx)} px` : "cell size unavailable; using original scale";
  setStatus(`${cfg.detectors[ui.detector.value].label} · ${BACKENDS[backend]} · ${fps}${Math.round(ms)} ms · ${tiles} tiles · ${size}`);
}

function crop() {
  // The part of the video frame that the zoom shows: its center, 1/zoom of each side, in video pixels.
  const zoom = Number(ui.zoom.value);
  const w = ui.video.videoWidth, h = ui.video.videoHeight;
  const sw = Math.max(1, Math.round(w / zoom)), sh = Math.max(1, Math.round(h / zoom));
  return { sx: Math.round((w - sw) / 2), sy: Math.round((h - sh) / 2), sw, sh };
}

function applyZoom() {
  resetTracking();
  const zoom = Number(ui.zoom.value);
  ui.zoomValue.textContent = `${zoom.toFixed(1)}×`;
  ui.video.style.transform = zoom > 1 ? `scale(${zoom})` : "";
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const nextPaint = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));

async function loop(id) {
  rate = { since: performance.now(), frames: 0, fps: 0 };
  while (running && id === generation) {
    // Paused while the page or this tab is hidden: the camera stays on, the detector rests.
    if (document.hidden || !element.offsetParent || ui.video.readyState < 2) {
      resetTracking();
      if (!element.offsetParent) setStatus("Paused while the tab is hidden.");
      await sleep(250);
      rate = { since: performance.now(), frames: 0, fps: 0 };
      continue;
    }
    const detectorId = ui.detector.value;
    if (cfg.backend !== "local") {
      try { await ensureBrowserDetector(detectorId); }
      catch (error) { stop(); setStatus(`The detector did not load: ${error.message}`, true); return; }
      if (!running || id !== generation) return;
      if (detectorId !== ui.detector.value) continue;
    }
    const { sx, sy, sw, sh } = crop();
    const revision = trackingRevision;
    const stable = ui.stable.checked;
    const conf = Number(ui.conf.value);
    // Keep lower-scoring observations for established tracks; only new cells must cross the selected threshold.
    const inferenceConf = stable ? Math.max(cfg.confFloor ?? 0.05, conf - 0.1) : conf;
    const bitmap = await createImageBitmap(ui.video, sx, sy, sw, sh);
    let result;
    try {
      result = await detectFrame(bitmap, inferenceConf, detectorId);
    } catch (error) {
      stop();
      setStatus(`Detection stopped: ${error.message}`, true);
      return;
    }
    if (!running || id !== generation) return;
    if (revision !== trackingRevision) continue; // settings or camera changed while inference was running
    const detections = result.boxes.map((box, i) => [...box, result.subtypes?.[i] || null]);
    const boxes = stable ? tracker.update(detections, performance.now(), conf, result.width, result.height) : detections;
    draw({ ...result, boxes });
    showCounts(boxes);
    showRate(result);
    await nextPaint();
  }
}

function cameraError(error) {
  if (!navigator.mediaDevices) return "This browser gives the page no camera access (it needs HTTPS).";
  switch (error && error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera access was denied. Allow the camera for this site in the browser, then start again.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera found.";
    case "NotReadableError":
      return "The camera is busy: another application may be using it.";
    default:
      return `The camera did not start: ${(error && error.message) || error}`;
  }
}

async function listCameras(current) {
  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  ui.camera.replaceChildren(
    ...cameras.map((camera, i) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${i + 1}`;
      option.selected = camera.deviceId === current;
      return option;
    }),
  );
  ui.camera.disabled = cameras.length < 2;
}

function closeCamera() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  ui.video.srcObject = null;
}

async function openCamera(deviceId) {
  closeCamera();
  const size = { width: { ideal: 1280 }, height: { ideal: 720 } };
  const video = deviceId ? { deviceId: { exact: deviceId }, ...size } : size;
  stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  ui.video.srcObject = stream;
  await ui.video.play();
  await listCameras(stream.getVideoTracks()[0].getSettings().deviceId);
  recalibrate();
}

async function start() {
  setState("starting");
  setStatus("Asking for camera access…");
  try {
    await openCamera(ui.camera.value || undefined);
  } catch (error) {
    closeCamera();
    setState("idle");
    setStatus(cameraError(error), true);
    return;
  }
  if (cfg.backend === "local") backend = "local";
  else {
    setStatus("Loading the detector…");
    try {
      await ensureBrowserDetector(ui.detector.value);
    } catch (error) {
      stop();
      setStatus(`The detector did not load: ${error.message}`, true);
      return;
    }
    if (!stream) return; // stopped while loading
  }
  running = true;
  setState("running");
  setStatus(`Detecting on ${BACKENDS[backend]}…`);
  generation += 1;
  loop(generation);
}

function stop() {
  running = false;
  resetTracking();
  closeCamera();
  ui.overlay.getContext("2d").clearRect(0, 0, ui.overlay.width, ui.overlay.height);
  setState("idle");
  setStatus("Camera off.");
}

async function capture() {
  if (!stream || ui.video.readyState < 2) return;
  const { sx, sy, sw, sh } = crop(); // what the zoom shows
  const frame = document.createElement("canvas");
  frame.width = sw;
  frame.height = sh;
  frame.getContext("2d").drawImage(ui.video, sx, sy, sw, sh, 0, 0, sw, sh);
  ui.capture.disabled = true;
  try {
    const blob = await new Promise((resolve) => frame.toBlob(resolve, "image/png"));
    const { path } = await upload(new File([blob], "camera-frame.png", { type: "image/png" }));
    trigger("upload", { path, detector: ui.detector.value });
  } catch (error) {
    setStatus(`The frame was not sent: ${error.message}`, true);
  } finally {
    ui.capture.disabled = !running;
  }
}

ui.start.addEventListener("click", start);
ui.stop.addEventListener("click", stop);
ui.capture.addEventListener("click", capture);
ui.recalibrate.addEventListener("click", recalibrate);
ui.camera.addEventListener("change", () => {
  if (stream) openCamera(ui.camera.value).catch((error) => setStatus(cameraError(error), true));
});
ui.detector.addEventListener("change", () => {
  recalibrate();
  if (running) setStatus(`Switching to ${cfg.detectors[ui.detector.value].label}…`);
});
ui.zoom.addEventListener("input", applyZoom);
ui.conf.addEventListener("input", () => {
  resetTracking();
  ui.confValue.textContent = Number(ui.conf.value).toFixed(2);
});
ui.stable.addEventListener("change", resetTracking);
ui.subtypeConf?.addEventListener("input", () => {
  ui.subtypeConfValue.textContent = Number(ui.subtypeConf.value).toFixed(2);
});
if (!navigator.mediaDevices) {
  ui.start.disabled = true;
  setStatus(cameraError(null), true);
}
