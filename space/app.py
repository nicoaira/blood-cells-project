"""Blood cell detector: find and count white blood cells, red blood cells and platelets in a blood smear image.

A Gradio web app for YOLO26m fine-tuned on TXL-PBC and personal 400x/1000x microscope images. It runs on Hugging Face Spaces
with ZeroGPU, and locally on CUDA with `make app`. scripts/deploy_space.py stages this file in build/space next
to the weights, model_info.json (inference settings and test metrics) and the example images; that folder is
what the Space receives.

The GPU runs once per image and returns every box down to the evaluation's confidence floor. The confidence slider
and the display options then filter and redraw those boxes on the CPU, as the dashboard does, so trying another
threshold costs no GPU time.

The Live camera tab detects cells in webcam video in the visitor's browser, with the model's ONNX export (web/)
and live.js; ZeroGPU, which lends a GPU per request, is far too slow for video. A frame captured there comes back
to the Image tab, through the GPU pipeline above.
"""

from __future__ import annotations

import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # no Ultralytics usage analytics or downloads from the Space

import spaces  # noqa: E402  (ZeroGPU needs it imported before torch)

import csv  # noqa: E402
import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402
from pi_heif import register_heif_opener  # noqa: E402
from ultralytics import YOLO  # noqa: E402
import smear  # noqa: E402  (shared with local microscope-photo inference)
from wbc_classifier import WBCClassifier  # noqa: E402

register_heif_opener()  # HEIC/HEIF, the photo format of iPhones and many Android phones, for Pillow and so Gradio

HERE = Path(__file__).resolve().parent
INFO = json.loads((HERE / "model_info.json").read_text(encoding="utf-8"))
LOCAL = INFO["live"].get("backend") == "local"
REQUESTED_DEVICE = os.environ.get("BLOOD_CELLS_DEVICE", "cuda" if LOCAL else "auto")
if REQUESTED_DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Use `make setup-gpu` to install GPU dependencies.")
DEVICE = torch.device("cuda" if REQUESTED_DEVICE != "cpu" and torch.cuda.is_available() else "cpu")
DEVICE_LABEL = torch.cuda.get_device_name(DEVICE) if LOCAL and DEVICE.type == "cuda" else "CPU"
CLASS_NAMES: list[str] = INFO["class_names"]
CLASS_COLORS = {"WBC": "#2a78d6", "RBC": "#eb6834", "Platelets": "#1baf7a"}  # the dashboard's class colors
CLASS_NOTES = {"WBC": "white blood cells", "RBC": "red blood cells", "Platelets": "thrombocytes"}
DRAW_ORDER = ("RBC", "WBC", "Platelets")  # the many RBC boxes first, so the rarer classes stay on top
IMGSZ = INFO["imgsz"]
CONF_FLOOR = INFO["conf_floor"]
DEFAULT_CONF = round(INFO["operating_conf"], 2)
DEFAULT_DETECTOR = INFO["default_detector"]
DETECTOR_INFO = INFO["detectors"]
DETECTOR_CHOICES = [(f"{d['label']} — {d['weights']}", key) for key, d in DETECTOR_INFO.items()]
MAX_DISPLAY_SIDE = 2048  # larger images are drawn downscaled; the table keeps original pixel coordinates
CSV_HEADERS = ["Class", "Confidence", "x1", "y1", "x2", "y2", "WBC subtype", "Subtype confidence"]
TABLE_ORDER = (0, 6, 7, 1, 2, 3, 4, 5)
TABLE_HEADERS = ["Class", "WBC subtype", "Subtype confidence", "Detection confidence", "x1", "y1", "x2", "y2"]
SUBTYPE_INFO = INFO["wbc_classifier"]
SUBTYPE_NAMES = SUBTYPE_INFO["classes"]
SUBTYPE_LABELS = [name.replace("_", " ").capitalize() for name in SUBTYPE_NAMES]
DEFAULT_SUBTYPE_CONF = SUBTYPE_INFO["default_conf"]
EXAMPLES = sorted(str(p) for p in (HERE / "examples").glob("*.png"))
CSV_DIR = Path(tempfile.mkdtemp(prefix="blood-cells-"))
UPLOAD_DIR = Path(gr.utils.get_upload_folder()).resolve()  # where Gradio stores uploads, such as captured frames
BROWSER_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif"}  # images every browser displays

# Live camera tab. The browser fetches ONNX Runtime Web from jsDelivr and the model from this app's static files.
ORT_WEB = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/"
ORT_WASM_BYTES = 28_312_028  # ort-wasm-simd-threaded.jsep.wasm of that release, which the browser downloads too
LIVE_SETTINGS = {
    "backend": "local" if LOCAL else "browser",
    "deviceName": DEVICE_LABEL if LOCAL else None,
    "imgsz": IMGSZ,
    "iou": INFO["nms_iou"],
    "maxDet": INFO["max_det"],
    "classNames": CLASS_NAMES,
    "classColors": [CLASS_COLORS[name] for name in CLASS_NAMES],
    "classNotes": [CLASS_NOTES[name] for name in CLASS_NAMES],
    "drawRank": [DRAW_ORDER.index(name) for name in CLASS_NAMES],
    "smear": INFO["smear"],
    "confFloor": CONF_FLOOR,
    "wbcClass": CLASS_NAMES.index("WBC"),
    "subtypeNames": SUBTYPE_LABELS,
    "defaultSubtypeConf": DEFAULT_SUBTYPE_CONF,
    "subtypesEnabled": LOCAL,
    "defaultDetector": DEFAULT_DETECTOR,
    "detectors": {key: {"label": d["label"], "weights": d["weights"]} for key, d in DETECTOR_INFO.items()},
}
if not LOCAL:
    LIVE_MODEL = HERE / INFO["live"]["onnx"]
    gr.set_static_paths([LIVE_MODEL.parent])
    LIVE_SETTINGS.update({
        "modelUrl": f"/gradio_api/file={LIVE_MODEL.as_posix()}",
        "modelBytes": LIVE_MODEL.stat().st_size,
        "ortScript": ORT_WEB + "ort.webgpu.min.js",
        "ortBase": ORT_WEB,
        "size": INFO["live"]["input_size"],
    })
    for key, detector in DETECTOR_INFO.items():
        live_model = HERE / detector["live"]["onnx"]
        LIVE_SETTINGS["detectors"][key].update({"modelUrl": f"/gradio_api/file={live_model.as_posix()}",
            "modelBytes": live_model.stat().st_size, "size": detector["live"]["input_size"]})

# On ZeroGPU, CUDA is emulated outside @spaces.GPU, so the model is placed on "cuda" here, at startup, as the
# ZeroGPU docs ask. Fusing conv and batch norm now (Ultralytics does it on the first predict otherwise) leaves the
# GPU calls nothing to modify in the model.
detector_models = {}
for key, detector in DETECTOR_INFO.items():
    detector_model = YOLO(str(HERE / detector["weights"]))
    if [detector_model.names[i] for i in range(len(CLASS_NAMES))] != CLASS_NAMES:
        raise ValueError(f"Detector {key} does not use the expected WBC/RBC/Platelets class order")
    detector_model.fuse()
    detector_models[key] = detector_model.to(DEVICE)
subtype_model = WBCClassifier(HERE / SUBTYPE_INFO["weights"], str(DEVICE))
if subtype_model.names != SUBTYPE_NAMES:
    raise ValueError("WBC classifier class order does not match the app metadata")


def classify_wbcs(image_bgr: np.ndarray, boxes: np.ndarray) -> list:
    return subtype_model.classify_detections(np.ascontiguousarray(image_bgr[:, :, ::-1]), boxes,
        wbc_class=CLASS_NAMES.index("WBC"), padding=SUBTYPE_INFO["crop_padding"])


def subtype_label(probabilities, conf: float) -> tuple[str, float | None]:
    if probabilities is None:
        return "Uncertain", None
    index = int(np.argmax(probabilities))
    score = float(probabilities[index])
    return (SUBTYPE_LABELS[index] if score >= conf else "Uncertain"), score


def get_detector(detector_id: str):
    if detector_id not in detector_models:
        raise gr.Error("Choose one of the available cell detector models.")
    return detector_models[detector_id]


@spaces.GPU(duration=60)
def detect(image_bgr: np.ndarray, detector_id: str = DEFAULT_DETECTOR) -> tuple[np.ndarray, list, float, str]:
    """Every box down to CONF_FLOOR, as rows of x1, y1, x2, y2, confidence, class in the image's own pixels.

    The same cell-size normalization, field mask, overlapping tiles and duplicate removal as the local
    microscope-photo workflow. The original pixel coordinates are preserved for the table and API.
    """
    start = perf_counter()
    result = smear.detect(get_detector(detector_id), image_bgr, conf=CONF_FLOOR, device=DEVICE, batch=8)
    subtypes = classify_wbcs(image_bgr, result["boxes"])
    device = torch.cuda.get_device_name() if DEVICE.type == "cuda" else "CPU"
    return result["boxes"], subtypes, (perf_counter() - start) * 1000, device


def draw(image: np.ndarray, boxes: np.ndarray, labels: bool, subtype_rows: list | None = None,
         show_subtypes: bool = True) -> np.ndarray:
    """The image with a box per detection in its class color and, optionally, class and confidence tags."""
    h, w = image.shape[:2]
    scale = min(1.0, MAX_DISPLAY_SIDE / max(h, w))
    canvas = Image.fromarray(image)
    if scale < 1:
        canvas = canvas.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    side = max(canvas.size)
    width = max(2, round(side / 320))
    font = ImageFont.load_default(size=max(11, round(side / 50)))
    pen = ImageDraw.Draw(canvas)
    ordered = sorted(enumerate(boxes.tolist()), key=lambda pair: DRAW_ORDER.index(CLASS_NAMES[int(pair[1][5])]))
    for _, (x1, y1, x2, y2, _, cls) in ordered:
        pen.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=CLASS_COLORS[CLASS_NAMES[int(cls)]],
                      width=width)
    if labels or show_subtypes:  # after every box, so no box crosses a tag
        pad = max(1, width // 2)
        for index, (x1, y1, _, _, score, cls) in ordered:
            name = CLASS_NAMES[int(cls)]
            if name == "WBC" and show_subtypes and subtype_rows is not None:
                subtype, subtype_score = subtype_rows[index][6:8]
                text = f"WBC / {subtype}"
                if subtype_score is not None:
                    text += f" {subtype_score:.2f}"
                if labels:
                    text += f" (det {score:.2f})"
            elif labels:
                text = f"{name} {score:.2f}"
            else:
                continue
            left, top, right, bottom = pen.textbbox((0, 0), text, font=font)
            tw, th = right - left + 2 * pad, bottom - top + 2 * pad
            tx = min(x1 * scale, canvas.width - tw)
            ty = y1 * scale - th if y1 * scale >= th else y1 * scale  # above the box, or inside it at the top edge
            pen.rectangle([tx, ty, tx + tw, ty + th], fill=CLASS_COLORS[name])
            pen.text((tx + pad - left, ty + pad - top), text, fill="white", font=font)
    return np.asarray(canvas)


def counts_html(counts: list[int] | None, conf: float, timing: str = "") -> str:
    if counts is None:
        return '<div class="counts-empty">Drop an image or pick an example to count its cells.</div>'
    tiles = [
        f'<div class="tile"><div class="name"><span class="swatch swatch-{i}"></span>{name}</div>'
        f'<div class="value">{n}</div><div class="note">{CLASS_NOTES[name]}</div></div>'
        for i, (name, n) in enumerate(zip(CLASS_NAMES, counts))
    ]
    tiles.append(f'<div class="tile total"><div class="name">Total</div><div class="value">{sum(counts)}</div>'
                 '<div class="note">cells</div></div>')
    return (f'<div class="counts">{"".join(tiles)}</div>'
            f'<p class="counts-note">At confidence ≥ {conf:.2f}. Model inference: {timing}.</p>')


def write_csv(rows: list[list], detector_id: str = DEFAULT_DETECTOR) -> str:
    path = Path(tempfile.mkdtemp(dir=CSV_DIR)) / "detections.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([h.lower() for h in CSV_HEADERS] + ["detector model"])
        writer.writerows([*row, DETECTOR_INFO[detector_id]["weights"]] for row in rows)
    return str(path)


def select(boxes: np.ndarray, conf: float, subtypes: list | None = None,
           subtype_conf: float = DEFAULT_SUBTYPE_CONF) -> tuple[np.ndarray, list[int], list[list]]:
    """The boxes at or above conf, sorted by class and then by confidence, their count per class, and table rows."""
    indices = np.flatnonzero(boxes[:, 4] >= conf)
    indices = indices[np.lexsort((-boxes[indices, 4], boxes[indices, 5]))]
    kept = boxes[indices]
    counts = [int((kept[:, 5] == i).sum()) for i in range(len(CLASS_NAMES))]
    rows = [[CLASS_NAMES[int(c)], round(s, 3), *(round(v, 1) for v in xyxy)] for *xyxy, s, c in kept.tolist()]
    for index, row in zip(indices, rows):
        name, score = subtype_label(subtypes[index] if subtypes is not None else None, subtype_conf) \
            if row[0] == "WBC" else ("", None)
        row.extend([name, round(score, 3) if score is not None else None])
    return kept, counts, rows


def subtype_counts(rows: list) -> dict:
    counts = {name: 0 for name in [*SUBTYPE_NAMES, "uncertain"]}
    labels = dict(zip(SUBTYPE_LABELS, SUBTYPE_NAMES))
    for row in rows:
        if row[0] == "WBC":
            counts[labels.get(row[6], "uncertain")] += 1
    return counts


def subtype_counts_html(rows: list) -> str:
    counts = subtype_counts(rows)
    items = "".join(f'<span class="subtype-count">{label}: <b>{counts[name]}</b></span>'
                    for name, label in zip([*SUBTYPE_NAMES, "uncertain"], [*SUBTYPE_LABELS, "Uncertain"]))
    return '<div class="subtype-summary"><strong>WBC subtypes</strong><div class="subtype-counts">' + items + '</div></div>'


def render(state: dict | None, conf: float, shown: list[str], labels: bool,
           subtype_conf: float = DEFAULT_SUBTYPE_CONF, show_subtypes: bool = True):
    """Outputs for the stored detections at this threshold: drawing, counts, table and CSV download."""
    if state is None:
        return None, counts_html(None, conf), [], gr.DownloadButton(value=None, interactive=False)
    kept, counts, rows = select(state["boxes"], conf, state["subtypes"], subtype_conf)
    mask = np.isin(kept[:, 5], [CLASS_NAMES.index(name) for name in shown])
    visible_rows = [row for row, visible in zip(rows, mask) if visible]
    return (draw(state["image"], kept[mask], labels, visible_rows, show_subtypes),
            counts_html(counts, conf, state["timing"]) + subtype_counts_html(rows),
            [[row[index] for index in TABLE_ORDER] for row in rows],
            gr.DownloadButton(value=write_csv(rows, state["detector"]), interactive=True))


def run(image: np.ndarray | None, state: dict | None, conf: float, shown: list[str], labels: bool,
        subtype_conf: float = DEFAULT_SUBTYPE_CONF, show_subtypes: bool = True,
        detector_id: str = DEFAULT_DETECTOR):
    """Detect the cells in a new image (on the GPU) and render them; clears the outputs when the image is removed.

    An image whose detections are already shown is skipped: the image input reports one upload as two changes, and
    the model is deterministic, so running it again would only spend the visitor's GPU quota.
    """
    if image is None:
        return None, *render(None, conf, shown, labels, subtype_conf, show_subtypes)
    if (state is not None and state["detector"] == detector_id and state["image"].shape == image.shape
            and np.array_equal(state["image"], image)):
        return gr.skip()
    boxes, subtypes, ms, device = detect(np.ascontiguousarray(image[:, :, ::-1]), detector_id)
    state = {"image": image, "boxes": boxes, "subtypes": subtypes, "detector": detector_id,
             "timing": f"{ms:.0f} ms on {device} · {DETECTOR_INFO[detector_id]['label']}"}
    return state, *render(state, conf, shown, labels, subtype_conf, show_subtypes)


def detect_api(image: np.ndarray | None, conf: float) -> dict:
    """Count the cells in a blood smear image: the /detect API endpoint.

    Returns the threshold used, the image size, the count per class and one detection per cell, with its class,
    confidence and box corners [x1, y1, x2, y2] in image pixels, sorted by class and then by confidence.
    """
    return detect_model_api(image, conf, DEFAULT_DETECTOR)


def detect_model_api(image: np.ndarray | None, conf: float, detector_id: str = DEFAULT_DETECTOR) -> dict:
    """Analyze a photo with an explicit detector choice; the subtype classifier stays the same."""
    if image is None:
        raise gr.Error("Send an image.")
    boxes, subtypes, ms, device = detect(np.ascontiguousarray(image[:, :, ::-1]), detector_id)
    _, counts, rows = select(boxes, conf, subtypes)
    return {
        "model": DETECTOR_INFO[detector_id]["run_name"],
        "detector": detector_id,
        "weights": DETECTOR_INFO[detector_id]["weights"],
        "conf": conf,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "counts": {**dict(zip(CLASS_NAMES, counts)), "total": sum(counts)},
        "wbc_classifier": SUBTYPE_INFO["weights"],
        "wbc_subtype_conf": DEFAULT_SUBTYPE_CONF,
        "wbc_subtype_counts": subtype_counts(rows),
        "detections": [{"class": r[0], "confidence": r[1], "box": r[2:6],
                        **({"subtype": r[6], "subtype_confidence": r[7]} if r[0] == "WBC" else {})} for r in rows],
        "inference": f"{ms:.0f} ms on {device}",
    }


def detect_live_api(frame: str, conf: float, red_cell_px: float | None,
                    detector_id: str = DEFAULT_DETECTOR) -> dict:
    """Local camera frames use the same PyTorch model as photos, with a cached cell-size measurement.

    Data URLs avoid accumulating an uploaded file on disk for every video frame. Calls share the image queue,
    so the model is used by only one request at a time. The measurement belongs to the browser camera session.
    """
    if not LOCAL:
        raise gr.Error("Local camera inference is available through `make app`.")
    if not 0 <= conf <= 1 or (red_cell_px is not None and (not np.isfinite(red_cell_px) or red_cell_px <= 0)):
        raise gr.Error("Invalid confidence or cell-size measurement.")
    try:
        header, encoded = frame.split(",", 1)
        if header not in ("data:image/jpeg;base64", "data:image/png;base64"):
            raise ValueError("Unsupported frame format")
        image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    except Exception:
        raise gr.Error("The camera frame could not be read.")
    start = perf_counter()
    result = smear.detect(get_detector(detector_id), bgr, conf=conf, device=DEVICE, batch=8, red_cell_px=red_cell_px)
    subtypes = classify_wbcs(bgr, result["boxes"])
    return {"boxes": result["boxes"].tolist(), "subtypes": subtypes,
            "redCellPx": result["red_cell_px"], "tiles": result["tiles"],
            "width": image.width, "height": image.height, "ms": (perf_counter() - start) * 1000,
            "device": str(DEVICE), "deviceName": DEVICE_LABEL,
            "model": DETECTOR_INFO[detector_id]["run_name"], "detector": detector_id}


def show_upload(payload: dict | None):
    """Show an upload that browsers cannot display, such as a HEIC phone photo or a TIFF, as a PNG of it.

    Takes the image input's raw value (the event skips preprocessing) to see the file's name. The PNG holds the
    pixels that Gradio gave `run` for the original, so the change it causes finds them already detected.
    """
    name = (payload or {}).get("orig_name") or ""
    if not payload or not payload.get("path") or Path(name).suffix.lower() in BROWSER_FORMATS:
        return gr.skip()
    path = Path(payload["path"]).resolve()
    if not path.is_relative_to(UPLOAD_DIR) or not path.is_file():
        return gr.skip()
    try:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # as Gradio's own preprocessing
    except Exception:
        raise gr.Error(f"{name or 'This file'} is not an image this app can read. Try PNG, JPEG, WebP or HEIC.")
    return np.asarray(image)


def capture(evt: gr.EventData):
    """Show a frame captured in the Live camera tab in the Image tab, which then detects its cells on the GPU."""
    path = Path(str(getattr(evt, "path", ""))).resolve()
    if not path.is_relative_to(UPLOAD_DIR) or not path.is_file():  # only a file that live.js uploaded
        raise gr.Error("The captured frame did not arrive. Try again.")
    detector_id = getattr(evt, "detector", DEFAULT_DETECTOR)
    get_detector(detector_id)
    return detector_id, np.asarray(Image.open(path).convert("RGB")), gr.Tabs(selected="image")


def about_markdown() -> str:
    test = INFO["validation"]
    per_class = "\n".join(
        f"| {c['class']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['mAP50']:.3f} | {c['mAP50-95']:.3f} |"
        for c in test["per_class"]
    )
    o = test["overall"]
    counts = INFO["validation_counting"]
    counting = "\n".join(
        f"| {name} | {counts[name]['mae']:.2f} | {counts[name]['bias']:+.2f} |"
        for name in CLASS_NAMES
    )
    base = INFO["base_model"].removesuffix(".pt").replace("yolo", "YOLO", 1)
    repo = f" Code, training logs and reports: [{INFO['repo_url'].removeprefix('https://')}]({INFO['repo_url']})." \
        if INFO.get("repo_url") else ""
    live_notes = (
        f"**Live camera** runs on this computer's {DEVICE_LABEL}, using the same PyTorch model, calibration and "
        "field mask as photo analysis. Cell size is reused between frames. Use *Recalibrate* after changing "
        "the microscope objective. Photos and video stay on this computer."
        if LOCAL else
        "**Live camera** uses the same fine-tuned weights with cell-size calibration and overlapping tiles in the "
        "browser. Its smaller calibration input and approximate field mask can produce different counts from "
        "the full Image analysis. Use *Recalibrate* after changing the microscope objective; *Capture frame* runs "
        "the full photo pipeline."
    )
    return f"""
**Microscope detector (default).** `{INFO['weights']}`: {base}, first trained on TXL-PBC, then adapted to personal microscope
photos at 400× and 1000×, with TXL-PBC training images retained during fine-tuning.
Ultralytics {INFO['ultralytics_version']}.{repo}

**Original detector.** `{DETECTOR_INFO['original']['weights']}` is the earlier YOLO26m trained on TXL-PBC,
before adaptation to personal microscope photos. Choose the detector independently in the Image and Live camera
tabs. Both use the same cell-size normalization and tiling pipeline, and switching keeps the current detection
confidence threshold. The WBC subtype classifier stays the same. Original TXL-PBC test mAP50:
{DETECTOR_INFO['original']['test']['mAP50']:.3f}; mAP50–95: {DETECTOR_INFO['original']['test']['mAP50-95']:.3f}.
The personal validation figures below describe the microscope-adapted detector.

**Detection on personal validation tiles** ({counts['tiles']} tiles,
{sum(counts[name]['labeled'] for name in CLASS_NAMES)} labeled cells). These tiles were used for model and
threshold selection; they are not an independent test set. Precision and recall are at the validation
F1-optimal confidence, {INFO['operating_conf']:.4f}.

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
{per_class}
| **All classes** | **{o['precision']:.3f}** | **{o['recall']:.3f}** | **{o['mAP50']:.3f}** | **{o['mAP50-95']:.3f}** |

**Counting on personal validation tiles**, at confidence {counts['conf']:.4f}. Error in cells per tile:

| Class | Mean absolute error | Mean error |
| --- | ---: | ---: |
{counting}

**TXL-PBC test detection**, after fine-tuning: mAP50
{INFO['txl_test']['overall']['mAP50']:.3f}, mAP50–95 {INFO['txl_test']['overall']['mAP50-95']:.3f}.

**How an image is processed.** Red cell size is measured automatically, so 400× and 1000× photos do not need
separate models. The bright field is rescaled to match the training cell size and analyzed in overlapping
{IMGSZ}-pixel tiles. Duplicates are removed and boxes are mapped back to the original photo. If too few red cells
are found to measure, the photo is tiled at its original scale. The confidence slider filters saved detections
without running the model again. Whole-photo accuracy has not been measured independently.

{live_notes}

**Stabilize detections** is enabled by default for live video. It smooths boxes and confidence across frames,
requires two observations for a new cell, and holds a brief miss for up to 0.4 seconds. Established cells have
a small confidence buffer below the selected threshold. Tags marked ≈ show smoothed scores, and live counts use
the stabilized tracks. This can briefly lag slide movement. Turn it off for raw frame results; captured-image
analysis and the Image tab's measurements are independent of the live display.

**WBC subtype model.** `{SUBTYPE_INFO['weights']}` classifies square crops around detected WBCs into six
BloodMNIST categories: basophil, eosinophil, immature granulocytes, lymphocyte, monocyte and neutrophil.
On {SUBTYPE_INFO['test_images']:,} cleaned BloodMNIST test crops, accuracy is {SUBTYPE_INFO['test_accuracy']:.2%}
and macro-F1 {SUBTYPE_INFO['test_macro_f1']:.2%}. These figures measure classification of supplied cell crops;
they do not measure the complete detection-and-classification pipeline or your microscope photos.
The subtype threshold starts at {DEFAULT_SUBTYPE_CONF:.2f} and is a display setting, not a calibrated accuracy
guarantee. Low-scoring predictions are shown as *Uncertain*. Detection and subtype confidence are separate.
The classifier has no trained unknown/non-WBC class, so even a high score can be wrong on unfamiliar images.

**Limits.** The detector counts three broad classes; WBC subtype predictions come from the separate classifier.
TXL-PBC merges
four public collections (BCCD, BCDD, PBC and Raabin-WBC), and the test images come from the same sources, so
accuracy on new smears, stains or microscopes has to be measured there. This is a research
demo, not a diagnostic device.

**Examples** are test-split images from TXL-PBC (MIT License): Gan, L., Li, X. & Wang, X. A curated and
re-annotated peripheral blood cell dataset integrating four public resources. *Sci Data* 12, 1694 (2025).
[doi:10.1038/s41597-025-05980-z](https://doi.org/10.1038/s41597-025-05980-z)
"""


if LOCAL:
    LIVE_DESCRIPTION = (f"The model runs locally on <strong>{DEVICE_LABEL}</strong>, through the Python server. "
                        "Both photos and video stay on this computer. No browser model download is needed.")
else:
    LIVE_DESCRIPTION = ("The model runs in this browser: on the GPU through WebGPU where the browser supports it, "
                        "and otherwise on the CPU, much more slowly. The first start downloads about "
                        f"{round((LIVE_MODEL.stat().st_size + ORT_WASM_BYTES) / 1e6)} MB, which the browser then "
                        "keeps. Video frames stay on this device unless you capture one.")

LIVE_SUBTYPE_CONTROLS = f"""
    <label class="live-check"><input class="live-subtypes" type="checkbox" checked> Show WBC subtype labels</label>
    <label class="live-field"><span>WBC subtype confidence <b class="live-subtype-conf-value">{DEFAULT_SUBTYPE_CONF:.2f}</b></span>
      <input class="live-subtype-conf" type="range" min="0" max="1" step="0.05" value="{DEFAULT_SUBTYPE_CONF}">
    </label>
    <p class="live-hint">Low-confidence subtypes are shown as Uncertain. WBC labels show subtype scores;
    detection confidence is separate. Subtype accuracy on your microscope images has not been measured.</p>
""" if LOCAL else '<p class="live-hint">Capture a frame for WBC subtype classification in the Image tab.</p>'

LIVE_HTML = f"""
<div class="live">
  <div class="live-main">
    <div class="live-stage">
      <div class="live-frame">
        <video class="live-video" playsinline muted></video>
        <canvas class="live-overlay"></canvas>
      </div>
      <div class="live-placeholder">
        <p><strong>Point a camera at a blood smear</strong>, such as a microscope camera, and the model marks the
        cells as the video plays.</p>
        <p>{LIVE_DESCRIPTION}</p>
      </div>
    </div>
    <div class="live-bar">
      <button class="live-button live-primary live-start">Start camera</button>
      <button class="live-button live-stop" disabled>Stop</button>
      <button class="live-button live-recalibrate" disabled>Recalibrate</button>
      <select class="live-camera" aria-label="Camera" disabled><option value="">Default camera</option></select>
    </div>
    <label class="live-zoom-field" title="Digital zoom: the detector sees only the magnified center of the frame">
      <span>Zoom</span>
      <input class="live-zoom" type="range" min="1" max="6" step="0.1" value="1">
      <b class="live-zoom-value">1.0×</b>
    </label>
  </div>
  <div class="live-side">
    <label class="live-field"><span>Cell detector model</span>
      <select class="live-detector">{"".join(f'<option value="{key}"' + (' selected' if key == DEFAULT_DETECTOR else '') + f'>{d["label"]} — {d["weights"]}</option>' for key, d in DETECTOR_INFO.items())}</select>
    </label>
    <p class="live-hint">Switches the WBC/RBC/platelet detector. WBC subtype classification stays the same.
    The current detection threshold is kept; switching clears tracks and recalibrates cell size.</p>
    <div class="live-counts"><div class="counts-empty">Counts appear here while the camera runs.</div></div>
    <label class="live-field"><span>Confidence threshold <b class="live-conf-value">{DEFAULT_CONF:.2f}</b></span>
      <input class="live-conf" type="range" min="{CONF_FLOOR}" max="0.95" step="0.01" value="{DEFAULT_CONF}">
    </label>
    <label class="live-check"><input class="live-tags" type="checkbox"> Tag boxes with class and confidence</label>
    {LIVE_SUBTYPE_CONTROLS}
    <div>
      <label class="live-check"><input class="live-stable" type="checkbox" checked> Stabilize detections</label>
      <p class="live-hint">Smooths boxes and confidence across frames. A cell must appear twice before it is shown;
      a brief miss is kept for up to 0.4 s. The threshold applies to new cells; established cells have a small buffer.
      Turn off to see raw results. Capture frame gives a fresh, independent measurement.</p>
    </div>
    <div>
      <button class="live-button live-capture" disabled>Capture frame</button>
      <p class="live-hint">Cell size adjusts automatically for 400× and 1000×. Use Recalibrate after changing
      objectives. Capture sends the current frame to the Image tab for full analysis and CSV export.</p>
    </div>
    <p class="live-status" role="status" aria-live="polite"></p>
  </div>
</div>
"""

CSS = "\n".join([
    ".counts { display: grid; grid-template-columns: repeat(auto-fit, minmax(104px, 1fr)); gap: 8px; }",
    ".counts .tile { border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg);"
    " padding: 10px 12px; background: var(--block-background-fill); }",
    ".counts .name { display: flex; align-items: center; gap: 6px; font-weight: 600;"
    " color: var(--body-text-color); }",
    ".counts .swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }",
    ".counts .value { font-size: 30px; font-weight: 650; line-height: 1.25; font-variant-numeric: tabular-nums;"
    " color: var(--body-text-color); }",
    ".counts .note, .counts-note, .counts-empty { font-size: var(--text-sm);"
    " color: var(--body-text-color-subdued); }",
    ".counts-note { margin: 8px 0 0; }",
    ".subtype-summary { margin-top: 12px; }",
    ".subtype-counts { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 6px; }",
    ".subtype-count { font-size: var(--text-sm); }",
    ".counts-empty { padding: 14px 2px; }",
    *(f".counts .swatch-{i} {{ background: {CLASS_COLORS[name]}; }}" for i, name in enumerate(CLASS_NAMES)),
    # Live camera tab
    ".live { display: grid; grid-template-columns: minmax(0, 3fr) minmax(250px, 2fr); gap: 16px;"
    " align-items: start; }",
    "@media (max-width: 860px) { .live { grid-template-columns: 1fr; } }",
    ".live-stage { position: relative; min-height: 320px; overflow: hidden; border-radius: var(--radius-lg);"
    " border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }",
    ".live-frame { position: relative; overflow: hidden; }",
    ".live-video { display: block; width: 100%; height: auto; transform-origin: center; }",
    ".live-overlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }",
    ".live-placeholder { position: absolute; inset: 0; display: flex; flex-direction: column; justify-content: center;"
    " gap: 4px; padding: 24px 28px; background: var(--block-background-fill); color: var(--body-text-color-subdued); }",
    ".live-placeholder[hidden] { display: none; }",
    ".live-placeholder strong { color: var(--body-text-color); }",
    ".live-bar { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }",
    ".live-button { padding: var(--button-large-padding); border-radius: var(--button-large-radius);"
    " border: var(--button-border-width) solid var(--button-secondary-border-color);"
    " background: var(--button-secondary-background-fill); color: var(--button-secondary-text-color);"
    " font-size: var(--button-large-text-size); font-weight: var(--button-large-text-weight); cursor: pointer; }",
    ".live-button:hover:not(:disabled) { background: var(--button-secondary-background-fill-hover); }",
    ".live-primary { border-color: var(--button-primary-border-color);"
    " background: var(--button-primary-background-fill); color: var(--button-primary-text-color); }",
    ".live-primary:hover:not(:disabled) { background: var(--button-primary-background-fill-hover); }",
    ".live-button:disabled { opacity: 0.5; cursor: not-allowed; }",
    ".live-capture { width: 100%; }",
    ".live-camera { flex: 1 1 180px; min-width: 0; padding: 8px 10px; border-radius: var(--input-radius);"
    " border: 1px solid var(--input-border-color); background: var(--input-background-fill);"
    " color: var(--body-text-color); }",
    ".live-zoom-field { display: flex; align-items: center; gap: 10px; margin-top: 10px; font-weight: 600;"
    " color: var(--body-text-color); }",
    ".live-zoom { flex: 1; accent-color: var(--color-accent); }",
    ".live-zoom-value { min-width: 3.5em; text-align: right; font-variant-numeric: tabular-nums; }",
    ".live-side { display: flex; flex-direction: column; gap: 16px; }",
    ".live-field { display: flex; flex-direction: column; gap: 6px; font-weight: 600; color: var(--body-text-color); }",
    ".live-conf { width: 100%; accent-color: var(--color-accent); }",
    ".live-check { display: flex; align-items: center; gap: 8px; color: var(--body-text-color); }",
    ".live-hint, .live-status { margin: 6px 0 0; font-size: var(--text-sm); color: var(--body-text-color-subdued); }",
    ".live-status.live-error { color: var(--error-text-color); }",
])

with gr.Blocks(title="Blood cell detector", delete_cache=(3600, 3600), analytics_enabled=not LOCAL) as demo:
    gr.Markdown(
        "# Blood cell detector\n"
        "Drop a peripheral blood smear image, or pick an example, to find and count its **white blood cells**, "
        "**red blood cells** and **platelets**. *Live camera* does it on video from a webcam or microscope "
        "camera. Choose the **microscope-adapted YOLO26m** or the **original TXL-PBC detector**, with automatic cell-size "
        "adjustment for 400× and 1000× photos. Detected WBCs are classified into **six subtypes**; see *About the model* below for "
        "accuracy and limits."
    )
    if LOCAL:
        gr.Markdown(f"**Local inference:** {DEVICE_LABEL} · Choose the cell detector in each tab · "
                    f"**WBC classifier:** `{SUBTYPE_INFO['weights']}` · Images and live camera are processed on this computer.")
    state = gr.State(None)
    with gr.Tabs() as tabs:
        with gr.Tab("Image", id="image"), gr.Row(equal_height=False):
            with gr.Column():
                image_detector = gr.Dropdown(choices=DETECTOR_CHOICES, value=DEFAULT_DETECTOR,
                    label="Cell detector model", info="Both models detect WBC, RBC and platelets. Switching keeps the current detection threshold.")
                input_image = gr.Image(label="Blood smear image", type="numpy", image_mode="RGB", format="png",
                                       sources=["upload", "clipboard"], height=440)
                detect_button = gr.Button("Detect cells", variant="primary")
                if EXAMPLES:
                    gr.Examples(EXAMPLES, inputs=input_image, label="Examples from the TXL-PBC test split")
            with gr.Column():
                output_image = gr.Image(label="Detections", type="numpy", format="png", interactive=False,
                                        height=440, buttons=["download", "fullscreen"])
                counts = gr.HTML(counts_html(None, DEFAULT_CONF), padding=False)
                conf = gr.Slider(CONF_FLOOR, 0.95, value=DEFAULT_CONF, step=0.01, label="Confidence threshold",
                                 info=f"{DEFAULT_CONF:.2f} is the operating point chosen on the validation split")
                with gr.Row():
                    shown = gr.CheckboxGroup(CLASS_NAMES, value=CLASS_NAMES, label="Show boxes for")
                    labels = gr.Checkbox(False, label="Tag boxes with class and confidence")
                show_subtypes = gr.Checkbox(True, label="Show WBC subtype labels")
                subtype_confidence = gr.Slider(0, 1, value=DEFAULT_SUBTYPE_CONF, step=0.05,
                    label="WBC subtype confidence", info="Lower-scoring subtypes are shown as Uncertain. Separate from detection confidence.")
                with gr.Accordion("Detections table", open=False):
                    table = gr.Dataframe(headers=TABLE_HEADERS, datatype=["str", "str"] + ["number"] * 6,
                                         interactive=False, max_height=360, show_row_numbers=True)
                    gr.Markdown("Box corners in original image pixels. WBC tags show subtype scores; "
                                "the table gives detection and subtype confidence separately.")
                    download = gr.DownloadButton("Download CSV", value=None, interactive=False, size="sm")
        with gr.Tab("Live camera", id="live"):
            live_js = "\n".join((HERE / name).read_text(encoding="utf-8")
                                for name in ("live_tracking.js", "live.js"))
            live = gr.HTML(html_template=LIVE_HTML, js_on_load=live_js,
                           live=LIVE_SETTINGS)
    with gr.Accordion("About the model", open=False):
        gr.Markdown(about_markdown())

    view = [conf, shown, labels, subtype_confidence, show_subtypes]
    outputs = [state, output_image, counts, table, download]
    # The UI events are private. They share one queue with the API, so the model serves one image at a time
    # (Ultralytics models are not thread-safe) and a repeated change sees the state its predecessor stored.
    inputs = [input_image, state, *view, image_detector]
    input_image.change(run, inputs, outputs, api_visibility="private", concurrency_id="detect")
    input_image.upload(show_upload, input_image, input_image, preprocess=False, show_progress="hidden",
                       api_visibility="private")
    detect_button.click(run, inputs, outputs, api_visibility="private", concurrency_id="detect")
    image_detector.change(run, inputs, outputs, api_visibility="private", concurrency_id="detect",
                          trigger_mode="always_last")
    for control in view:
        control.change(render, [state, *view], outputs[1:], show_progress="hidden", trigger_mode="always_last",
                       api_visibility="private")
    live.upload(capture, None, [image_detector, input_image, tabs], api_visibility="private")
    # The public API, /detect: an image and a threshold in, JSON out. A hidden button carries the endpoint.
    api_result = gr.JSON(visible=False)
    gr.Button(visible=False).click(detect_api, [input_image, conf], api_result, api_name="detect",
                                   concurrency_id="detect")
    gr.Button(visible=False).click(detect_model_api, [input_image, conf, image_detector], api_result,
                                   api_name="detect_with_model", concurrency_id="detect")
    if LOCAL:
        live_frame = gr.Textbox(visible=False)
        live_cell_size = gr.Number(value=None, visible=False)
        gr.Button(visible=False).click(detect_live_api, [live_frame, conf, live_cell_size, image_detector], api_result,
                                       api_name="detect_live", concurrency_id="detect")


if __name__ == "__main__":
    local_options = ({"server_name": "127.0.0.1", "server_port": int(os.environ.get("BLOOD_CELLS_PORT", "7860")),
                      "inbrowser": os.environ.get("BLOOD_CELLS_OPEN") == "1", "share": False} if LOCAL else {})
    demo.launch(theme=gr.themes.Soft(primary_hue="violet", neutral_hue="stone"), css=CSS, **local_options)
