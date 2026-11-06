"""Label your own blood smear pictures in Label Studio, starting from the model's boxes.

    python scripts/labeling.py prepare [--source DIR]   # tiles and pre-labels in data/labeling
    python scripts/labeling.py serve [--port 8089]      # Label Studio with the tiles as labeling tasks
    python scripts/labeling.py export                   # the corrected labels as a YOLO dataset in data/UIC_microscope_photos

prepare reads the photos and videos in data/Blood_smear_pictures_mine and its subfolders. A video gives its sharpest
frame in every 2 seconds, and a picture that is a copy of another is used once. The model measures the red cells
of each picture, which is then rescaled so that a red cell is RED_CELL_PX wide, whatever the objective and the
phone zoom were, and cut into 640-pixel tiles inside the bright field of view. A few sharp tiles per picture, spread
across it, become the labeling tasks, with the model's boxes as pre-labels. A white cell box without a violet or
blue nucleus inside is dropped: on these smears the model mistakes clumps of red cells for white cells. Tasks are ordered
one tile per picture first, so labeling from the top covers every picture early.

serve starts Label Studio on this computer (data stays here) with a project holding the tasks; each one opens with
the pre-labels to correct. The first start creates the account and prints its login. Labels live in
data/labeling/label-studio; run serve again after another prepare to add the tasks of new pictures. prepare never
deletes or replaces a tile it wrote before, since labels may refer to it.

export, while serve runs, writes the submitted labels as a YOLO dataset: one in five pictures of each session goes
to val, so that tiles of the same picture are never on both sides. Skipped tiles, tiles never submitted (drafts)
and tiles submitted exactly as pre-labeled (Ctrl+Enter submits and moves on, so they may never have been reviewed)
are left out; --keep-unchanged keeps the last ones.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from common import CLASS_NAMES, MODELS_DIR, ROOT, load_config, write_json

SOURCE_DIR = ROOT / "data" / "Blood_smear_pictures_mine"
LABEL_DIR = ROOT / "data" / "labeling"
TILES_DIR = LABEL_DIR / "tiles"
FRAMES_DIR = LABEL_DIR / "frames"
STUDIO_DIR = LABEL_DIR / "label-studio"
MINE_DIR = ROOT / "data" / "UIC_microscope_photos"
DOCUMENT_ROOT = ROOT / "data"  # Label Studio serves files below it

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
RED_CELL_PX = 100       # red cell diameter in the tiles; TXL-PBC's median is 87 px in the original images
TILE = 640              # tile side, the training image size
MIN_FIELD = 0.7         # share of a tile that must lie inside the bright field of view
MIN_SHARPNESS = 0.35    # tiles blurrier than this share of the picture's sharp tiles are not used
MIN_CELLS = 3           # tiles with fewer pre-labeled cells are not used
VIDEO_WINDOW_S = 2.0    # one frame per window of a video: the sharpest
PRELABEL_CONF = 0.25    # pre-label threshold; corrections are quicker to delete than to draw
NUCLEUS_SHARE = 0.22    # share of nucleus-stained pixels a white cell box needs (see nucleus_share)
DUPLICATE_CORR = 0.9    # fingerprints this correlated are the same picture
SAME_FIELD_CORR = 0.5   # video frames this correlated show the same field
PROJECT_TITLE = "Blood cells: my smears"
DEFAULT_PORT = 8089     # Label Studio's own default, 8080, is often taken by another dashboard


# ---------------------------------------------------------------- reading pictures


def capture_time(path: Path) -> datetime:
    """When a picture was taken: EXIF, else a phone-style IMG_YYYYMMDD_HHMMSS name, else the file time."""
    try:
        from PIL import Image

        exif = Image.open(path).getexif()
        stamp = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal, DateTime
        if stamp:
            return datetime.strptime(str(stamp).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    m = re.search(r"(20\d{6})[_-]?(\d{6})", path.stem)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return datetime.fromtimestamp(path.stat().st_mtime)


def load_photo(path: Path) -> np.ndarray:
    """A photo as a BGR array, upright as the phone shows it (HEIC through pi-heif)."""
    from PIL import Image, ImageOps

    try:
        from pi_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])


def sharpness(gray: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def video_frames(path: Path) -> list[tuple[float, np.ndarray]]:
    """The sharpest frame of each VIDEO_WINDOW_S window, when it is not much blurrier than the video's sharp frames.

    Phones held to an eyepiece shake and refocus, so most frames are blurred; the centre of the frame decides.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))  # OpenCV applies the rotation the phone recorded
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scores = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        r = min(h, w) // 4
        scores.append(sharpness(gray[h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r]))
    cap.release()
    if not scores:
        return []
    floor = 0.6 * float(np.percentile(scores, 90))
    window = max(1, round(VIDEO_WINDOW_S * fps))
    wanted = []
    for start in range(0, len(scores), window):
        best = start + int(np.argmax(scores[start:start + window]))
        if scores[best] >= floor:
            wanted.append(best)
    frames, cap, index = [], cv2.VideoCapture(str(path)), 0
    wanted_set = set(wanted)
    while wanted_set:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted_set:
            frames.append((index / fps, frame))
            wanted_set.discard(index)
        index += 1
    cap.release()
    return frames


def fingerprint(img: np.ndarray) -> np.ndarray:
    """Cell-scale texture of the picture's centre, normalized.

    Two encodings of one picture (a phone's HEIC and JPEG) correlate at 1.0; different fields of the same slide,
    whose eyepiece circle looks alike, near 0; video frames of one field half a second apart, 0.5 to 0.97.
    """
    import cv2

    h, w = img.shape[:2]
    centre = cv2.cvtColor(img[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(centre, (192, 192), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= cv2.GaussianBlur(g, (0, 0), 4)
    return (g - g.mean()) / (g.std() + 1e-6)


@dataclass
class Picture:
    name: str            # unique name, the tiles' prefix
    source: str          # file under the project, with ?t= seconds for a video frame
    kind: str            # photo or video frame
    taken: datetime
    image: np.ndarray = field(repr=False)


def read_pictures(source_dir: Path) -> tuple[list[Picture], list[dict]]:
    """Every photo and selected video frame under source_dir, without duplicates, oldest first."""
    import cv2

    files = sorted(p for p in source_dir.rglob("*") if p.is_file())
    pictures, skipped, prints = [], [], []
    for path in files:
        suffix = path.suffix.lower()
        rel = path.relative_to(ROOT).as_posix()
        if suffix in PHOTO_SUFFIXES:
            try:
                candidates = [(path.stem, rel, "photo", capture_time(path), load_photo(path))]
            except Exception as error:
                skipped.append({"source": rel, "reason": f"unreadable: {error}"})
                continue
        elif suffix in VIDEO_SUFFIXES:
            start = capture_time(path)
            frames = video_frames(path)
            print(f"  {path.name}: {len(frames)} sharp frames", flush=True)
            candidates = [(f"{path.stem}_t{t:05.1f}s", f"{rel}?t={t:.2f}", "video frame", start + timedelta(seconds=t),
                           frame) for t, frame in frames]
        else:
            continue
        for name, rel_source, kind, taken, image in candidates:
            mark = fingerprint(image)
            video = rel_source.split("?t=")[0] if kind == "video frame" else None
            twin = next((p for p, f in zip(pictures, prints)
                         if (c := float((mark * f).mean())) >= DUPLICATE_CORR
                         or (video and p.source.split("?t=")[0] == video and c >= SAME_FIELD_CORR)), None)
            if twin is not None:
                skipped.append({"source": rel_source, "reason": f"same picture or field as {twin.source}"})
                continue
            pictures.append(Picture(name, rel_source, kind, taken, image))
            prints.append(mark)
    if any(p.kind == "video frame" for p in pictures):
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        for p in pictures:
            if p.kind == "video frame":
                cv2.imwrite(str(FRAMES_DIR / f"{p.name}.png"), p.image)
    pictures.sort(key=lambda p: (p.taken, p.name))
    return pictures, skipped


# ---------------------------------------------------------------- measuring and tiling


def field_mask(img: np.ndarray) -> np.ndarray:
    """The bright field of view: the eyepiece's circle, or the whole picture when the phone zoomed past it.

    Stray light around the eyepiece can be as bright as the slide but is strongly colored (a pink glow below the
    circle in the video), while the slide, blurred over its cells, is pale; saturated red-to-magenta areas are left
    out.
    """
    import cv2

    h, w = img.shape[:2]
    sigma = max(3.0, 0.01 * min(h, w))
    smooth = cv2.GaussianBlur(img, (0, 0), sigma)
    blurred = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    p99 = float(np.percentile(blurred, 99))
    ch, cw = max(1, h // 10), max(1, w // 10)
    corners = (blurred[:ch, :cw], blurred[:ch, -cw:], blurred[-ch:, :cw], blurred[-ch:, -cw:])
    if min(float(np.median(c)) for c in corners) >= 0.3 * p99:
        return np.ones((h, w), dtype=bool)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    s0 = float(np.median(hsv[h // 4:3 * h // 4, w // 4:3 * w // 4, 1]))
    glow = ((hsv[..., 0] < 15) | (hsv[..., 0] > 150)) & (hsv[..., 1] > s0 + 0.2 * 255)
    mask = ((blurred > 0.5 * p99) & ~glow).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        mask = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)  # dark clumps inside are field too
    return filled.astype(bool)


def red_cell_size(model, img: np.ndarray, mask: np.ndarray, device) -> tuple[float | None, int]:
    """Median red cell diameter in the picture's pixels, and the number of red cells it rests on.

    The model looks at the field's centre at five scales, each time through a 960-pixel input. Its boxes are only
    trusted at a scale where red cells come out 50 to 180 pixels wide, the sizes it was trained on: much larger,
    it boxes fragments; much smaller, its boxes are loose. Of those scales, the one finding the most red cells wins.
    """
    ys, xs = np.nonzero(mask)
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(xs) else (img.shape[0] // 2, img.shape[1] // 2)
    found = []
    for s in (0.25, 0.5, 1.0, 2.0, 3.0):
        half = int(480 / s)
        im = resize(img[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half], s)
        size = int(np.ceil(max(im.shape[:2]) / 32) * 32)
        r = model.predict(im, imgsz=size, conf=0.4, max_det=3000, device=device, verbose=False)[0]
        b = r.boxes.data.cpu().numpy()
        red = b[b[:, 5] == CLASS_NAMES.index("RBC")]
        if len(red) >= 8:
            native = float(np.median(np.sqrt((red[:, 2] - red[:, 0]) * (red[:, 3] - red[:, 1])))) / s
            found.append((len(red), native, native * s))
    trusted = [f for f in found if 50 <= f[2] <= 180] or found
    if not trusted:
        return None, 0
    count, native, _ = max(trusted, key=lambda f: f[0])
    return native, count


def resize(img: np.ndarray, scale: float, nearest: bool = False) -> np.ndarray:
    import cv2

    if nearest:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interpolation)


def grid(length: int) -> list[int]:
    """Tile origins along one side, the grid centred on the picture."""
    n = length // TILE
    offset = (length - n * TILE) // 2
    return [offset + i * TILE for i in range(n)]


def glass_color(tile: np.ndarray) -> np.ndarray:
    """B, G, R of the slide's empty glass in a tile: its brightest pixels."""
    f = tile.astype(np.float32) + 1.0
    lum = f.mean(axis=2)
    return np.median(f[lum >= np.percentile(lum, 97)], axis=0)


def nucleus_share(tile: np.ndarray, box, glass: np.ndarray) -> float:
    """Share of the central part of a box stained like a nucleus.

    Romanowsky stains color nuclei violet, absorbing green far more than blue, or with some stains blue, absorbing
    red more than green. Red cells, even the dark grey-violet ones of thick areas, absorb the three colors more
    evenly, and the black edge of the field absorbs everything. Optical densities are taken against the glass
    color, which removes the color cast of the lamp and the phone. Calibrated by eye on 354 of the model's white
    cell boxes in these pictures: real white cells, sharp or blurred, score 0.23 to 0.9; clumps of red cells,
    blur streaks and the field's edge at most 0.21.
    """
    x1, y1, x2, y2 = box
    dx, dy = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
    patch = tile[int(y1 + dy):max(int(y2 - dy), int(y1 + dy) + 1), int(x1 + dx):max(int(x2 - dx), int(x1 + dx) + 1)]
    if patch.size == 0:
        return 0.0
    od = np.clip(np.log(glass / (patch.astype(np.float32) + 1.0)), 0, None)
    b, g, r = od[..., 0], od[..., 1], od[..., 2]
    violet = np.maximum(g, r) - b > 0.5
    blue = (r - b > 0.3) & (r > g + 0.05)
    not_black = od.mean(axis=-1) < 2.2
    return float(np.mean((violet | blue) & not_black))


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def clean_prelabels(tile: np.ndarray, field: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Pre-labels worth showing: inside the field, white cells with a nucleus, sizes that fit the class (tiles are
    scaled so a red cell is RED_CELL_PX wide: a platelet is much smaller, a red cell box twice that is a clump),
    one class per cell."""
    wbc, rbc, platelet = (CLASS_NAMES.index(c) for c in ("WBC", "RBC", "Platelets"))
    glass = glass_color(tile)
    kept = []
    for b in sorted(boxes.tolist(), key=lambda b: -b[4]):
        cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
        if not field[min(cy, TILE - 1), min(cx, TILE - 1)]:
            continue
        cls, side = int(b[5]), float(np.sqrt((b[2] - b[0]) * (b[3] - b[1])))
        if (cls == platelet and side > 0.6 * RED_CELL_PX) or (cls == rbc and side > 1.8 * RED_CELL_PX):
            continue
        if cls == wbc and nucleus_share(tile, b[:4], glass) < NUCLEUS_SHARE:
            continue
        if any(iou(b, k) > 0.7 for k in kept):  # the same cell under two classes: keep the surer one
            continue
        kept.append(b)
    return np.array(kept, dtype=np.float32).reshape(-1, 6)


def predict_tiles(model, tiles: list[np.ndarray], device, batch: int = 16) -> list[np.ndarray]:
    out = []
    for i in range(0, len(tiles), batch):
        results = model.predict(tiles[i:i + batch], imgsz=TILE, conf=PRELABEL_CONF, max_det=1000, device=device,
                                verbose=False)
        out.extend(r.boxes.data.cpu().numpy() for r in results)
    return out


def spread(points: list[tuple[float, float]], first: int, k: int) -> list[int]:
    """k indices of points far apart from one another, starting with `first` (farthest-point sampling)."""
    chosen = [first]
    while len(chosen) < min(k, len(points)):
        dist = [min((p[0] - points[c][0]) ** 2 + (p[1] - points[c][1]) ** 2 for c in chosen) for p in points]
        for c in chosen:
            dist[c] = -1
        chosen.append(int(np.argmax(dist)))
    return chosen


def tile_picture(model, picture: Picture, per_picture: int, device) -> tuple[dict, list[dict]]:
    """Measure, rescale and tile one picture; the chosen tiles with their pre-labels, best first."""
    import cv2

    img = picture.image
    h, w = img.shape[:2]
    field = field_mask(img)
    size, measured_on = red_cell_size(model, img, field, device)
    info = {"source": picture.source, "kind": picture.kind, "taken": picture.taken.isoformat(timespec="seconds"),
            "width": w, "height": h, "red_cell_px": round(size, 1) if size else "",
            "measured_on_cells": measured_on, "scale": "", "candidates": 0, "tiles": 0, "note": ""}
    if size is None:
        info["note"] = "no red cells found to measure"
        return info, []
    scale = RED_CELL_PX / size
    info["scale"] = round(scale, 4)
    big = resize(img, scale)
    big_field = resize(field.astype(np.uint8), scale, nearest=True).astype(bool)
    H, W = big.shape[:2]
    candidates = []
    for y in grid(H):
        for x in grid(W):
            coverage = float(big_field[y:y + TILE, x:x + TILE].mean())
            if coverage >= MIN_FIELD:
                tile = big[y:y + TILE, x:x + TILE]
                candidates.append({"x": x, "y": y, "field": coverage, "tile": tile,
                                   "sharpness": sharpness(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY))})
    info["candidates"] = len(candidates)
    if not candidates:
        info["note"] = "no tile inside the field of view"
        return info, []
    floor = MIN_SHARPNESS * float(np.percentile([c["sharpness"] for c in candidates], 90))
    candidates = [c for c in candidates if c["sharpness"] >= floor]
    for c, boxes in zip(candidates, predict_tiles(model, [c["tile"] for c in candidates], device)):
        c["boxes"] = clean_prelabels(c["tile"], big_field[c["y"]:c["y"] + TILE, c["x"]:c["x"] + TILE], boxes)
    candidates = [c for c in candidates if len(c["boxes"]) >= MIN_CELLS]
    if not candidates:
        info["note"] = "no sharp tile with cells"
        return info, []
    first = int(np.argmax([c["sharpness"] for c in candidates]))
    order = spread([(c["x"], c["y"]) for c in candidates], first, per_picture)
    chosen = []
    for rank, i in enumerate(order):
        c = candidates[i]
        r, col = c["y"] // TILE, c["x"] // TILE
        chosen.append({
            "tile": f"{picture.name}_r{r:02d}c{col:02d}.png", "image": c["tile"], "boxes": c["boxes"], "rank": rank,
            "source": picture.source, "taken": info["taken"], "red_cell_px": info["red_cell_px"], "scale": scale,
            # the tile's region in the original picture's pixels
            "x0": round(c["x"] / scale), "y0": round(c["y"] / scale),
            "x1": round((c["x"] + TILE) / scale), "y1": round((c["y"] + TILE) / scale),
            "field": round(c["field"], 3), "sharpness": round(c["sharpness"], 1),
        })
    info["tiles"] = len(chosen)
    return info, chosen


def prediction(boxes: np.ndarray, model_version: str) -> dict:
    """Pre-labels in Label Studio's format: percentages of the tile."""
    result = []
    for i, (x1, y1, x2, y2, conf, cls) in enumerate(boxes.tolist()):
        result.append({
            "id": f"p{i}", "type": "rectanglelabels", "from_name": "label", "to_name": "image",
            "original_width": TILE, "original_height": TILE, "image_rotation": 0, "score": round(conf, 3),
            "value": {"x": 100 * x1 / TILE, "y": 100 * y1 / TILE, "width": 100 * (x2 - x1) / TILE,
                      "height": 100 * (y2 - y1) / TILE, "rotation": 0, "rectanglelabels": [CLASS_NAMES[int(cls)]]},
        })
    score = float(np.mean(boxes[:, 4])) if len(boxes) else 0.0
    return {"model_version": model_version, "score": round(score, 3), "result": result}


def prepare(args) -> None:
    import cv2
    import torch
    from ultralytics import YOLO

    cfg = load_config()
    weights = args.weights or MODELS_DIR / f"{cfg['run_name']}.pt"
    device = 0 if torch.cuda.is_available() else "cpu"
    model_version = f"{weights.stem}-prelabel"
    print(f"Reading pictures in {shown(args.source)}…")
    pictures, skipped = read_pictures(args.source)
    for s in skipped:
        print(f"  skipped {s['source']}: {s['reason']}")
    print(f"{len(pictures)} pictures; measuring, rescaling and tiling on {'the GPU' if device == 0 else 'the CPU'}…")
    model = YOLO(str(weights))
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    infos, chosen = [], []
    for picture in pictures:
        per_picture = args.per_frame if picture.kind == "video frame" else args.per_photo
        info, tiles = tile_picture(model, picture, per_picture, device)
        infos.append(info)
        chosen.extend(tiles)
        print(f"  {picture.name}: red cell {info['red_cell_px'] or '?'} px, {info['tiles']} of {info['candidates']} "
              f"tiles{' (' + info['note'] + ')' if info['note'] else ''}", flush=True)
    # One tile per picture first, oldest picture first: labeling from the top covers every picture early.
    chosen.sort(key=lambda t: (t["rank"], t["taken"], t["tile"]))
    earlier = {}
    if (LABEL_DIR / "tiles.csv").exists():
        with (LABEL_DIR / "tiles.csv").open(encoding="utf-8") as f:
            earlier = {r["tile"]: r for r in csv.DictReader(f)}
    tasks, rows = [], []
    for order, t in enumerate(chosen, 1):
        path = TILES_DIR / t["tile"]
        if not path.exists():
            cv2.imwrite(str(path), t["image"], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        elif not np.array_equal(cv2.imread(str(path)), t["image"]):
            # Labels in Label Studio may refer to the existing image, so it is never replaced.
            print(f"  kept the earlier {t['tile']}: this run would cut it differently; not offered again")
            continue
        counts = [int((t["boxes"][:, 5] == k).sum()) for k in range(len(CLASS_NAMES))]
        rows.append({"order": order, "tile": t["tile"], "source": t["source"], "taken": t["taken"],
                     "red_cell_px": t["red_cell_px"], "scale": round(t["scale"], 4), "x0": t["x0"], "y0": t["y0"],
                     "x1": t["x1"], "y1": t["y1"], "field": t["field"], "sharpness": t["sharpness"],
                     **{f"pre_{c}": n for c, n in zip(CLASS_NAMES, counts)}})
        tasks.append({"data": {"image": f"/data/local-files/?d={(TILES_DIR / t['tile']).relative_to(DOCUMENT_ROOT)}",
                               "tile": t["tile"], "source": t["source"], "order": order},
                      "predictions": [prediction(t["boxes"], model_version)]})
    # Tiles of earlier runs stay on disk and in tiles.csv whether chosen again or not: they may be labeled.
    write_json(LABEL_DIR / "tasks.json", tasks, indent=None)
    write_csv(LABEL_DIR / "tiles.csv", list((earlier | {r["tile"]: r for r in rows}).values()))
    write_csv(LABEL_DIR / "pictures.csv", infos + [{"source": s["source"], "note": s["reason"]} for s in skipped])
    totals = {c: sum(r[f"pre_{c}"] for r in rows) for c in CLASS_NAMES}
    print(f"{len(tasks)} tasks from {sum(1 for i in infos if i['tiles'])} pictures, pre-labeled with "
          + ", ".join(f"{n} {c}" for c, n in totals.items())
          + f". Next: `make label` opens them in Label Studio.")


def shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- Label Studio

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" canRotate="false" strokeWidth="2" opacity="0.15">
    <Label value="WBC" background="#2a78d6" hotkey="1"/>
    <Label value="RBC" background="#eb6834" hotkey="2"/>
    <Label value="Platelets" background="#1baf7a" hotkey="3"/>
  </RectangleLabels>
</View>"""

INSTRUCTIONS = """<h3>Correct the model's boxes</h3>
<p>Each tile opens with the model's boxes: blue white cells (1), orange red cells (2), green platelets (3).
Make every cell in the tile have exactly one box of the right class, then <b>Submit</b> (Ctrl+Enter).</p>
<p style="margin-top:10px">&bull; <b>Box every cell</b> that is at least about half visible, including cells cut off
by the tile's edge or the dark edge of the field. Leave out slivers.</p>
<p>&bull; <b>Overlapping cells</b> get one box each, around the whole cell as you would trace its outline.</p>
<p>&bull; <b>White cells</b>: every kind, and the box takes in the cytoplasm, not only the nucleus. Watch the small
round dark-violet ones the size of a red cell: lymphocytes, which the model often calls red cells.</p>
<p>&bull; <b>Platelets</b>: one box per platelet you can tell apart, also when one lies on a red cell.</p>
<p>&bull; Stain deposits, dust and smudges get no box.</p>
<p style="margin-top:10px">Fix a box: drag its corners. Change its class: select it, then press 1, 2 or 3. Delete
it: select it, then Backspace. Draw one: press 1, 2 or 3, then drag. Undo: Ctrl+Z.</p>
<p>A tile too blurred or too crowded to label: press <b>Skip</b>. These instructions: the Instructions button.</p>"""


def studio_executable() -> str:
    found = shutil.which("label-studio") or str(Path.home() / ".local" / "bin" / "label-studio")
    if not Path(found).exists():
        raise SystemExit("Label Studio is not installed: run `uv tool install --python 3.12 label-studio`")
    return found


def credentials() -> dict:
    path = STUDIO_DIR / "credentials.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    STUDIO_DIR.mkdir(parents=True, exist_ok=True)
    creds = {"username": f"{os.environ.get('USER', 'labeler')}@localhost", "password": secrets.token_urlsafe(9),
             "token": secrets.token_hex(20)}
    write_json(path, creds)
    path.chmod(0o600)
    return creds


class Studio:
    """The few Label Studio API calls this script needs, authenticated with the account's legacy token."""

    def __init__(self, port: int, token: str):
        import requests

        self.base = f"http://localhost:{port}"
        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Token {token}"

    def call(self, method: str, path: str, **kwargs):
        response = self.http.request(method, self.base + path, timeout=300, **kwargs)
        if not response.ok:
            raise SystemExit(f"Label Studio {method} {path}: HTTP {response.status_code} {response.text[:400]}")
        return response.json() if response.content else None

    def up(self) -> bool:
        """Whether Label Studio, and not some other server on the port, answers."""
        import requests

        try:  # /version answers JSON wrapped in <pre> for a browser-like Accept header
            response = self.http.get(self.base + "/version", timeout=3)
            return response.ok and '"label-studio-os-package"' in response.text
        except requests.RequestException:
            return False

    def project(self) -> dict | None:
        found = self.call("GET", "/api/projects", params={"page_size": 1000, "title": PROJECT_TITLE})
        projects = found["results"] if isinstance(found, dict) else found
        return next((p for p in projects if p["title"] == PROJECT_TITLE), None)

    def tasks(self, project_id: int, annotated_only: bool) -> list[dict]:
        params = {"exportType": "JSON"} | ({} if annotated_only else {"download_all_tasks": "true"})
        return self.call("GET", f"/api/projects/{project_id}/export", params=params)


def ensure_project(studio: Studio, model_version: str) -> dict:
    """The labeling project, created on the first start, with any tasks from tasks.json it does not hold yet."""
    project = studio.project()
    if project is None:
        project = studio.call("POST", "/api/projects", json={
            "title": PROJECT_TITLE, "label_config": LABEL_CONFIG, "expert_instruction": INSTRUCTIONS,
            "show_instruction": True, "show_skip_button": True, "enable_empty_annotation": True,
            "maximum_annotations": 1, "show_collab_predictions": True, "model_version": model_version,
            "description": "Tiles of my blood smear pictures, pre-labeled by the model (scripts/labeling.py).",
        })
        studio.call("POST", "/api/storages/localfiles", json={
            "project": project["id"], "title": "Tiles", "path": str(TILES_DIR), "use_blob_urls": True,
            "regex_filter": r".*\.png",
        })
    tasks_file = LABEL_DIR / "tasks.json"
    if not tasks_file.exists():
        raise SystemExit("No tasks yet: run `make label-prepare` first")
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    present = {t["data"].get("tile") for t in studio.tasks(project["id"], annotated_only=False)}
    new = [t for t in tasks if t["data"]["tile"] not in present]
    if new:
        studio.call("POST", f"/api/projects/{project['id']}/import", json=new)
        print(f"Added {len(new)} tasks to the project ({len(present) + len(new)} in all).")
    return project


def port_in_use(port: int) -> bool:
    import socket

    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def show_login(studio: Studio, project: dict, creds: dict) -> str:
    url = f"{studio.base}/projects/{project['id']}/data"
    print(f"\nLabel Studio: {url}\n  login: {creds['username']}\n  password: {creds['password']}\n"
          "Open the project, press 'Label All Tasks', correct each tile and Submit.", flush=True)
    return url


def serve(args) -> None:
    executable = studio_executable()
    creds = credentials()
    cfg = load_config()
    studio = Studio(args.port, creds["token"])
    if studio.up():  # already running, from an earlier `make label`
        show_login(studio, ensure_project(studio, f"{cfg['run_name']}-prelabel"), creds)
        return
    if port_in_use(args.port):
        raise SystemExit(f"Port {args.port} is used by another program: choose another one, for example "
                         f"`make label LABEL_PORT={args.port + 1}`")
    env = os.environ | {
        "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
        "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(DOCUMENT_ROOT),
        "LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN": "true",
        "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK": "true",
    }
    command = [executable, "start", "--no-browser", "--port", str(args.port), "--data-dir", str(STUDIO_DIR),
               "--username", creds["username"], "--password", creds["password"], "--user-token", creds["token"],
               "--enable-legacy-api-token", "--internal-host", "127.0.0.1"]  # reachable from this computer only
    server = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL if not args.verbose else None,
                              stderr=subprocess.STDOUT if not args.verbose else None)
    try:
        for _ in range(240):  # the first start migrates its database
            if studio.up():
                break
            if server.poll() is not None:
                raise SystemExit(f"Label Studio stopped (exit code {server.returncode}); rerun with --verbose")
            time.sleep(1)
        else:
            raise SystemExit("Label Studio did not start within 4 minutes; rerun with --verbose")
        project = ensure_project(studio, f"{cfg['run_name']}-prelabel")
        url = show_login(studio, project, creds)
        print("Ctrl+C here stops the server; labels are kept.", flush=True)
        if args.open:
            import webbrowser

            webbrowser.open(url)
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()


# ---------------------------------------------------------------- export


def yolo_lines(annotation: dict) -> list[str]:
    lines = []
    for r in annotation.get("result", []):
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        cls = CLASS_NAMES.index(v["rectanglelabels"][0])
        x1, y1 = max(v["x"], 0.0) / 100, max(v["y"], 0.0) / 100
        x2, y2 = min(v["x"] + v["width"], 100.0) / 100, min(v["y"] + v["height"], 100.0) / 100
        if x2 > x1 and y2 > y1:
            lines.append(f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return lines


def box_set(result: list[dict]) -> list[tuple]:
    """A region list as sorted (class, x, y, width, height) rows, to compare an annotation with its pre-labels."""
    return sorted((r["value"]["rectanglelabels"][0], *(round(r["value"][k], 2) for k in ("x", "y", "width", "height")))
                  for r in result if r.get("type") == "rectanglelabels")


def assign_splits(rows: list[dict]) -> dict[str, str]:
    """Split by picture: in each session (pictures taken less than 5 minutes apart), every fifth picture is val.

    Frames of one video are grouped in 10-second runs, so neighbouring frames of the same field stay together.
    """
    groups = {}
    for r in rows:
        source = r["source"]
        if "?t=" in source:
            path, t = source.split("?t=")
            source = f"{path}#{int(float(t) // 10)}"
        groups.setdefault(source, r["taken"])
    split, session, last = {}, 0, None
    index_in_session = 0
    for source, taken in sorted(groups.items(), key=lambda kv: kv[1]):
        moment = datetime.fromisoformat(taken)
        if last is not None and moment - last > timedelta(minutes=5):
            session, index_in_session = session + 1, 0
        split[source] = "val" if index_in_session % 5 == 2 else "train"
        index_in_session, last = index_in_session + 1, moment
    return {r["tile"]: split[r["source"] if "?t=" not in r["source"] else
                             f"{r['source'].split('?t=')[0]}#{int(float(r['source'].split('?t=')[1]) // 10)}"]
            for r in rows}


def export(args) -> None:
    creds = credentials()
    studio = Studio(args.port, creds["token"])
    if not studio.up():
        raise SystemExit("Label Studio is not running: start it with `make label` in another terminal")
    project = studio.project()
    if project is None:
        raise SystemExit("No labeling project yet: run `make label` first")
    with (LABEL_DIR / "tiles.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    splits = assign_splits(rows)
    prelabels = {t["data"]["tile"]: t["predictions"][0]["result"]
                 for t in json.loads((LABEL_DIR / "tasks.json").read_text(encoding="utf-8"))}
    done, skipped, unchanged = {}, 0, []
    for task in studio.tasks(project["id"], annotated_only=True):
        annotations = sorted(task.get("annotations", []), key=lambda a: a.get("updated_at") or "")
        if not annotations:
            continue
        latest = annotations[-1]
        if latest.get("was_cancelled"):
            skipped += 1
            continue
        tile = task["data"]["tile"]
        if not args.keep_unchanged and tile in prelabels and box_set(latest["result"]) == box_set(prelabels[tile]):
            unchanged.append(task["id"])  # submitted as the model drew it: possibly never reviewed
            continue
        done[tile] = yolo_lines(latest)
    if MINE_DIR.exists():
        shutil.rmtree(MINE_DIR)
    counts = {s: {"tiles": 0, **{c: 0 for c in CLASS_NAMES}} for s in ("train", "val")}
    for tile, lines in done.items():
        split = splits.get(tile, "train")
        (MINE_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (MINE_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(TILES_DIR / tile, MINE_DIR / "images" / split / tile)
        (MINE_DIR / "labels" / split / f"{Path(tile).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split]["tiles"] += 1
        for line in lines:
            counts[split][CLASS_NAMES[int(line.split()[0])]] += 1
    (MINE_DIR / "data.yaml").write_text(  # an absolute path: Ultralytics reads a relative one from its datasets dir
        f"path: {MINE_DIR}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_NAMES)), encoding="utf-8")
    write_json(MINE_DIR / "summary.json", {"tiles": len(done), "skipped_tiles": skipped,
                                           "unchanged_tasks_left_out": sorted(unchanged), "splits": counts})
    for split, c in counts.items():
        print(f"{split}: {c['tiles']} tiles, " + ", ".join(f"{c[k]} {k}" for k in CLASS_NAMES))
    if unchanged:
        print(f"Left out {len(unchanged)} tiles submitted exactly as pre-labeled (tasks {', '.join(map(str, sorted(unchanged)))});"
              " --keep-unchanged keeps them.")
    print(f"{len(done)} labeled tiles ({skipped} skipped as unusable) written to {shown(MINE_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="cut the pictures into pre-labeled tiles")
    p.add_argument("--source", type=Path, default=SOURCE_DIR)
    p.add_argument("--weights", type=Path, help="model that pre-labels (default: the trained run's weights)")
    p.add_argument("--per-photo", type=int, default=3, help="tiles taken from each photo")
    p.add_argument("--per-frame", type=int, default=2, help="tiles taken from each video frame")
    s = sub.add_parser("serve", help="run Label Studio with the tasks")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--open", action="store_true", help="open the project in the browser")
    s.add_argument("--verbose", action="store_true", help="show Label Studio's own output")
    e = sub.add_parser("export", help="write the submitted labels as a YOLO dataset in data/UIC_microscope_photos")
    e.add_argument("--port", type=int, default=DEFAULT_PORT)
    e.add_argument("--keep-unchanged", action="store_true",
                   help="also use tiles submitted exactly as pre-labeled (left out by default: maybe never reviewed)")
    args = parser.parse_args()
    {"prepare": prepare, "serve": serve, "export": export}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
