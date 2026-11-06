"""Detect and count cells in a microscope video, frame by frame, and save it with the boxes drawn.

    python scripts/predict_video.py VIDEO [--weights models/<run>.pt] [--conf C] [--out predictions] [--device 0]

The red cells are measured once, on the sharpest frame of each eighth of the video (the magnification does not
change while filming); every frame then goes through scripts/smear.py at that scale, on the GPU in 16-bit floats,
and its processing time is recorded. Two videos are written to --out, with the counts and the detector's speed on
each frame:

* <name>_every_frame.mp4: each frame with its own detections, as if the detector kept up with the camera;
* <name>_realtime.mp4: what a live view would show on this computer. The detector takes the newest frame each time
  it is free and its boxes stay up until the next result, so they refresh at the speed it reaches and lag behind a
  moving slide, as in the app's Live camera tab.

<name>_frames.csv has the counts and the time of every frame. Uses the fine-tuned model by default, at the threshold
chosen on my val tiles (results/finetune.json).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np

import smear
from common import CLASS_NAMES, MODELS_DIR, load_config
from predict import COLORS_BGR, FALLBACK_CONF, finetuned_conf


def sharpness_per_frame(path: Path) -> list[float]:
    import cv2

    cap, scores = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        r = min(h, w) // 4
        scores.append(float(cv2.Laplacian(gray[h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r], cv2.CV_64F).var()))
    cap.release()
    return scores


def frames(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))  # OpenCV applies the rotation the phone recorded
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def measure(model, path: Path, scores: list[float], device) -> float:
    """Median red cell diameter over the sharpest frame of each eighth of the video."""
    bounds = np.linspace(0, len(scores), 9).astype(int)
    picks = {int(a + np.argmax(scores[a:b])) for a, b in zip(bounds, bounds[1:]) if b > a}
    sizes = []
    for i, frame in enumerate(frames(path)):
        if i in picks:
            size = smear.red_cell_size(model, frame, smear.field_mask(frame), device)
            if size:
                sizes.append(size)
    if not sizes:
        raise SystemExit("No red cells found to measure in the video")
    return float(np.median(sizes))


def encoder(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    """An ffmpeg process turning raw BGR frames into an H.264 MP4 that browsers and editors play."""
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
         "-r", f"{fps:.5f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def draw(frame: np.ndarray, boxes: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    thickness = max(2, round(w / 540))
    for x1, y1, x2, y2, _, cls in sorted(boxes.tolist(), key=lambda b: b[5] == 1, reverse=True):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), COLORS_BGR[CLASS_NAMES[int(cls)]], thickness)
    scale = w / 1100
    y = int(46 * scale)
    for text in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, round(2 * scale)))
        cv2.rectangle(out, (int(14 * scale), y - th - int(10 * scale)), (int(30 * scale) + tw, y + int(10 * scale)),
                      (0, 0, 0), -1)
        cv2.putText(out, text, (int(22 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    max(1, round(2 * scale)), cv2.LINE_AA)
        y += th + int(26 * scale)
    return out


def count_line(boxes: np.ndarray) -> str:
    return "   ".join(f"{c} {int((boxes[:, 5] == k).sum())}" for k, c in enumerate(CLASS_NAMES))


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, help="default: the fine-tuned model, else the trained one")
    parser.add_argument("--conf", type=float, help="confidence threshold (default: chosen on my val tiles)")
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--device", default=None, help="'cpu' or a GPU index (default: GPU 0 when there is one)")
    args = parser.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    tuned = MODELS_DIR / f"{cfg['finetune']['run_name']}.pt"
    weights = args.weights or (tuned if tuned.exists() else MODELS_DIR / f"{cfg['run_name']}.pt")
    conf = args.conf if args.conf is not None else (finetuned_conf() if weights == tuned else FALLBACK_CONF)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    half = device != "cpu"
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(f"{args.video.name}: measuring the red cells on sharp frames…", flush=True)
    scores = sharpness_per_frame(args.video)
    size = measure(model, args.video, scores, device)
    print(f"red cell {size:.0f} px; detecting in {len(scores)} frames on "
          f"{torch.cuda.get_device_name(0) if half else 'the CPU'}…", flush=True)

    results = []
    for i, frame in enumerate(frames(args.video)):
        if i == 0:  # warm up, so that the first frame's time is not the GPU's start-up
            smear.detect(model, frame, conf=conf, device=device, red_cell_px=size, half=half)
        start = time.perf_counter()
        found = smear.detect(model, frame, conf=conf, device=device, red_cell_px=size, half=half)
        results.append((found["boxes"], time.perf_counter() - start, found["tiles"]))
        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(scores)} frames, {np.mean([r[1] for r in results]) * 1000:.0f} ms per frame",
                  flush=True)
    seconds = np.array([r[1] for r in results])
    speed = f"detector {1 / seconds.mean():.1f} frames/s ({seconds.mean() * 1000:.0f} ms, {results[0][2]} tiles)"

    stem = args.video.stem
    h, w = next(frames(args.video)).shape[:2]
    every, live = encoder(out_dir / f"{stem}_every_frame.mp4", w, h, fps), encoder(out_dir / f"{stem}_realtime.mp4",
                                                                                  w, h, fps)
    shown, pending = None, None  # the result on screen; the frame being detected and when it finishes
    with (out_dir / f"{stem}_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", *CLASS_NAMES, "detect_ms"])
        for i, frame in enumerate(frames(args.video)):
            if i >= len(results):
                break
            now = i / fps
            boxes, took, _ = results[i]
            writer.writerow([i, round(now, 3), *(int((boxes[:, 5] == k).sum()) for k in range(len(CLASS_NAMES))),
                             round(took * 1000, 1)])
            every.stdin.write(draw(frame, boxes, [count_line(boxes), f"every frame analysed | {speed}"]).tobytes())
            if pending and pending[1] <= now:
                shown, pending = pending[0], None
            if pending is None:  # the detector is free: it takes the newest frame
                pending = (i, now + took)
            if shown is None:
                lines = ["detecting the first frame...", f"real time on this computer | {speed}"]
                live_boxes = np.zeros((0, 6), np.float32)
            else:
                live_boxes = results[shown][0]
                lines = [count_line(live_boxes),
                         f"real time on this computer | boxes from {1000 * (now - shown / fps):.0f} ms ago"]
            live.stdin.write(draw(frame, live_boxes, lines).tobytes())
    for proc in (every, live):
        proc.stdin.close()
        proc.wait()
    print(f"{speed}; videos and {stem}_frames.csv in {out_dir}")


if __name__ == "__main__":
    main()
