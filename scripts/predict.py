"""Detect and count blood cells in your own smear images with the trained model.

    python scripts/predict.py IMAGES [--smear] [--weights models/<run>.pt] [--conf C] [--out predictions]
                              [--device cpu|0]

IMAGES is an image file, a directory or a glob. Prints the WBC, RBC and platelet
counts per image, and writes annotated images, YOLO-format label files and
counts.csv under --out.

By default each image goes whole through the model, as TXL-PBC's crops did: the
inference the evaluation used (predict_with_margin in kaggle/train_kernel.py), at
the operating point chosen on the validation split (results/metrics.json).

--smear is for whole microscope pictures at any magnification, such as phone photos
through the eyepiece (HEIC too): the model measures the red cells, the picture is
rescaled so that they are as large as in the fine-tuning tiles, and predicted tile
by tile (scripts/smear.py). It uses the fine-tuned model (models/<finetune.run_name>.pt)
at the threshold that maximized F1 on my val tiles (results/finetune.json).
Needs requirements-ml.txt.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from common import CLASS_NAMES, KERNEL_SOURCE, MODELS_DIR, RESULTS_DIR, load_config

sys.path.insert(0, str(KERNEL_SOURCE.parent))
from train_kernel import predict_with_margin  # noqa: E402

FALLBACK_CONF = 0.25
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}  # formats OpenCV reads
PHONE_SUFFIXES = {".heic", ".heif"}  # read through pi-heif in --smear mode
COLORS_BGR = {"WBC": (214, 120, 42), "RBC": (52, 104, 235), "Platelets": (122, 175, 27)}  # the app's colors


def operating_conf() -> float:
    try:
        return json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))["operating_point"]["val"]["conf"]
    except (FileNotFoundError, KeyError):
        return FALLBACK_CONF


def finetuned_conf() -> float:
    try:
        return json.loads((RESULTS_DIR / "finetune.json").read_text(encoding="utf-8"))["fine_tuned"]["my_val"][
            "f1_optimal_conf"]
    except (FileNotFoundError, KeyError):
        return FALLBACK_CONF


def load_bgr(path: Path):
    """A picture as a BGR array, upright as the camera shows it (HEIC through pi-heif)."""
    import numpy as np
    from PIL import Image, ImageOps

    if path.suffix.lower() in PHONE_SUFFIXES:
        from pi_heif import register_heif_opener

        register_heif_opener()
    return np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))[:, :, ::-1])


def predict_smears(paths: list[Path], weights: Path, conf: float, device, out: Path) -> list[dict]:
    """--smear: each picture measured, rescaled and predicted tile by tile; annotated copies and labels written."""
    import cv2
    from ultralytics import YOLO

    import smear

    model = YOLO(str(weights))
    rows = []
    for path in paths:
        img = load_bgr(path)
        found = smear.detect(model, img, conf=conf, device=device)
        boxes = found["boxes"]
        h, w = img.shape[:2]
        thickness = max(2, round(max(h, w) / 900))
        for x1, y1, x2, y2, _, cls in sorted(boxes.tolist(), key=lambda b: b[5] == 1, reverse=True):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), COLORS_BGR[CLASS_NAMES[int(cls)]], thickness)
        cv2.imwrite(str(out / f"{path.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out / "labels" / f"{path.stem}.txt").write_text("".join(
            f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
            for x1, y1, x2, y2, _, c in boxes.tolist()), encoding="utf-8")
        counts = Counter(CLASS_NAMES[int(c)] for c in boxes[:, 5].tolist())
        size = found["red_cell_px"]
        rows.append({"image": path.name, **{c: counts.get(c, 0) for c in CLASS_NAMES},
                     "red_cell_px": round(size, 1) if size else "", "tiles": found["tiles"]})
        print(f"{path.name}: " + ", ".join(f"{counts.get(c, 0)} {c}" for c in CLASS_NAMES)
              + f"  (red cell {size:.0f} px, {found['tiles']} tiles)" if size else "  (no red cells to measure)")
    return rows


def image_paths(source: str, suffixes: set[str]) -> list[Path]:
    """Image files named by a file, a directory or a glob, sorted."""
    path = Path(source)
    if path.is_dir():
        candidates = sorted(path.iterdir())
    elif path.is_file():
        candidates = [path]
    else:
        candidates = sorted(Path(p) for p in glob.glob(source, recursive=True))
    return [p for p in candidates if p.suffix.lower() in suffixes]


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="image file, directory or glob")
    parser.add_argument("--smear", action="store_true",
                        help="whole microscope pictures at any magnification: rescaled and predicted tile by tile")
    parser.add_argument("--weights", type=Path, help="default: the trained model, or the fine-tuned one with --smear")
    parser.add_argument("--conf", type=float, help="confidence threshold (default: the one chosen on validation)")
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--device", help="'cpu' or a GPU index such as '0' (default: automatic)")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    paths = image_paths(args.source, IMAGE_SUFFIXES | (PHONE_SUFFIXES if args.smear else set()))
    if not paths:
        raise SystemExit(f"No images found at {args.source}")
    out = args.out.resolve()
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if args.smear:
        tuned = MODELS_DIR / f"{cfg['finetune']['run_name']}.pt"
        weights = args.weights or (tuned if tuned.exists() else MODELS_DIR / f"{cfg['run_name']}.pt")
        conf = args.conf if args.conf is not None else (finetuned_conf() if weights == tuned else FALLBACK_CONF)
        rows = predict_smears(paths, weights, conf, args.device, out)
    else:
        conf = args.conf if args.conf is not None else operating_conf()
        model = YOLO(str(args.weights or MODELS_DIR / f"{cfg['run_name']}.pt"))
        rows = []
        for r in predict_with_margin(model, paths, conf, cfg["train"]["imgsz"], cfg["train"]["batch"], args.device):
            stem = Path(r.path).stem
            cv2.imwrite(str(out / f"{stem}.jpg"), r.plot())
            r.save_txt(out / "labels" / f"{stem}.txt")
            counts = Counter(CLASS_NAMES[int(c)] for c in r.boxes.cls.tolist())
            rows.append({"image": Path(r.path).name, **{c: counts.get(c, 0) for c in CLASS_NAMES}})
            print(f"{Path(r.path).name}: " + ", ".join(f"{counts.get(c, 0)} {c}" for c in CLASS_NAMES))
    with (out / "counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} images at conf ≥ {conf:.2f}; annotated images, labels and counts.csv in {out}")


if __name__ == "__main__":
    main()
