"""Detect and count blood cells in your own smear images with the trained model.

    python scripts/predict.py IMAGES [--weights models/<run>.pt] [--conf C] [--out predictions] [--device cpu|0]

IMAGES is an image file, a directory or a glob. Prints the WBC, RBC and platelet
counts per image, and writes annotated images, YOLO-format label files and
counts.csv under --out. The default confidence threshold is the operating point
chosen on the validation split (results/metrics.json). Inference is the one the
evaluation used (predict_with_margin in kaggle/train_kernel.py), so the counts
behave as reported. Needs requirements-ml.txt.
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


def operating_conf() -> float:
    try:
        return json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))["operating_point"]["val"]["conf"]
    except (FileNotFoundError, KeyError):
        return FALLBACK_CONF


def image_paths(source: str) -> list[Path]:
    """Image files named by a file, a directory or a glob, sorted."""
    path = Path(source)
    if path.is_dir():
        candidates = sorted(path.iterdir())
    elif path.is_file():
        candidates = [path]
    else:
        candidates = sorted(Path(p) for p in glob.glob(source, recursive=True))
    return [p for p in candidates if p.suffix.lower() in IMAGE_SUFFIXES]


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="image file, directory or glob")
    parser.add_argument("--weights", type=Path, default=MODELS_DIR / f"{cfg['run_name']}.pt")
    parser.add_argument("--conf", type=float, help="confidence threshold (default: operating point from validation)")
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--device", help="'cpu' or a GPU index such as '0' (default: automatic)")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    paths = image_paths(args.source)
    if not paths:
        raise SystemExit(f"No images found at {args.source}")
    conf = args.conf if args.conf is not None else operating_conf()
    out = args.out.resolve()
    (out / "labels").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    rows = []
    for r in predict_with_margin(model, paths, conf, cfg["train"]["imgsz"], cfg["train"]["batch"], args.device):
        stem = Path(r.path).stem
        cv2.imwrite(str(out / f"{stem}.jpg"), r.plot())
        r.save_txt(out / "labels" / f"{stem}.txt")
        counts = Counter(CLASS_NAMES[int(c)] for c in r.boxes.cls.tolist())
        rows.append({"image": Path(r.path).name, **{c: counts.get(c, 0) for c in CLASS_NAMES}})
        print(f"{Path(r.path).name}: " + ", ".join(f"{counts.get(c, 0)} {c}" for c in CLASS_NAMES))
    with (out / "counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", *CLASS_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} images at conf ≥ {conf:.2f}; annotated images, labels and counts.csv in {out}")


if __name__ == "__main__":
    main()
