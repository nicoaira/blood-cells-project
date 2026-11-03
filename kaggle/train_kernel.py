#!/usr/bin/env python3
"""Fine-tune YOLO26 on TXL-PBC blood-smear images (Kaggle GPU job).

scripts/kaggle_push_kernel.py pushes this file to Kaggle after replacing the
EMBEDDED_CONFIG placeholder with config/train.yaml (serialized as JSON). The same
file runs locally as a quick CPU smoke test of the whole pipeline:

    python kaggle/train_kernel.py --config config/train.yaml \
        --data-root data/TXL-PBC --out smoke --smoke

Progress is reported as one JSON object per line on stdout, prefixed with
EVENT_PREFIX. scripts/monitor.py follows the Kaggle log stream, parses these
events and turns them into live dashboard data. Every event is also appended
to <out>/artifacts/events.jsonl so the run can be rebuilt from the outputs.

Outputs under --out (/kaggle/working on Kaggle):
    runs/train/              Ultralytics training run (results.csv, plots, weights/best.pt)
    eval/<split>/            Ultralytics validation plots for each evaluated split
    artifacts/summary.json   run metadata, environment, training summary, headline metrics
    artifacts/evaluation.json per-split metrics, per-class table, confusion matrix, curves
    artifacts/predictions.json boxes predicted on the val and test images
    artifacts/events.jsonl   every progress event emitted during the run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

EVENT_PREFIX = "@@BC@@ "
EMBEDDED_CONFIG = "__EMBEDDED_CONFIG__"  # replaced with JSON by scripts/kaggle_push_kernel.py
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
BATCH_EVENT_INTERVAL_S = 5.0
CURVE_POINTS = 101
PREDICT_MARGIN = 16  # gray border around each image when predicting, in network pixels (see predict_with_margin)

# Tiny settings that exercise every stage of the pipeline on a CPU in a few minutes.
SMOKE_OVERRIDES = {
    "model": "yolo26n.pt",
    "train": {"epochs": 2, "imgsz": 320, "batch": 8, "workers": 2, "fraction": 0.1, "patience": 5, "close_mosaic": 1},
}

_events_file: Path | None = None


def _json_default(obj):
    """Serialize numpy scalars/arrays, tensors and paths found in event payloads."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def emit(kind: str, **payload) -> None:
    """Print a machine-readable progress event and append it to events.jsonl."""
    line = json.dumps({"type": kind, "t": round(time.time(), 3), **payload}, default=_json_default)
    print(EVENT_PREFIX + line, flush=True)
    if _events_file is not None:
        with _events_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def stage(name: str, message: str = "") -> None:
    emit("stage", stage=name, message=message)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


# --------------------------------------------------------------------------- setup


def load_config(args) -> dict:
    if args.config:
        import yaml

        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    elif EMBEDDED_CONFIG.startswith("{"):
        cfg = json.loads(EMBEDDED_CONFIG)
    else:
        raise SystemExit("No configuration: pass --config or push the kernel with scripts/kaggle_push_kernel.py")
    if args.smoke:
        cfg["model"] = SMOKE_OVERRIDES["model"]
        cfg["train"] = {**cfg["train"], **SMOKE_OVERRIDES["train"]}
    return cfg


def install_ultralytics(version: str) -> None:
    """Install the pinned Ultralytics release into the Kaggle image (torch is preinstalled)."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-input", f"ultralytics=={version}"]
    subprocess.run(cmd, check=True)


def describe_environment() -> dict:
    import torch
    import ultralytics

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(
            cuda=torch.version.cuda,
            gpu=props.name,
            gpu_count=torch.cuda.device_count(),
            gpu_mem_gb=round(props.total_memory / 1e9, 1),
        )
    return env


def find_dataset_root(data_root: str | None, work_dir: Path) -> Path:
    """Return a directory holding data.yaml, images/ and labels/.

    Locally this is --data-root. On Kaggle the attached dataset is searched for
    under /kaggle/input; if Kaggle kept the upload as a zip archive it is extracted.
    """
    if data_root:
        return Path(data_root).resolve()
    hits = sorted(glob.glob("/kaggle/input/**/data.yaml", recursive=True))
    if hits:
        return Path(hits[0]).parent
    archives = sorted(glob.glob("/kaggle/input/**/*.zip", recursive=True))
    if not archives:
        raise FileNotFoundError("No data.yaml or zip archive found under /kaggle/input")
    extract_dir = work_dir / "extracted"
    with zipfile.ZipFile(archives[0]) as zf:
        zf.extractall(extract_dir)
    hits = sorted(glob.glob(f"{extract_dir}/**/data.yaml", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{archives[0]} does not contain data.yaml")
    return Path(hits[0]).parent


def prepare_data(source: Path, work_dir: Path) -> tuple[Path, dict]:
    """Copy the dataset to a writable directory and write a data.yaml with absolute paths.

    Ultralytics writes label caches next to the label files, and /kaggle/input is
    read-only, so training always reads from the copy.
    """
    import yaml

    names = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))["names"]
    data_dir = work_dir / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    for sub in ("images", "labels"):
        shutil.copytree(source / sub, data_dir / sub)
    counts = {split: len(list((data_dir / "images" / split).glob("*.png"))) for split in ("train", "val", "test")}
    data_yaml = work_dir / "data.yaml"
    spec = {
        "path": str(data_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(names)),
    }
    data_yaml.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return data_yaml, {"source": str(source), "images": counts, "names": names}


# --------------------------------------------------------------------------- training


class ProgressTracker:
    """Ultralytics callbacks that turn trainer state into progress events."""

    def __init__(self):
        self.batch_i = 0
        self.last_batch_event = 0.0
        self.loop_done = False

    def callbacks(self) -> dict:
        return {
            "on_train_start": self.on_train_start,
            "on_train_epoch_start": self.on_train_epoch_start,
            "on_train_batch_end": self.on_train_batch_end,
            "on_train_epoch_end": self.on_train_epoch_end,
            "on_fit_epoch_end": self.on_fit_epoch_end,
            "on_train_end": self.on_train_end,
        }

    @staticmethod
    def _gpu_mem_gb() -> float:
        import torch

        return round(torch.cuda.memory_reserved() / 1e9, 2) if torch.cuda.is_available() else 0.0

    @staticmethod
    def _strip(d: dict | None) -> dict:
        """{'train/box_loss': x, 'metrics/mAP50(B)': y} -> {'box_loss': x, 'mAP50': y}."""
        return {k.split("/", 1)[-1].removesuffix("(B)"): round(float(v), 6) for k, v in (d or {}).items()}

    def on_train_start(self, trainer):
        from ultralytics.utils.torch_utils import get_flops, get_num_params

        opt = trainer.optimizer
        emit(
            "train_start",
            epochs=trainer.epochs,
            batches_per_epoch=len(trainer.train_loader),
            batch_size=trainer.batch_size,
            imgsz=trainer.args.imgsz,
            train_images=len(trainer.train_loader.dataset),
            val_images=len(trainer.test_loader.dataset),
            optimizer=type(opt).__name__,
            lr0=opt.param_groups[0].get("initial_lr", opt.param_groups[0]["lr"]),
            params=get_num_params(trainer.model),
            gflops=round(get_flops(trainer.model, trainer.args.imgsz), 1),
            device=str(trainer.device),
            amp=bool(trainer.amp),
            save_dir=str(trainer.save_dir),
        )

    def on_train_epoch_start(self, trainer):
        self.batch_i = 0
        emit("epoch_start", epoch=trainer.epoch + 1, epochs=trainer.epochs)

    def on_train_batch_end(self, trainer):
        self.batch_i += 1
        nb = len(trainer.train_loader)
        now = time.time()
        if now - self.last_batch_event < BATCH_EVENT_INTERVAL_S and self.batch_i < nb:
            return
        self.last_batch_event = now
        emit(
            "batch",
            epoch=trainer.epoch + 1,
            epochs=trainer.epochs,
            batch=self.batch_i,
            batches=nb,
            loss=self._strip(trainer.tloss),
            gpu_mem_gb=self._gpu_mem_gb(),
        )

    def on_train_epoch_end(self, trainer):
        emit("validating", epoch=trainer.epoch + 1)

    def _class_metrics(self, trainer) -> dict:
        metrics = getattr(trainer.validator, "metrics", None)
        if metrics is None or not len(metrics.box.ap_class_index):
            return {}
        box, names = metrics.box, trainer.data["names"]
        return {
            names[int(c)]: {
                "precision": round(float(box.p[i]), 6),
                "recall": round(float(box.r[i]), 6),
                "mAP50": round(float(box.ap50[i]), 6),
                "mAP50-95": round(float(box.ap[i]), 6),
            }
            for i, c in enumerate(box.ap_class_index)
        }

    def on_fit_epoch_end(self, trainer):
        metrics = {k: v for k, v in (trainer.metrics or {}).items() if not k.startswith("val/")}
        val_loss = {k: v for k, v in (trainer.metrics or {}).items() if k.startswith("val/")}
        if self.loop_done:  # Ultralytics re-fires this callback after validating best.pt in final_eval()
            emit("final_val", metrics=self._strip(metrics), per_class=self._class_metrics(trainer))
            return
        emit(
            "epoch",
            epoch=trainer.epoch + 1,
            epochs=trainer.epochs,
            epoch_time=round(float(trainer.epoch_time or 0), 2),
            elapsed=round(time.time() - trainer.train_time_start, 1),
            train_loss=self._strip(trainer.label_loss_items(trainer.tloss, prefix="train")),
            val_loss=self._strip(val_loss),
            metrics=self._strip(metrics),
            per_class=self._class_metrics(trainer),
            lr=self._strip(trainer.lr),
            fitness=round(float(trainer.fitness or 0), 6),
            best_fitness=round(float(trainer.best_fitness or 0), 6),
            best_epoch=int(trainer.stopper.best_epoch),
            patience=int(trainer.stopper.patience),
            gpu_mem_gb=self._gpu_mem_gb(),
        )
        if trainer.stop:
            self.loop_done = True

    def on_train_end(self, trainer):
        emit(
            "train_end",
            epochs_completed=trainer.epoch + 1,
            epochs_planned=trainer.epochs,
            early_stopped=trainer.epoch + 1 < trainer.epochs,
            best_epoch=int(trainer.stopper.best_epoch),
            best_fitness=round(float(trainer.best_fitness or 0), 6),
            train_time_s=round(time.time() - trainer.train_time_start, 1),
        )


def free_gpu_memory() -> None:
    """Collect unreachable objects and return cached CUDA memory to the driver.

    Called after training, whose model, EMA copy and optimizer state are no longer
    needed (evaluation reloads best.pt), and again before predicting.
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(cfg: dict, data_yaml: Path, out_dir: Path, device) -> Path:
    from ultralytics import YOLO

    model = YOLO(cfg["model"])
    for event, fn in ProgressTracker().callbacks().items():
        model.add_callback(event, fn)
    model.train(
        data=str(data_yaml),
        project=str(out_dir / "runs"),
        name="train",
        exist_ok=True,
        device=device,
        plots=True,
        **cfg["train"],
    )
    save_dir = Path(model.trainer.save_dir)
    del model
    free_gpu_memory()
    return save_dir


# --------------------------------------------------------------------------- evaluation


def _downsample(values, n: int = CURVE_POINTS) -> list[float]:
    import numpy as np

    values = np.asarray(values, dtype=float)
    idx = np.linspace(0, len(values) - 1, n).round().astype(int)
    return [round(float(v), 4) for v in values[idx]]


def summarize_metrics(m, with_curves: bool = True) -> dict:
    """Flatten an Ultralytics DetMetrics object into JSON-friendly data."""
    import numpy as np
    from ultralytics.utils.metrics import smooth

    box, names = m.box, m.names
    out = {
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP75": float(box.map75),
            "mAP50-95": float(box.map),
            "fitness": float(m.fitness),
        },
        "per_class": [
            {
                "class": row["Class"],
                "images": int(row["Images"]),
                "instances": int(row["Instances"]),
                "precision": float(row["Box-P"]),
                "recall": float(row["Box-R"]),
                "f1": float(row["Box-F1"]),
                "mAP50": float(row["mAP50"]),
                "mAP75": float(row["mAP75"]),
                "mAP50-95": float(row["mAP50-95"]),
            }
            for row in m.summary()
        ],
        "speed_ms": {k: round(float(v), 3) for k, v in m.speed.items()},
    }
    if len(box.f1_curve):
        # Same rule Ultralytics uses to pick the operating point for P/R/F1.
        i = int(smooth(np.asarray(box.f1_curve).mean(0), 0.1).argmax())
        out["f1_optimal_conf"] = round(float(box.px[i]), 4)
    if with_curves and len(box.f1_curve):
        class_names = [names[int(c)] for c in box.ap_class_index]
        out["curves"] = {
            "x": _downsample(box.px),
            "precision_recall": dict(zip(class_names, (_downsample(y) for y in box.prec_values))),
            "f1_confidence": dict(zip(class_names, (_downsample(y) for y in box.f1_curve))),
            "precision_confidence": dict(zip(class_names, (_downsample(y) for y in box.p_curve))),
            "recall_confidence": dict(zip(class_names, (_downsample(y) for y in box.r_curve))),
        }
    # No confusion matrix here: Ultralytics builds it at the mAP confidence floor (0.001). The report
    # recomputes it from predictions.json at the operating threshold chosen on the validation split.
    return out


def predict_with_margin(model, paths: list, conf: float, imgsz: int, batch: int, device=None):
    """Predict images in batches and yield one Ultralytics Results per image, in its original pixels.

    Each image first gets a gray border of PREDICT_MARGIN network pixels, the margin that Ultralytics'
    validation letterbox leaves around every image, and is predicted at imgsz + 2 * PREDICT_MARGIN, so
    the image itself is still resized to imgsz. Without the border the model also reports cells cut
    off at the image edge, which TXL-PBC does not annotate.

    The batches are built here because Ultralytics runs a list source as one batch whatever `batch`
    says; kernel versions 1 and 2 ran out of GPU memory pushing all 252 validation images through at
    once. scripts/predict.py uses this function too, so local predictions match the evaluation.
    """
    import cv2
    from ultralytics.engine.results import Results

    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i:i + batch]]
        images = [cv2.imread(p) for p in chunk]
        pads = [round(PREDICT_MARGIN * max(im.shape[:2]) / imgsz) for im in images]
        padded = [cv2.copyMakeBorder(im, d, d, d, d, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                  for im, d in zip(images, pads)]
        results = model.predict(padded, conf=conf, imgsz=imgsz + 2 * PREDICT_MARGIN, device=device, verbose=False)
        for path, im, d, r in zip(chunk, images, pads, results):
            h, w = im.shape[:2]
            boxes = r.boxes.data.clone()  # x1, y1, x2, y2, conf, class in the padded image
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - d).clamp(0, w)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - d).clamp(0, h)
            boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]  # drop boxes inside the border
            yield Results(im, path=path, names=r.names, boxes=boxes, speed=r.speed)


def predict_split(model, images_dir: Path, split: str, conf: float, imgsz: int, batch: int,
                  device) -> tuple[list, dict]:
    """Predict every image of a split and keep boxes above `conf` (normalized xyxy)."""
    records, speeds = [], []
    for r in predict_with_margin(model, sorted(images_dir.glob("*.png")), conf, imgsz, batch, device):
        b = r.boxes
        rows = zip(b.cls.tolist(), b.conf.tolist(), b.xyxyn.tolist())
        records.append(
            {
                "image": Path(r.path).name,
                "split": split,
                "width": int(r.orig_shape[1]),
                "height": int(r.orig_shape[0]),
                "boxes": [[int(c), round(s, 3), *(round(v, 4) for v in xyxy)] for c, s, xyxy in rows],
            }
        )
        speeds.append(r.speed)
    mean_speed = {k: round(sum(s[k] for s in speeds) / len(speeds), 3) for k in speeds[0]} if speeds else {}
    return records, mean_speed


def evaluate(cfg: dict, weights: Path, data_yaml: Path, out_dir: Path, device) -> dict:
    import yaml
    from ultralytics import YOLO

    model = YOLO(str(weights))
    tcfg, ecfg = cfg["train"], cfg["evaluate"]
    val_kwargs = dict(data=str(data_yaml), imgsz=tcfg["imgsz"], batch=tcfg["batch"], device=device)
    results = {}
    for split in ecfg["splits"]:
        m = model.val(split=split, plots=True, project=str(out_dir / "eval"), name=split, exist_ok=True, **val_kwargs)
        results[split] = summarize_metrics(m)
        emit("eval", split=split, overall=results[split]["overall"], per_class=results[split]["per_class"])

    # YOLO26 also ships an NMS-free one-to-one head; report it next to the default NMS path.
    m = model.val(split="test", nms=False, plots=False, project=str(out_dir / "eval"), name="test_nms_free",
                  exist_ok=True, **val_kwargs)
    results["test_nms_free"] = summarize_metrics(m, with_curves=False)
    emit("eval", split="test_nms_free", overall=results["test_nms_free"]["overall"])
    evaluation = {"splits": results, "predict_speed_ms": {}}
    write_json(out_dir / "artifacts" / "evaluation.json", evaluation)  # keep the metrics if predicting fails

    free_gpu_memory()
    data_dir = Path(yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["path"])
    predictions = []
    for split in ecfg["splits"]:
        records, evaluation["predict_speed_ms"][split] = predict_split(
            model, data_dir / "images" / split, split, ecfg["predict_conf"], tcfg["imgsz"], tcfg["batch"], device
        )
        predictions += records
    write_json(out_dir / "artifacts" / "predictions.json", {
        "conf_min": ecfg["predict_conf"],
        "box_format": "class, confidence, x1, y1, x2, y2 (normalized to image size)",
        "images": predictions,
    })
    return evaluation


# --------------------------------------------------------------------------- main


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", help="YAML config (default: the config embedded at push time)")
    p.add_argument("--data-root", help="dataset directory with data.yaml (default: search /kaggle/input)")
    p.add_argument("--out", help="output directory (default: /kaggle/working)")
    p.add_argument("--smoke", action="store_true", help="tiny CPU run that exercises the whole pipeline")
    return p.parse_args()


def main() -> None:
    global _events_file
    args = parse_args()
    cfg = load_config(args)
    out_dir = Path(args.out or "/kaggle/working").resolve()
    work_dir = Path("/kaggle/tmp") if ON_KAGGLE else out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    _events_file = out_dir / "artifacts" / "events.jsonl"
    _events_file.write_text("", encoding="utf-8")
    t0 = time.time()
    emit("hello", run_name=cfg["run_name"], config=cfg, smoke=args.smoke, on_kaggle=ON_KAGGLE)

    try:
        stage("install", "Installing Ultralytics" if ON_KAGGLE else "Using local environment")
        if ON_KAGGLE:
            install_ultralytics(cfg["kaggle"]["ultralytics_version"])
        env = describe_environment()
        emit("env", **env)
        if ON_KAGGLE and not env["cuda_available"]:
            raise RuntimeError("No GPU attached to the Kaggle session; refusing to train on CPU")
        device = 0 if env["cuda_available"] else "cpu"

        stage("data", "Preparing dataset")
        source = find_dataset_root(args.data_root, work_dir)
        data_yaml, data_info = prepare_data(source, work_dir)
        emit("data", **data_info)

        # Weight downloads (pretrained model, AMP check) land in the CWD; keep them out of the outputs.
        os.chdir(work_dir)
        stage("train", f"Fine-tuning {cfg['model']}")
        run_dir = train(cfg, data_yaml, out_dir, device)
        best = run_dir / "weights" / "best.pt"

        stage("evaluate", "Evaluating best.pt on the validation and test splits")
        evaluation = evaluate(cfg, best, data_yaml, out_dir, device)
        write_json(out_dir / "artifacts" / "evaluation.json", evaluation)

        stage("export", "Writing run summary")
        (run_dir / "weights" / "last.pt").unlink(missing_ok=True)  # training finished; best.pt is the deliverable
        summary = {
            "run_name": cfg["run_name"],
            "config": cfg,
            "smoke": args.smoke,
            "environment": env,
            "data": data_info,
            "weights": str(best.relative_to(out_dir)),
            "weights_mb": round(best.stat().st_size / 1e6, 1),
            "total_time_s": round(time.time() - t0, 1),
            "metrics": {split: r["overall"] for split, r in evaluation["splits"].items()},
        }
        write_json(out_dir / "artifacts" / "summary.json", summary)
        emit("done", total_time_s=summary["total_time_s"], metrics=summary["metrics"])
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
