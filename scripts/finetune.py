"""Fine-tune the trained model on my own smear tiles (data/UIC_microscope_photos), on the local GPU, and compare it with the base.

    python scripts/finetune.py [--data data/UIC_microscope_photos] [--epochs N]

Training starts from the TXL-PBC model (config: finetune.base) and sees TXL-PBC's training images together with my
labeled tiles from scripts/labeling.py, repeated finetune.mine_repeat times per epoch so that they weigh about a
third; keeping TXL-PBC in the mix keeps what the model already knows. The other settings are config/train.yaml's
`train`, with the overrides in `finetune.train`. The best epoch is chosen on my val tiles, the target domain.

Then the base and the fine-tuned model are both evaluated on my val tiles (mAP, and per-tile count errors at each
model's F1-optimal confidence) and on TXL-PBC's test split, to check that nothing was forgotten. The weights go to
models/<finetune.run_name>.pt and the comparison to results/finetune.json.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from common import CLASS_NAMES, DATA_DIR, KERNEL_SOURCE, MODELS_DIR, RESULTS_DIR, ROOT, load_config, write_json

sys.path.insert(0, str(KERNEL_SOURCE.parent))
from train_kernel import summarize_metrics  # noqa: E402

MINE_DIR = ROOT / "data" / "UIC_microscope_photos"
WORK_DIR = ROOT / "build" / "finetune"
RUNS_DIR = ROOT / "runs" / "finetune"


def link(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target, target_is_directory=True)


def build_dataset(repeat: int, mine: Path) -> tuple[Path, Path, Path]:
    """data.yaml files for training (TXL-PBC train + my train tiles, repeated; my val tiles as val), and for
    evaluating on my val tiles and on TXL-PBC's test split.

    Each repeat is a separate symlinked folder, so the repeated images have distinct paths, and Ultralytics' label
    caches land in build/ instead of the datasets.
    """
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    for sub in ("images", "labels"):
        link(DATA_DIR / sub / "train", WORK_DIR / "txl" / sub / "train")
        link(DATA_DIR / sub / "test", WORK_DIR / "txl" / sub / "test")
        link(mine / sub / "val", WORK_DIR / "mine_val" / sub / "val")
        for k in range(repeat):
            link(mine / sub / "train", WORK_DIR / f"mine_{k}" / sub / "train")
    names = dict(enumerate(CLASS_NAMES))
    train_yaml = WORK_DIR / "train.yaml"
    train_yaml.write_text(yaml.safe_dump({
        "path": str(WORK_DIR),
        "train": ["txl/images/train"] + [f"mine_{k}/images/train" for k in range(repeat)],
        "val": "mine_val/images/val",
        "names": names,
    }, sort_keys=False), encoding="utf-8")
    mine_yaml = WORK_DIR / "mine_val.yaml"
    mine_yaml.write_text(yaml.safe_dump({"path": str(WORK_DIR / "mine_val"), "train": "images/val",
                                         "val": "images/val", "names": names}, sort_keys=False), encoding="utf-8")
    txl_yaml = WORK_DIR / "txl_test.yaml"
    txl_yaml.write_text(yaml.safe_dump({"path": str(WORK_DIR / "txl"), "train": "images/train",
                                        "val": "images/test", "names": names}, sort_keys=False), encoding="utf-8")
    return train_yaml, mine_yaml, txl_yaml


def count_errors(model, images_dir: Path, conf: float, imgsz: int, device) -> dict:
    """Per-tile count errors (predicted minus labeled) for each class, at one confidence threshold."""
    paths = sorted(images_dir.glob("*.png"))
    diffs = {c: [] for c in CLASS_NAMES}
    truth = {c: 0 for c in CLASS_NAMES}
    for i in range(0, len(paths), 16):
        for path, r in zip(paths[i:i + 16], model.predict([str(p) for p in paths[i:i + 16]], conf=conf, imgsz=imgsz,
                                                          device=device, verbose=False, max_det=1000)):
            label = Path(str(path).replace("/images/", "/labels/")).with_suffix(".txt")
            labeled = [int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()]
            predicted = r.boxes.cls.int().tolist()
            for k, c in enumerate(CLASS_NAMES):
                diffs[c].append(predicted.count(k) - labeled.count(k))
                truth[c] += labeled.count(k)
    out = {"conf": round(conf, 4), "tiles": len(paths)}
    for c in CLASS_NAMES:
        d = np.array(diffs[c])
        out[c] = {"labeled": truth[c], "mae": round(float(np.abs(d).mean()), 3), "bias": round(float(d.mean()), 3),
                  "total_error_pct": round(100 * float(d.sum()) / max(truth[c], 1), 1)}
    return out


def evaluate(weights: Path, mine_yaml: Path, txl_yaml: Path, imgsz: int, batch: int, device) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs = dict(imgsz=imgsz, batch=batch, device=device, plots=False, verbose=False, split="val")
    mine = summarize_metrics(model.val(data=str(mine_yaml), **kwargs), with_curves=False)
    txl = summarize_metrics(model.val(data=str(txl_yaml), **kwargs), with_curves=False)
    counts = count_errors(model, WORK_DIR / "mine_val" / "images" / "val", mine.get("f1_optimal_conf", 0.25), imgsz,
                          device)
    return {"weights": str(weights.relative_to(ROOT)), "my_val": mine, "my_val_counts": counts, "txl_test": txl}


def table(base: dict, tuned: dict) -> str:
    lines = [f"{'':24s}{'base':>10s}{'fine-tuned':>12s}"]
    for key, label in (("my_val", "my val tiles"), ("txl_test", "TXL-PBC test")):
        for metric in ("mAP50", "mAP50-95"):
            lines.append(f"{label + ' ' + metric:24s}{base[key]['overall'][metric]:10.3f}"
                         f"{tuned[key]['overall'][metric]:12.3f}")
    for i, c in enumerate(CLASS_NAMES):
        b = base["my_val"]["per_class"][i] if i < len(base["my_val"]["per_class"]) else None
        t = tuned["my_val"]["per_class"][i] if i < len(tuned["my_val"]["per_class"]) else None
        if b and t:
            lines.append(f"{'my val ' + c + ' mAP50-95':24s}{b['mAP50-95']:10.3f}{t['mAP50-95']:12.3f}")
    for c in CLASS_NAMES:
        lines.append(f"{'my val ' + c + ' count MAE':24s}{base['my_val_counts'][c]['mae']:10.2f}"
                     f"{tuned['my_val_counts'][c]['mae']:12.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=MINE_DIR,
                        help="my labeled tiles: images/ and labels/ with train and val (default: data/UIC_microscope_photos)")
    parser.add_argument("--epochs", type=int, help="override finetune.train.epochs")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    cfg = load_config()
    ft = cfg["finetune"]
    train_args = cfg["train"] | ft["train"] | ({"epochs": args.epochs} if args.epochs else {})
    mine = args.data.resolve()
    if not (mine / "images" / "train").exists():
        raise SystemExit(f"No labeled tiles in {args.data}: run `make label-export` first")
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("No CUDA GPU found: training on the CPU will be very slow (see `make setup-gpu`).")
    train_yaml, mine_yaml, txl_yaml = build_dataset(ft["mine_repeat"], mine)
    base = ROOT / ft["base"]

    model = YOLO(str(base))
    model.train(data=str(train_yaml), project=str(RUNS_DIR), name=ft["run_name"], exist_ok=True, device=device,
                plots=True, **train_args)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    out = MODELS_DIR / f"{ft['run_name']}.pt"
    shutil.copy2(best, out)
    del model
    torch.cuda.empty_cache()

    imgsz, batch = train_args["imgsz"], train_args["batch"]
    print("Evaluating the base and the fine-tuned model…")
    result = {"base": evaluate(base, mine_yaml, txl_yaml, imgsz, batch, device),
              "fine_tuned": evaluate(out, mine_yaml, txl_yaml, imgsz, batch, device),
              "train": {k: v for k, v in train_args.items()}, "mine_repeat": ft["mine_repeat"],
              "run_dir": str(Path(RUNS_DIR / ft["run_name"]).relative_to(ROOT))}
    write_json(RESULTS_DIR / "finetune.json", result)
    print(table(result["base"], result["fine_tuned"]))
    print(f"Fine-tuned weights: {out.relative_to(ROOT)}; comparison: results/finetune.json")


if __name__ == "__main__":
    main()
