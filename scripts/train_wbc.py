"""Train and evaluate a six-class WBC subtype classifier on the local CUDA GPU.

    python scripts/train_wbc.py [--config config/wbc.yaml] [--resume] [--epochs N] [--run-name NAME]

Original images are unchanged. Exact duplicates are assigned to only one split (test, then val, then train);
all copies with conflicting labels are quarantined. Validation macro-F1 selects the checkpoint. Test predictions
are made only after training and model selection finish. No web app or detector weights are modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from common import MODELS_DIR, REPORTS_DIR, RESULTS_DIR, ROOT, RUNS_DIR, write_json
from wbc_classifier import MEAN, STD, image_transform, make_model


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CellImages(Dataset):
    def __init__(self, records: list[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return self.transform(Image.fromarray(row["pixels"])), row["label"]


def prepare_data(root: Path, classes: list[str], run: Path) -> tuple[dict, dict]:
    groups = defaultdict(list)
    original = {}
    for split in ("train", "val", "test"):
        original[split] = {}
        for label, name in enumerate(classes):
            paths = sorted((root / split / name).glob("*.png"))
            if not paths:
                raise ValueError(f"No images for {split}/{name}")
            original[split][name] = len(paths)
            for path in paths:
                with Image.open(path) as im:
                    if im.size != (64, 64) or im.mode != "RGB":
                        raise ValueError(f"Expected a 64x64 RGB image: {path} is {im.size}/{im.mode}")
                    pixels = np.asarray(im).copy()
                digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                groups[digest].append({"file": str(path.relative_to(root)), "split": split,
                                       "label": label, "class": name, "sha256_pixels": digest, "pixels": pixels})
    priority = {"test": 0, "val": 1, "train": 2}
    kept = {split: [] for split in priority}
    excluded = []
    cross_split = conflicts = duplicates = 0
    for rows in groups.values():
        duplicates += len(rows) > 1
        cross_split += len({r["split"] for r in rows}) > 1
        if len({r["label"] for r in rows}) > 1:
            conflicts += 1
            for row in rows:
                excluded.append({k: v for k, v in row.items() if k != "pixels"} | {"reason": "conflicting_labels"})
            continue
        rows.sort(key=lambda row: (priority[row["split"]], row["file"]))
        kept[rows[0]["split"]].append(rows[0])
        for row in rows[1:]:
            excluded.append({k: v for k, v in row.items() if k != "pixels"} | {"reason": "exact_duplicate"})
    manifest = {}
    for split, rows in kept.items():
        rows.sort(key=lambda row: row["file"])
        manifest[split] = [{k: v for k, v in row.items() if k != "pixels"} for row in rows]
    sets = [set(r["sha256_pixels"] for r in manifest[s]) for s in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    audit = {"root": str(root), "native_size": 64, "classes": classes, "original_counts": original,
             "clean_counts": {s: dict(Counter(row["class"] for row in rows)) for s, rows in kept.items()},
             "duplicate_groups": duplicates, "cross_split_duplicate_groups": cross_split,
             "conflicting_label_groups": conflicts, "excluded": excluded, "manifest_sha256": fingerprint,
             "policy": "Quarantine conflicting labels; keep one exact copy, preferring test over val over train.",
             "scope": "Exact pixel duplicates only; patient-level separation and near-duplicate overlap are unknown."}
    write_json(run / "dataset_audit.json", audit)
    write_json(run / "manifest.json", manifest)
    print("Data audit:", json.dumps({"clean_sizes": {s: len(r) for s, r in kept.items()},
                                      "excluded": len(excluded), "fingerprint": fingerprint}), flush=True)
    return kept, audit


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros_like(tp), where=matrix.sum(axis=0) > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall > 0)
    return {"accuracy": float(tp.sum() / max(matrix.sum(), 1)), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist(), "support": support.tolist(), "confusion_matrix": matrix.tolist()}


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(rows, transform, cfg, *, training=False, epoch=0):
    gen = torch.Generator().manual_seed(cfg["seed"] + epoch)
    return DataLoader(CellImages(rows, transform), batch_size=cfg["batch_size"], shuffle=training,
                      num_workers=cfg["workers"], pin_memory=True, worker_init_fn=seed_worker, generator=gen)


@torch.inference_mode()
def evaluate(model, batches, classes: int, amp: bool, *, predictions=False):
    model.eval()
    matrix = np.zeros((classes, classes), dtype=np.int64)
    loss_sum = count = 0
    outputs = []
    for images, labels in batches:
        images = images.to("cuda", non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        truth = labels.cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
        np.add.at(matrix, (truth, pred), 1)
        loss_sum += loss.item(); count += len(labels)
        if predictions:
            outputs.extend(logits.float().softmax(1).cpu().tolist())
    result = metrics_from_confusion(matrix)
    result["loss"] = loss_sum / count
    return result, outputs


def save_checkpoint(path: Path, state: dict):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=ROOT / "config/wbc.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None: cfg["train"]["epochs"] = args.epochs
    if args.run_name: cfg["run_name"] = args.run_name
    train = cfg["train"]
    if train["device"] != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This training run requires a working local CUDA GPU.")
    if train["epochs"] < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_all(train["seed"])
    run = RUNS_DIR / "wbc" / cfg["run_name"]
    if (run / "last.pt").exists() and not args.resume:
        raise SystemExit(f"Run already exists: use --resume or --run-name. {run}")
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", cfg)
    data, audit = prepare_data((ROOT / cfg["data"]).resolve(), cfg["classes"], run)
    size = cfg["input_size"]
    # The cell has no canonical orientation. Mild color changes model different illumination/staining.
    transform = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(180, interpolation=T.InterpolationMode.BILINEAR, fill=(245, 245, 245)),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        T.ToTensor(), T.Normalize(MEAN, STD)])
    val_transform = image_transform(size)
    resume = torch.load(run / "last.pt", map_location="cpu", weights_only=True) if args.resume else None
    if resume and resume["manifest_sha256"] != audit["manifest_sha256"]:
        raise ValueError("Dataset changed since the saved training checkpoint")
    model = make_model(len(cfg["classes"]), pretrained=resume is None).to("cuda", memory_format=torch.channels_last)
    backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": train["backbone_lr"]},
                                  {"params": model.fc.parameters(), "lr": train["head_lr"]}],
                                 weight_decay=train["weight_decay"])
    def lr_factor(epoch):
        if epoch < train["warmup_epochs"]: return (epoch + 1) / train["warmup_epochs"]
        progress = (epoch - train["warmup_epochs"]) / max(1, train["epochs"] - train["warmup_epochs"])
        return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=train["amp"])
    counts = Counter(row["label"] for row in data["train"])
    weights = torch.tensor([len(data["train"]) / (len(counts) * counts[i]) for i in range(len(counts))], device="cuda")
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=train["label_smoothing"])
    history = []; best = -1.0; stale = 0; start_epoch = 1; best_epoch = 0
    if resume:
        model.load_state_dict(resume["state_dict"]); optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"]); scaler.load_state_dict(resume["scaler"])
        history = resume["history"]; best = resume["best_f1"]; stale = resume["stale"]
        start_epoch = resume["epoch"] + 1; best_epoch = resume["best_epoch"]
    environment = {"torch": str(torch.__version__), "torchvision": str(__import__("torchvision").__version__),
                   "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                   "device": "cuda:0", "amp": train["amp"], "cudnn_deterministic": True}
    metadata = {"architecture": cfg["architecture"], "classes": cfg["classes"], "input_size": size,
                "mean": list(MEAN), "std": list(STD), "manifest_sha256": audit["manifest_sha256"],
                "environment": environment, "config": cfg}
    print("Training:", json.dumps(environment), flush=True)
    print(f"{len(data['train'])} training crops; {len(data['val'])} validation crops. Test is reserved until selection finishes.", flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, train["epochs"] + 1):
        seed_all(train["seed"] + epoch)
        model.train(); loss_sum = total = correct = 0; tick = time.perf_counter()
        batches = loader(data["train"], transform, train, training=True, epoch=epoch)
        for batch, (images, labels) in enumerate(batches, start=1):
            images = images.to("cuda", non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=train["amp"]):
                logits = model(images); loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * len(labels); total += len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            if batch % 25 == 0:
                write_json(run / "progress.json", {"stage": "train", "epoch": epoch, "epochs": train["epochs"],
                    "batch": batch, "batches": len(batches), "train_loss": loss_sum / total,
                    "gpu": environment["gpu"], "updated_at": time.time()})
                print(f"Epoch {epoch:02d}/{train['epochs']} batch {batch}/{len(batches)} loss {loss_sum / total:.4f}", flush=True)
        val, _ = evaluate(model, loader(data["val"], val_transform, train), len(cfg["classes"]), train["amp"])
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total,
               "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
               "seconds": time.perf_counter() - tick, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if val["macro_f1"] > best + 1e-5:
            best = val["macro_f1"]; stale = 0; best_epoch = epoch
            save_checkpoint(run / "best.pt", {**metadata, "state_dict": model.state_dict(), "epoch": epoch,
                                               "validation": val})
        else: stale += 1
        scheduler.step()
        save_checkpoint(run / "last.pt", {**metadata, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
            "best_f1": best, "stale": stale, "history": history})
        write_json(run / "history.json", history)
        write_json(run / "progress.json", {"stage": "epoch_done", **row, "best_epoch": best_epoch,
                                            "best_val_macro_f1": best, "updated_at": time.time()})
        print(f"Epoch {epoch:02d}/{train['epochs']} train {correct / total:.4f} val acc {val['accuracy']:.4f} "
              f"macro-F1 {val['macro_f1']:.4f} best {best:.4f} ({row['seconds']:.1f}s)", flush=True)
        if stale >= train["patience"]:
            print(f"Early stopping after {stale} epochs without validation improvement.", flush=True)
            break
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    # Final validation and test use the deliverable's FP32 inference path, without test augmentation.
    val, _ = evaluate(model, loader(data["val"], val_transform, train), len(cfg["classes"]), False)
    test, probabilities = evaluate(model, loader(data["test"], val_transform, train), len(cfg["classes"]), False, predictions=True)
    MODELS_DIR.mkdir(exist_ok=True)
    deliverable = MODELS_DIR / f"{cfg['run_name']}.pt"
    save_checkpoint(deliverable, {**metadata, "state_dict": model.state_dict(), "epoch": best_epoch, "validation": val})
    result = {"run_name": cfg["run_name"], "weights": str(deliverable.relative_to(ROOT)),
              "weights_sha256": hashlib.sha256(deliverable.read_bytes()).hexdigest(), "config": cfg,
              "environment": environment, "best_epoch": best_epoch, "epochs_completed": len(history),
              "duration_seconds": time.perf_counter() - started, "dataset_audit": audit,
              "validation": val, "test": test, "history": history,
              "limits": ["Six-class classifier for cropped WBCs, not a whole-image detector.",
                         "Native input data is 64x64; resizing to 128x128 does not restore detail.",
                         "Exact duplicate overlap removed; patient-level split and near-duplicate overlap are unknown.",
                         "Accuracy on personal microscope WBC crops has not been measured.",
                         "Immature granulocytes form one dataset class; their individual maturation stages are not separated.",
                         "Softmax scores are not calibrated clinical probabilities; non-WBC or unfamiliar cells are forced into one of six classes."]}
    write_json(RESULTS_DIR / "wbc_classification.json", result)
    write_json(run / "test_predictions.json", [
        {"file": row["file"], "true_class": row["class"], "probabilities": p}
        for row, p in zip(data["test"], probabilities)])
    with (run / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    from report_wbc import build_report

    report = build_report()
    print(f"Evaluation report: {report.relative_to(ROOT)}", flush=True)
    write_json(run / "progress.json", {"stage": "complete", "best_epoch": best_epoch,
        "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"], "weights": str(deliverable),
        "updated_at": time.time()})
    print(f"Complete: {deliverable.relative_to(ROOT)}; test accuracy {test['accuracy']:.4%}, "
          f"macro-F1 {test['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
