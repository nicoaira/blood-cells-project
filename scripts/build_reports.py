"""Training and evaluation reports for a finished Kaggle run.

Reads the outputs that scripts/monitor.py downloads to runs/kaggle/v<N>/ and writes
    reports/training_report.md, reports/evaluation_report.md and their figures
    dashboard/data/training.js (final run state) and dashboard/data/evaluation.js
    results/ (headline metrics, per-epoch CSV, evaluation details) and models/<run_name>.pt

    python scripts/build_reports.py [--outputs runs/kaggle/v1] [--dest DIR]

--dest writes everything under another root directory (used to test the builder
on smoke-test outputs without touching the project's reports).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import plotting as P
from common import CLASS_NAMES, DATA_DIR, KERNEL_SOURCE, ROOT, RUNS_DIR, read_current_run, read_dashboard_data, \
    read_yolo_labels, relpath, write_dashboard_data, write_json
from run_state import RunState

sys.path.insert(0, str(KERNEL_SOURCE.parent))
from train_kernel import PREDICT_MARGIN  # noqa: E402

NC = len(CLASS_NAMES)
BG = NC  # "background" row/column of the confusion matrix
LOSS_TITLES = {"box_loss": "Box regression loss", "cls_loss": "Classification loss", "l1_loss": "L1 box loss",
               "dfl_loss": "Distribution focal loss"}
EXAMPLE_IMAGES = 6
SEED = 0


class Dest:
    """Destination layout, rooted at the project unless --dest is given."""

    def __init__(self, root: Path):
        self.root = root
        self.reports = root / "reports"
        self.figures = self.reports / "figures"
        self.dashboard = root / "dashboard"
        self.dashboard_data = self.dashboard / "data"
        self.results = root / "results"
        self.models = root / "models"


# --------------------------------------------------------------------------- loading


def load_outputs(out_dir: Path) -> dict:
    art = out_dir / "artifacts"
    summary = json.loads((art / "summary.json").read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "evaluation": json.loads((art / "evaluation.json").read_text(encoding="utf-8")),
        "predictions": json.loads((art / "predictions.json").read_text(encoding="utf-8")),
        "events": [json.loads(line) for line in (art / "events.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()],
        "results_csv": out_dir / "runs" / "train" / "results.csv",
        "weights": out_dir / summary["weights"],
    }


def console_from_log(out_dir: Path) -> list[str]:
    """Console lines from the Kaggle log that `kaggle kernels output` saves next to the outputs."""
    logs = sorted(out_dir.glob("*.log"))
    if not logs:
        return []
    text = logs[0].read_text(encoding="utf-8", errors="replace")
    parser = RunState({})
    try:
        for event in json.loads(text):
            parser.feed(event.get("stream_name") or "stdout", event.get("data") or "")
    except json.JSONDecodeError:
        parser.feed("stdout", text)
    parser.flush()
    return parser.console


def final_training_state(run: dict, events: list[dict], console: list[str]) -> RunState:
    state = RunState(run)
    for event in events:
        state.apply_event(event)
    state.set_status("complete")
    state.set_phase("finished")
    state.data["console"] = console[-250:]
    return state


# --------------------------------------------------------------------------- matching


def load_ground_truth(split: str, names: list[str]) -> dict[str, np.ndarray]:
    """Boxes per image as rows (class, x1, y1, x2, y2), normalized. Repeated rows are dropped, as in Ultralytics.

    Coordinates are rounded to 4 decimals, the precision stored in the dashboard data, so the dashboard's
    in-browser matching reproduces these numbers exactly.
    """
    gt = {}
    for name in names:
        rows = sorted(set(read_yolo_labels(DATA_DIR / "labels" / split / f"{Path(name).stem}.txt")))
        boxes = np.array([(c, x - w / 2, y - h / 2, x + w / 2, y + h / 2) for c, x, y, w, h in rows],
                         dtype=float).reshape(-1, 5)
        boxes[:, 1:] = boxes[:, 1:].round(4)
        gt[name] = boxes
    return gt


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (n, 4) and (m, 4) xyxy boxes. Normalized coordinates give the same IoU as pixels."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def greedy_match(pred: np.ndarray, gt: np.ndarray, iou_thr: float, class_aware: bool):
    """Match predictions, highest confidence first, to the best unclaimed ground-truth box.

    pred rows: (class, conf, x1, y1, x2, y2); gt rows: (class, x1, y1, x2, y2).
    Returns (matched (pred, gt) index pairs, unmatched preds, unmatched gt).
    """
    if not len(pred) or not len(gt):
        return [], list(range(len(pred))), list(range(len(gt)))
    ious = box_iou(pred[:, 2:6], gt[:, 1:5])
    if class_aware:
        ious = np.where(pred[:, None, 0] == gt[None, :, 0], ious, 0.0)
    claimed = np.zeros(len(gt), dtype=bool)
    pairs, unmatched = [], []
    for i in np.argsort(-pred[:, 1], kind="stable"):
        candidates = np.where(claimed, -1.0, ious[i])
        j = int(candidates.argmax())
        if candidates[j] >= iou_thr:
            claimed[j] = True
            pairs.append((int(i), j))
        else:
            unmatched.append(int(i))
    return pairs, unmatched, [int(j) for j in np.flatnonzero(~claimed)]


def operating_point(records: list[dict], gt: dict[str, np.ndarray], conf: float, iou_thr: float) -> dict:
    """Detection, confusion and counting statistics at a single confidence threshold."""
    tp, fp, fn = np.zeros(NC, int), np.zeros(NC, int), np.zeros(NC, int)
    cm = np.zeros((NC + 1, NC + 1), int)  # rows: true class, columns: predicted class
    images = []
    for rec in records:
        p = np.array(rec["boxes"], dtype=float).reshape(-1, 6)
        p = p[p[:, 1] >= conf]
        g = gt[rec["image"]]
        pairs, fps, fns = greedy_match(p, g, iou_thr, class_aware=True)
        for _, j in pairs:
            tp[int(g[j, 0])] += 1
        for i in fps:
            fp[int(p[i, 0])] += 1
        for j in fns:
            fn[int(g[j, 0])] += 1
        img_errors = len(fps) + len(fns)
        pairs, fps, fns = greedy_match(p, g, iou_thr, class_aware=False)
        for i, j in pairs:
            cm[int(g[j, 0]), int(p[i, 0])] += 1
        for i in fps:
            cm[BG, int(p[i, 0])] += 1
        for j in fns:
            cm[int(g[j, 0]), BG] += 1
        images.append({"image": rec["image"], "errors": img_errors,
                       "gt": np.bincount(g[:, 0].astype(int), minlength=NC).tolist(),
                       "pred": np.bincount(p[:, 0].astype(int), minlength=NC).tolist()})

    per_class, counting = [], []
    for c, name in enumerate(CLASS_NAMES):
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        per_class.append({"class": name, "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c]),
                          "precision": prec, "recall": rec,
                          "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})
        g_n = np.array([im["gt"][c] for im in images])
        p_n = np.array([im["pred"][c] for im in images])
        err = p_n - g_n
        present = g_n > 0
        counting.append({
            "class": name,
            "true_total": int(g_n.sum()), "pred_total": int(p_n.sum()),
            "mae": float(np.abs(err).mean()), "bias": float(err.mean()),
            "exact": float((err == 0).mean()),
            "within_1": float((np.abs(err) <= 1).mean()),
            "mape": float((np.abs(err[present]) / g_n[present]).mean()) if present.any() else None,
            "pearson_r": float(np.corrcoef(g_n, p_n)[0, 1]) if g_n.std() and p_n.std() else None,
        })
    return {"conf": conf, "iou": iou_thr, "per_class": per_class, "counting": counting,
            "confusion": {"labels": [*CLASS_NAMES, "background"], "matrix": cm.tolist()}, "images": images}


# --------------------------------------------------------------------------- figures


def _best_marker(ax, best_epoch: int | None) -> None:
    """Vertical hairline at the best epoch; also keeps the epoch axis on whole numbers."""
    ax.xaxis.set_major_locator(P.matplotlib.ticker.MaxNLocator(integer=True))
    if best_epoch:
        ax.axvline(best_epoch, color=P.AXIS, lw=1, zorder=0)


def _label_room(ax, last_epoch: int, share: float) -> None:
    """Widen the epoch axis by `share` for end labels, keeping ticks within the epochs that ran."""
    ax.set_xlim(right=last_epoch * (1 + share) + 1)
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= last_epoch])


def fig_training_losses(epochs: list[dict], best_epoch: int, mosaic_off: int | None, dest: Dest) -> str:
    keys = list(epochs[0]["train_loss"])
    x = [e["epoch"] for e in epochs]
    fig, axes = P.plt.subplots(1, len(keys), figsize=(10, 2.8))
    labels = []
    for ax, k in zip(np.atleast_1d(axes), keys):
        train = [e["train_loss"].get(k) for e in epochs]
        val = [e["val_loss"].get(k) for e in epochs]
        _best_marker(ax, best_epoch)
        if mosaic_off:
            ax.axvspan(mosaic_off - 0.5, x[-1] + 0.5, color=P.GRID, alpha=0.35, lw=0, zorder=0)
        ax.plot(x, train, color=P.MUTED, label="train")
        ax.plot(x, val, color=P.ACCENT, label="val")
        labels.append((ax, [(x[-1], train[-1], "train"), (x[-1], val[-1], "val")]))
        ax.set_title(LOSS_TITLES.get(k, k))
        ax.set_xlabel("epoch")
        ax.margins(x=0.02)
        _label_room(ax, x[-1], 0.14)
    np.atleast_1d(axes)[0].legend(loc="upper right")
    fig.tight_layout()
    for ax, points in labels:
        P.end_labels(ax, points)
    return P.save(fig, "training_losses", dest.figures)


def fig_training_metrics(epochs: list[dict], best_epoch: int, dest: Dest) -> str:
    keys = ["precision", "recall", "mAP50", "mAP50-95"]
    x = [e["epoch"] for e in epochs]
    fig, axes = P.plt.subplots(1, 4, figsize=(10, 2.6))
    for ax, k in zip(axes, keys):
        y = [e["metrics"].get(k) for e in epochs]
        _best_marker(ax, best_epoch)
        ax.plot(x, y, color=P.ACCENT)
        b = next(e for e in epochs if e["epoch"] == best_epoch) if best_epoch else epochs[-1]
        ax.plot([b["epoch"]], [b["metrics"][k]], "o", ms=6, color=P.ACCENT, mec=P.SURFACE, mew=2)
        ax.annotate(f"{b['metrics'][k]:.3f}", (b["epoch"], b["metrics"][k]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=9, color=P.INK_2)
        ax.set_title(f"Val {k}")
        ax.set_xlabel("epoch")
        lo = min(v for v in y if v is not None)
        ax.set_ylim(max(0.0, lo - 0.05), min(1.0, max(y) + 0.03))
    fig.tight_layout()
    return P.save(fig, "training_metrics", dest.figures)


def fig_training_per_class(epochs: list[dict], best_epoch: int, dest: Dest) -> str:
    x = [e["epoch"] for e in epochs]
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 3.0))
    labels = []
    for ax, k in zip(axes, ("mAP50", "mAP50-95")):
        _best_marker(ax, best_epoch)
        series = {c: [e["per_class"].get(c, {}).get(k) for e in epochs] for c in CLASS_NAMES}
        for c, y in series.items():
            ax.plot(x, y, color=P.CLASS_COLORS[c], label=c)
        overall = [e["metrics"].get(k) for e in epochs]
        ax.plot(x, overall, color=P.INK, lw=1.2, label="all classes")
        labels.append((ax, [(x[-1], y[-1], f"{c} {y[-1]:.2f}") for c, y in series.items()]))
        ax.set_title(f"Validation {k} per class")
        ax.set_xlabel("epoch")
        _label_room(ax, x[-1], 0.22)
        low = min(v for y in series.values() for v in y if v is not None)
        ax.set_ylim(max(0.0, low - 0.05), 1.0)
    axes[0].legend(loc="lower right", ncol=2)
    fig.tight_layout()
    for ax, points in labels:
        P.end_labels(ax, points)
    return P.save(fig, "training_per_class", dest.figures)


def fig_training_schedule(epochs: list[dict], dest: Dest) -> str:
    x = [e["epoch"] for e in epochs]
    fig, (ax1, ax2) = P.plt.subplots(1, 2, figsize=(10, 2.5))
    ax1.plot(x, [e["lr"].get("pg0") for e in epochs], color=P.ACCENT)
    ax1.set_title("Learning rate (weight parameters)")
    ax1.set_xlabel("epoch")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    times = [e["epoch_time"] for e in epochs]
    ax2.bar(x, times, width=0.7, color=P.ACCENT)
    ax2.axhline(float(np.median(times)), color=P.INK_2, lw=1)
    ax2.annotate(f"median {np.median(times):.1f} s", (x[-1], float(np.median(times))), xytext=(0, 4),
                 textcoords="offset points", ha="right", fontsize=9, color=P.INK_2)
    ax2.set_title("Epoch duration (train + validation)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("seconds")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return P.save(fig, "training_schedule", dest.figures)


def fig_eval_per_class(split_eval: dict, dest: Dest) -> str:
    keys = ["precision", "recall", "mAP50", "mAP50-95"]
    rows = {r["class"]: r for r in split_eval["per_class"]}
    fig, axes = P.plt.subplots(1, 4, figsize=(10, 2.4), sharey=True)
    y = np.arange(NC)[::-1]
    for ax, k in zip(axes, keys):
        vals = [rows[c][k] for c in CLASS_NAMES]
        ax.barh(y, vals, height=0.5, color=[P.CLASS_COLORS[c] for c in CLASS_NAMES])
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:.3f}", (v, yi), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9, color=P.INK_2)
        overall = split_eval["overall"][k]
        ax.axvline(overall, color=P.INK, lw=1)
        ax.annotate(f"all {overall:.3f}", (overall, y[0] + 0.55), xytext=(-3, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=P.INK_2)
        ax.set_ylim(-0.5, y[0] + 0.85)  # headroom above the top bar for the "all" label
        ax.set_xlim(0, 1.28)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(k)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, CLASS_NAMES)
    fig.tight_layout()
    return P.save(fig, "eval_per_class", dest.figures)


def fig_eval_curves(split_eval: dict, conf: float, dest: Dest) -> str:
    curves = split_eval["curves"]
    x = np.array(curves["x"])
    ap = {r["class"]: r["mAP50"] for r in split_eval["per_class"]}
    fig, (ax1, ax2) = P.plt.subplots(1, 2, figsize=(10, 3.3))
    for c in CLASS_NAMES:
        ax1.plot(x, curves["precision_recall"][c], color=P.CLASS_COLORS[c], label=f"{c} (AP50 {ap[c]:.3f})")
        ax2.plot(x, curves["f1_confidence"][c], color=P.CLASS_COLORS[c], label=c)
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Precision-recall (IoU 0.5)")
    ax1.legend(loc="lower left")
    ax2.axvline(conf, color=P.INK_2, lw=1)
    ax2.annotate(f"operating threshold {conf:.2f}", (conf, 0.05), xytext=(-5, 0), textcoords="offset points",
                 ha="right", fontsize=9, color=P.INK_2)
    ax2.set_xlabel("confidence threshold")
    ax2.set_ylabel("F1")
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("F1 vs confidence")
    ax2.legend(loc="center left")
    fig.tight_layout()
    return P.save(fig, "eval_curves", dest.figures)


def fig_eval_confusion(op: dict, dest: Dest) -> str:
    cm = np.array(op["confusion"]["matrix"], dtype=float)
    labels = op["confusion"]["labels"]
    shares = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cmap = P.matplotlib.colors.LinearSegmentedColormap.from_list("seq", [P.SURFACE] + P.SEQ_BLUE[1:])
    fig, ax = P.plt.subplots(figsize=(5.4, 4.4))
    ax.imshow(np.where(np.eye(NC + 1, dtype=bool) & (np.arange(NC + 1) == BG), np.nan, shares),
              cmap=cmap, vmin=0, vmax=1)
    for i in range(NC + 1):
        for j in range(NC + 1):
            if i == BG and j == BG:
                continue
            light = shares[i, j] > 0.55
            ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=10,
                    color="white" if light else P.INK, fontweight="semibold" if i == j else "normal")
    ax.set_xticks(range(NC + 1), [*CLASS_NAMES, "missed"])
    ax.set_yticks(range(NC + 1), [*CLASS_NAMES, "no cell"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.grid(False)
    # Surface-colored gaps between cells instead of borders.
    ax.set_xticks(np.arange(-0.5, NC + 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NC + 1), minor=True)
    ax.grid(which="minor", axis="both", color=P.SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Confusion matrix (conf ≥ {op['conf']:.2f}, IoU ≥ {op['iou']})")
    fig.tight_layout()
    return P.save(fig, "eval_confusion", dest.figures)


def fig_eval_counting(op: dict, dest: Dest) -> str:
    fig, axes = P.plt.subplots(1, NC, figsize=(10, 3.2))
    stats = {c["class"]: c for c in op["counting"]}
    for k, (ax, c) in enumerate(zip(axes, CLASS_NAMES)):
        pts = {}
        for im in op["images"]:
            key = (im["gt"][k], im["pred"][k])
            pts[key] = pts.get(key, 0) + 1
        gx, py = np.array(list(pts)).T
        n = np.array(list(pts.values()))
        hi = max(gx.max(), py.max()) + 1
        ax.plot([0, hi], [0, hi], color=P.AXIS, lw=1, zorder=0)
        ax.scatter(gx, py, s=18 + 14 * np.sqrt(n), color=P.CLASS_COLORS[c], edgecolor=P.SURFACE, linewidth=1.5,
                   zorder=2)
        ax.set_xlim(-0.6, hi)
        ax.set_ylim(-0.6, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("true count per image")
        ax.set_title(f"{c}: MAE {stats[c]['mae']:.2f}, exact {stats[c]['exact']:.0%}")
        if hi <= 8:
            ax.set_xticks(range(int(hi)))
            ax.set_yticks(range(int(hi)))
    axes[0].set_ylabel("predicted count per image")
    fig.tight_layout()
    return P.save(fig, "eval_counting", dest.figures)


def draw_errors(path: Path, pred: np.ndarray, gt: np.ndarray, conf: float, iou_thr: float, size: int = 360):
    with Image.open(path) as im:
        img = im.convert("RGB")
    s = size / max(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    W, H = img.size
    p = pred[pred[:, 1] >= conf]
    pairs, fps, fns = greedy_match(p, gt, iou_thr, class_aware=True)

    def rect(b, color, width=2):
        draw.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=color, width=width)

    for i, _ in pairs:
        rect(p[i, 2:6], P.STATUS["good"])
    for j in fns:
        rect(gt[j, 1:5], P.STATUS["warning"], 3)
    for i in fps:
        rect(p[i, 2:6], P.STATUS["critical"], 3)
    return img, len(pairs), len(fps), len(fns)


def fig_eval_examples(records: list[dict], gt: dict, op: dict, split: str, dest: Dest) -> str:
    errors = {im["image"]: im["errors"] for im in op["images"]}
    worst = sorted(records, key=lambda r: -errors[r["image"]])[: EXAMPLE_IMAGES // 2]
    rest = [r for r in records if r not in worst]
    picks = worst + random.Random(SEED).sample(rest, EXAMPLE_IMAGES - len(worst))
    fig, axes = P.plt.subplots(2, EXAMPLE_IMAGES // 2, figsize=(10, 7.0))
    for ax, rec in zip(axes.flat, picks):
        pred = np.array(rec["boxes"], dtype=float).reshape(-1, 6)
        img, n_tp, n_fp, n_fn = draw_errors(DATA_DIR / "images" / split / rec["image"], pred, gt[rec["image"]],
                                            op["conf"], op["iou"])
        ax.imshow(img)
        ax.set_title(f"{n_tp} correct · {n_fp} false · {n_fn} missed", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [P.matplotlib.patches.Patch(facecolor="none", edgecolor=P.STATUS[k], lw=2, label=lab)
               for k, lab in (("good", "✓ correct (TP)"), ("critical", "✗ false detection (FP)"),
                              ("warning", "! missed cell (FN)"))]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.04, hspace=0.12, bottom=0.06, top=0.95)
    return P.save(fig, "eval_examples", dest.figures)


# --------------------------------------------------------------------------- reports


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {s:02d} s" if m else f"{s} s")


def _iso(t: float | None) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if t else "–"


def stage_timeline(state: dict, run: dict) -> list[tuple[str, str, str]]:
    stages = state["stages"]
    end = state.get("finished_at")
    rows = []
    pushed = run.get("pushed_at")
    if pushed and state.get("started_at"):
        queued = state["started_at"] - datetime.fromisoformat(pushed).timestamp()
        rows.append(("queued on Kaggle", pushed.replace("T", " ").replace("+00:00", " UTC"), _fmt_duration(queued)))
    for i, s in enumerate(stages):
        nxt = stages[i + 1]["t"] if i + 1 < len(stages) else end
        rows.append((s["stage"], _iso(s["t"]), _fmt_duration(nxt - s["t"] if nxt else None)))
    return rows


def write_training_report(state: dict, run: dict, summary: dict, figures: dict, dest: Dest) -> Path:
    cfg, env, train = state["config"], state["env"] or {}, state["train"] or {}
    epochs, end = state["epochs"], state["train_end"] or {}
    best_epoch = end.get("best_epoch") or max(epochs, key=lambda e: e["fitness"])["epoch"]
    best = next(e for e in epochs if e["epoch"] == best_epoch)
    last = epochs[-1]
    tcfg = cfg["train"]
    val_box = [e["val_loss"].get("box_loss") for e in epochs]
    min_val_epoch = epochs[int(np.argmin(val_box))]["epoch"]
    rise = (val_box[-1] - min(val_box)) / min(val_box)
    if min_val_epoch == last["epoch"]:
        overfit_text = "Validation box loss was lowest in the last epoch, so there is no sign of overfitting."
    elif rise < 0.05:
        overfit_text = (f"Validation box loss reached its minimum at epoch {min_val_epoch} and ended within "
                        f"{rise:.1%} of it, so there is no sign of overfitting.")
    else:
        overfit_text = (f"Validation box loss bottomed out at epoch {min_val_epoch} and ended {rise:.0%} higher, a "
                        "mild sign of overfitting late in training. The checkpoint is chosen by validation "
                        "mAP50-95, so it is unaffected.")
    if np.mean(val_box[-5:]) < np.mean([e["train_loss"].get("box_loss") for e in epochs[-5:]]):
        overfit_text += (" Validation loss sits below training loss because training images are augmented "
                         "(mosaic, flips, color jitter) and validation images are not.")
    stop_text = (f"early stopping ended training after epoch {end['epochs_completed']}, "
                 f"{tcfg['patience']} epochs without improvement"
                 if end.get("early_stopped") else f"all {end.get('epochs_planned', len(epochs))} planned epochs ran")
    pcs = ", ".join(f"{c} {best['per_class'][c]['mAP50-95']:.3f}" for c in CLASS_NAMES if c in best["per_class"])
    lines = [
        f"# Training report: {cfg['model'].removesuffix('.pt').upper()} on TXL-PBC",
        "",
        f"_Kaggle kernel [`{run['kernel']}`]({run['url']}) version {run['version']} · generated "
        f"{datetime.now(timezone.utc):%Y-%m-%d}. Live and interactive version: the **Training** tab of the "
        "dashboard._",
        "",
        "## Summary",
        "",
        f"- Fine-tuned **{cfg['model']}** (COCO-pretrained) on a **{env.get('gpu', 'GPU')}** in "
        f"**{_fmt_duration(end.get('train_time_s'))}**, training all {train.get('params', 0) / 1e6:.1f} M parameters "
        f"of the {NC}-class model ({train.get('gflops', 0):.0f} GFLOPs at {tcfg['imgsz']} px); {stop_text}.",
        f"- **Best epoch {best_epoch}:** validation mAP50-95 **{best['metrics']['mAP50-95']:.3f}**, mAP50 "
        f"**{best['metrics']['mAP50']:.3f}**, precision {best['metrics']['precision']:.3f}, "
        f"recall {best['metrics']['recall']:.3f}. These weights are the deliverable (`best.pt`).",
        f"- Per-class mAP50-95 at the best epoch: {pcs}.",
        f"- {overfit_text}",
        "",
        "## Setup",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Model | `{cfg['model']}` (YOLO26, NMS-free capable head, no DFL), COCO-pretrained |",
        f"| Data | TXL-PBC: {train.get('train_images')} train / {train.get('val_images')} val images |",
        f"| Image size · batch | {tcfg['imgsz']} px · {tcfg['batch']} ({train.get('batches_per_epoch')} "
        "iterations per epoch) |",
        f"| Optimizer | {train.get('optimizer', tcfg['optimizer'])}, lr0 {tcfg['lr0']}, linear decay to "
        f"lr0×{tcfg['lrf']}, {tcfg['warmup_epochs']} warm-up epochs, weight decay {tcfg['weight_decay']} |",
        f"| Epochs | {tcfg['epochs']} planned, early-stopping patience {tcfg['patience']} |",
        f"| Augmentation | mosaic {tcfg['mosaic']} (off for the last {tcfg['close_mosaic']} epochs), "
        f"flips LR {tcfg['fliplr']} / UD {tcfg['flipud']}, scale ±{tcfg['scale']}, translate ±{tcfg['translate']}, "
        f"HSV {tcfg['hsv_h']}/{tcfg['hsv_s']}/{tcfg['hsv_v']}, no rotation or mixup |",
        f"| Precision | {'mixed precision (AMP)' if train.get('amp') else 'FP32'}, seed {tcfg['seed']}, "
        "deterministic |",
        f"| Hardware | Kaggle {env.get('gpu', '?')} ({env.get('gpu_mem_gb', '?')} GB), {env.get('cpu_count')} vCPUs; "
        "one GPU of the T4 ×2 session is used |",
        f"| Software | Python {env.get('python')}, PyTorch {env.get('torch')}, CUDA {env.get('cuda')}, "
        f"Ultralytics {env.get('ultralytics')} |",
        "",
        "## Losses",
        "",
        "YOLO26 trains two detection heads together: one-to-many, used with NMS, and one-to-one, NMS-free. The "
        "optimized loss is a weighted sum of both, and the weight moves from the one-to-many head (0.8 in the first "
        "epoch) to the one-to-one head (0.9 by the last planned epoch). Ultralytics logs the one-to-one head's "
        "losses, which are the curves below. The gray band marks the final epochs without mosaic augmentation, and "
        "the vertical line marks the best epoch.",
        "",
        f"![Losses]({figures['losses']})",
        "",
        "## Validation metrics",
        "",
        f"![Validation metrics]({figures['metrics']})",
        "",
        "| Epoch | Precision | Recall | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    marks = sorted({1, max(1, len(epochs) // 4), max(1, len(epochs) // 2), best_epoch, last["epoch"]})
    for ep in marks:
        e = next(x for x in epochs if x["epoch"] == ep)
        m = e["metrics"]
        tag = " (best)" if ep == best_epoch else (" (last)" if ep == last["epoch"] else "")
        lines.append(f"| {ep}{tag} | {m['precision']:.3f} | {m['recall']:.3f} | {m['mAP50']:.3f} | "
                     f"{m['mAP50-95']:.3f} |")
    weakest = min(CLASS_NAMES, key=lambda c: best["per_class"].get(c, {}).get("mAP50-95", 1))
    lines += [
        "",
        "## Per-class progress",
        "",
        f"{weakest} is the hardest class at the best epoch "
        f"(mAP50-95 {best['per_class'][weakest]['mAP50-95']:.3f}). The dataset report explains why: "
        "platelets are the rarest and smallest cells.",
        "",
        f"![Per-class metrics]({figures['per_class']})",
        "",
        "## Learning rate and epoch time",
        "",
        f"![Schedule]({figures['schedule']})",
        "",
        "## Run timeline",
        "",
        "| Stage | Started | Duration |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in stage_timeline(state, run)]
    weights = dest.models / f"{summary['run_name']}.pt"
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Weights: [`{relpath(weights, dest.root)}`](../{relpath(weights, dest.root)}) "
        f"({summary['weights_mb']} MB)",
        "- Per-epoch metrics: [`results/results.csv`](../results/results.csv); progress events: "
        "[`results/events.jsonl`](../results/events.jsonl)",
        "- Training configuration: [`config/train.yaml`](../config/train.yaml)",
        "",
    ]
    path = dest.reports / "training_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_evaluation_report(ev: dict, ops: dict, summary: dict, run: dict, figures: dict, dest: Dest) -> Path:
    test, val, e2e = ev["splits"]["test"], ev["splits"]["val"], ev["splits"].get("test_nms_free")
    op = ops["test"]
    o = test["overall"]
    n_cells = sum(r["instances"] for r in test["per_class"])
    n_images = len(op["images"])
    f1 = {r["class"]: r for r in op["per_class"]}
    cnt = {r["class"]: r for r in op["counting"]}
    speed = test["speed_ms"]
    cm = np.array(op["confusion"]["matrix"])
    cross = int(cm[:NC, :NC].sum() - np.trace(cm[:NC, :NC]))
    missed, extra = int(cm[:NC, BG].sum()), int(cm[BG, :NC].sum())
    dataset = read_dashboard_data("dataset") or {}
    n_dups = len(dataset.get("near_duplicates", []))
    raw_cells = dataset.get("per_split", {}).get("test", {}).get("cells_total", n_cells)
    dedup_note = f" after dropping {raw_cells - n_cells} repeated label rows" if raw_cells > n_cells else ""
    lines = [
        f"# Evaluation report: {summary['config']['model'].removesuffix('.pt').upper()} on TXL-PBC",
        "",
        f"_Kaggle kernel [`{run['kernel']}`]({run['url']}) version {run['version']} · generated "
        f"{datetime.now(timezone.utc):%Y-%m-%d}. Interactive version with a prediction explorer: the "
        "**Performance** tab of the dashboard._",
        "",
        "All numbers come from the held-out **test** split "
        f"({n_images} images, {n_cells:,} cells{dedup_note}), which played no part in training or model "
        "selection, unless stated otherwise. The model is `best.pt`, the epoch with the highest validation "
        "mAP50-95.",
        "",
        "## Summary",
        "",
        f"- **mAP50 {o['mAP50']:.3f} · mAP50-95 {o['mAP50-95']:.3f}** on test, with precision {o['precision']:.3f} "
        f"and recall {o['recall']:.3f} at the F1-optimal confidence.",
        "- Per class (mAP50-95): " + ", ".join(f"{r['class']} {r['mAP50-95']:.3f}" for r in test["per_class"]) + ".",
        f"- At the operating threshold **conf ≥ {op['conf']:.2f}**, chosen on the validation split by maximum F1, "
        "F1 is " + ", ".join(f"{c} {f1[c]['f1']:.3f}" for c in CLASS_NAMES) +
        f". The remaining errors are {missed:,} missed cells, {extra:,} extra detections and {cross:,} cells "
        "found but given the wrong class.",
        f"- **Cell counting:** the WBC count is exact in {cnt['WBC']['exact']:.0%} of images. The RBC count is off "
        f"by {cnt['RBC']['mae']:.2f} cells per image on average (MAE; {cnt['RBC']['mape']:.1%} relative), and the "
        f"platelet count by {cnt['Platelets']['mae']:.2f}.",
        f"- **Speed:** {speed['inference']:.1f} ms per image for inference on a Kaggle "
        f"{summary['environment'].get('gpu', 'GPU')} (+{speed['preprocess']:.1f} ms pre-processing, "
        f"+{speed['postprocess']:.1f} ms NMS and post-processing) at {summary['config']['train']['imgsz']} px, "
        "measured during batched validation.",
        "",
        "## Test metrics per class",
        "",
        "Ultralytics metrics: precision and recall at the confidence that maximizes the mean F1; AP over the "
        "full confidence range.",
        "",
        "| Class | Images | Cells | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in test["per_class"]:
        lines.append(f"| {r['class']} | {r['images']} | {r['instances']:,} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} | {r['mAP50']:.3f} | {r['mAP75']:.3f} | "
                     f"{r['mAP50-95']:.3f} |")
    lines += [
        f"| **all** | {n_images} | {n_cells:,} | **{o['precision']:.3f}** | **{o['recall']:.3f}** | | "
        f"**{o['mAP50']:.3f}** | **{o['mAP75']:.3f}** | **{o['mAP50-95']:.3f}** |",
        "",
        f"![Per-class test metrics]({figures['per_class']})",
        "",
        "## Validation versus test",
        "",
        "| Split | Precision | Recall | mAP50 | mAP75 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, s in (("val", val), ("test", test)):
        v = s["overall"]
        lines.append(f"| {name} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP75']:.3f} | "
                     f"{v['mAP50-95']:.3f} |")
    gap = val["overall"]["mAP50-95"] - o["mAP50-95"]
    lines += [
        "",
        f"Test mAP50-95 is {abs(gap):.3f} {'below' if gap > 0 else 'above'} validation. "
        + ("Some of that gap is expected: validation picked the checkpoint"
           + (f", and {n_dups} validation images show cells that also appear in the training set "
              "(see the dataset report)" if n_dups else "") + "." if gap > 0.005 else
           "The model generalizes from the validation split to unseen test images.")
        + " Both rows validate `best.pt` in full precision. The training report's validation figures were "
          "measured during training in mixed precision, so they can differ in the third decimal.",
        "",
        "## Precision-recall and F1 curves",
        "",
        f"![Curves]({figures['curves']})",
        "",
        "## Operating threshold",
        "",
        f"Deploying the detector requires one confidence threshold. Maximizing F1 on the **validation** split gives "
        f"**{op['conf']:.2f}**. Applying it to test, a detection is correct when it overlaps a cell of the same "
        f"class with IoU ≥ {op['iou']}:",
        "",
        "| Class | Correct (TP) | False detections (FP) | Missed (FN) | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in CLASS_NAMES:
        r = f1[c]
        lines.append(f"| {c} | {r['tp']:,} | {r['fp']:,} | {r['fn']:,} | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{r['f1']:.3f} |")
    lines += [
        "",
        "These detections, like those behind the confusion matrix, the counts, the examples and the dashboard "
        "explorer, come from the inference that `scripts/predict.py` uses: each image first gets a "
        f"{PREDICT_MARGIN}-pixel gray margin at the network's {summary['config']['train']['imgsz']} px scale, about "
        "what Ultralytics' validation adds. Without it, the model also reports cells cut off at the image edge, which "
        "the annotators left out.",
        "",
        "## Confusion matrix",
        "",
        "Class-agnostic matching at the operating threshold. Rows are the annotated class; *no cell* collects "
        "detections without a matching annotation, and *missed* collects annotated cells without a detection. "
        "Shading is the share of the row.",
        "",
        f"![Confusion matrix]({figures['confusion']})",
        "",
        "## Cell counting",
        "",
        "Blood counts are often what matters downstream, so this section compares the number of detected cells "
        "per image with the annotated number, at the operating threshold.",
        "",
        "| Class | Annotated | Detected | MAE per image | Bias per image | Relative error | Exact count | "
        "Within ±1 | Pearson r |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in CLASS_NAMES:
        r = cnt[c]
        mape = "–" if r["mape"] is None else f"{r['mape']:.1%}"
        pr = "–" if r["pearson_r"] is None else f"{r['pearson_r']:.3f}"
        lines.append(f"| {c} | {r['true_total']:,} | {r['pred_total']:,} | {r['mae']:.2f} | {r['bias']:+.2f} | "
                     f"{mape} | {r['exact']:.0%} | {r['within_1']:.0%} | {pr} |")
    lines += [
        "",
        f"![Counting]({figures['counting']})",
        "",
    ]
    if e2e:
        eo = e2e["overall"]
        es = e2e["speed_ms"]
        lines += [
            "## NMS-free head",
            "",
            "YOLO26 also has a one-to-one head that needs no non-maximum suppression (`nms=False`). The default "
            "path above uses the one-to-many head with NMS. On the same test images:",
            "",
            "| Head | Precision | Recall | mAP50 | mAP50-95 | Inference + post-processing (ms/img) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| one-to-many + NMS (default) | {o['precision']:.3f} | {o['recall']:.3f} | {o['mAP50']:.3f} | "
            f"{o['mAP50-95']:.3f} | {speed['inference'] + speed['postprocess']:.1f} |",
            f"| one-to-one, NMS-free | {eo['precision']:.3f} | {eo['recall']:.3f} | {eo['mAP50']:.3f} | "
            f"{eo['mAP50-95']:.3f} | {es['inference'] + es['postprocess']:.1f} |",
            "",
        ]
    lines += [
        "## Examples",
        "",
        "Test images with the most errors (top row) and a random selection (bottom row), at the operating "
        "threshold. Green boxes are correct detections, red boxes false detections, and yellow boxes "
        "annotated cells the model missed.",
        "",
        f"![Examples]({figures['examples']})",
        "",
        "## Limitations",
        "",
        "- TXL-PBC merges several public collections, so the test images come from the same sources and "
        "microscopes as the training images. Accuracy on smears from a new lab, stain or scanner is unknown "
        "until it is measured there.",
        "- Ground truth is one annotation per cell with no inter-annotator agreement, so part of the residual "
        "error, especially for overlapping RBC and tiny platelets, is label noise.",
        "- The model detects and counts three broad classes. It does not tell WBC subtypes apart and is not a "
        "validated diagnostic device.",
        "",
    ]
    path = dest.reports / "evaluation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- main


def explorer_images(records: list[dict], gt: dict, split: str, dest: Dest) -> list[dict]:
    return [{
        "name": r["image"],
        "image": relpath(DATA_DIR / "images" / split / r["image"], dest.dashboard),
        "width": r["width"],
        "height": r["height"],
        "gt": [[int(b[0]), *(round(float(v), 4) for v in b[1:])] for b in gt[r["image"]]],
        "pred": r["boxes"],
    } for r in records]


def build(out_dir: Path, run: dict, console: list[str] | None = None, dest_root: Path = ROOT) -> None:
    dest = Dest(dest_root)
    P.apply_style()
    o = load_outputs(out_dir)
    summary, ev = o["summary"], o["evaluation"]
    iou_thr = summary["config"]["evaluate"]["iou_match"]

    state = final_training_state(run, o["events"], console or console_from_log(out_dir))
    state.write(force=True, directory=dest.dashboard_data)
    sd = state.data

    by_split = {}
    for rec in o["predictions"]["images"]:
        by_split.setdefault(rec["split"], []).append(rec)
    gts = {split: load_ground_truth(split, [r["image"] for r in recs]) for split, recs in by_split.items()}
    conf = ev["splits"]["val"].get("f1_optimal_conf", 0.25)
    ops = {split: operating_point(recs, gts[split], conf, iou_thr) for split, recs in by_split.items()}

    epochs = sd["epochs"]
    best_epoch = (sd["train_end"] or {}).get("best_epoch") or max(epochs, key=lambda e: e["fitness"])["epoch"]
    close = summary["config"]["train"]["close_mosaic"]
    planned = (sd["train_end"] or {}).get("epochs_planned", len(epochs))
    mosaic_off = planned - close + 1 if close and planned - close + 1 <= epochs[-1]["epoch"] else None
    train_figs = {
        "losses": fig_training_losses(epochs, best_epoch, mosaic_off, dest),
        "metrics": fig_training_metrics(epochs, best_epoch, dest),
        "per_class": fig_training_per_class(epochs, best_epoch, dest),
        "schedule": fig_training_schedule(epochs, dest),
    }
    test = ev["splits"]["test"]
    eval_figs = {
        "per_class": fig_eval_per_class(test, dest),
        "curves": fig_eval_curves(test, conf, dest),
        "confusion": fig_eval_confusion(ops["test"], dest),
        "counting": fig_eval_counting(ops["test"], dest),
        "examples": fig_eval_examples(by_split["test"], gts["test"], ops["test"], "test", dest),
    }
    training_report = write_training_report(sd, run, summary, train_figs, dest)
    evaluation_report = write_evaluation_report(ev, ops, summary, run, eval_figs, dest)

    write_dashboard_data("evaluation", {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run,
        "model": summary["config"]["model"],
        "best_epoch": best_epoch,
        "weights_mb": summary["weights_mb"],
        "environment": summary["environment"],
        "classes": list(CLASS_NAMES),
        "iou_match": iou_thr,
        "operating_conf": conf,
        "conf_min": o["predictions"]["conf_min"],
        "splits": ev["splits"],
        "predict_speed_ms": ev["predict_speed_ms"],
        "operating_point": {s: {k: v for k, v in op.items() if k != "images"} for s, op in ops.items()},
        "explorer": explorer_images(by_split["test"], gts["test"], "test", dest),
    }, directory=dest.dashboard_data)

    dest.results.mkdir(parents=True, exist_ok=True)
    dest.models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(o["results_csv"], dest.results / "results.csv")
    shutil.copy2(out_dir / "artifacts" / "events.jsonl", dest.results / "events.jsonl")
    shutil.copy2(out_dir / "artifacts" / "evaluation.json", dest.results / "evaluation.json")
    shutil.copy2(o["weights"], dest.models / f"{summary['run_name']}.pt")
    write_json(dest.results / "metrics.json", {
        "run": run,
        "model": summary["config"]["model"],
        "best_epoch": best_epoch,
        "epochs_completed": (sd["train_end"] or {}).get("epochs_completed"),
        "train_time_s": (sd["train_end"] or {}).get("train_time_s"),
        "environment": summary["environment"],
        "metrics": {split: s["overall"] for split, s in ev["splits"].items()},
        "per_class": {split: s["per_class"] for split, s in ev["splits"].items() if "per_class" in s},
        "operating_point": {s: {k: v for k, v in op.items() if k != "images"} for s, op in ops.items()},
    })
    print(f"Wrote {relpath(training_report, dest.root)}, {relpath(evaluation_report, dest.root)}, "
          "dashboard data, results/ and models/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", type=Path, help="downloaded Kaggle outputs (default: the current run)")
    parser.add_argument("--dest", type=Path, default=ROOT, help="root directory for everything written")
    args = parser.parse_args()
    run = read_current_run()
    out_dir = args.outputs or RUNS_DIR / "kaggle" / f"v{run['version']}"
    version = out_dir.name.removeprefix("v")
    if version.isdigit() and int(version) != run["version"]:
        # An earlier version of the same kernel; only the latest push time is recorded.
        run = {**run, "version": int(version), "pushed_at": None}
    build(out_dir, run, dest_root=args.dest)


if __name__ == "__main__":
    main()
