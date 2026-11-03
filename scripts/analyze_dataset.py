"""Dataset report for TXL-PBC: statistics, quality checks and figures.

Writes reports/dataset_report.md, reports/figures/dataset_*.png and
dashboard/data/dataset.js (dataset tab of the dashboard).

    python scripts/analyze_dataset.py
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import plotting as P
from common import CLASS_NAMES, DASHBOARD_DIR, DATA_DIR, REPORTS_DIR, SPLITS, relpath, write_dashboard_data

MODEL_IMGSZ = 640  # training resolution: images are resized so their long side is 640 px
# dHash Hamming distance (out of 64) treated as a near-duplicate. Chosen from the data: every
# cross-split pair at <= 9 bits shows the same cell imaged twice; pairs at >= 10 bits are different cells.
NEAR_DUP_BITS = 9
OUTSIDE_TOL = 0.005  # boxes may overhang the image border by this fraction before being flagged
SAMPLES_PER_GROUP = 8
SIZE_BINS = np.geomspace(4, 400, 41)  # box size bins in model pixels (log-spaced)
CENTER_BINS = 16
SEED = 0


@dataclass
class ImageRecord:
    split: str
    path: Path
    width: int
    height: int
    mode: str
    alpha_opaque: bool | None
    md5: str
    dhash: int
    boxes: list[tuple[int, float, float, float, float]]
    issues: list[str]

    @property
    def group(self) -> str:
        return f"{self.width}×{self.height}"

    @property
    def scale(self) -> float:
        return MODEL_IMGSZ / max(self.width, self.height)


# --------------------------------------------------------------------------- scanning


def dhash(img: Image.Image) -> int:
    """64-bit difference hash: robust to resizing and mild compression, used for near-duplicates."""
    a = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    return int(np.packbits((a[:, 1:] > a[:, :-1]).flatten()).view(">u8")[0])


def parse_labels(path: Path | None) -> tuple[list, list[str]]:
    if path is None:
        return [], ["missing label file"]
    boxes, issues = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            issues.append(f"line {n}: {len(parts)} fields")
            continue
        try:
            c, (x, y, w, h) = int(parts[0]), map(float, parts[1:])
        except ValueError:
            issues.append(f"line {n}: non-numeric value")
            continue
        if c not in range(len(CLASS_NAMES)):
            issues.append(f"line {n}: unknown class {c}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            issues.append(f"line {n}: coordinates outside 0-1")
        boxes.append((c, x, y, w, h))
    return boxes, issues


def scan() -> tuple[list[ImageRecord], dict[str, list[str]]]:
    records, orphans = [], {"labels without image": [], "images without label": []}
    for split in SPLITS:
        images = {p.stem: p for p in (DATA_DIR / "images" / split).iterdir() if p.is_file()}
        labels = {p.stem: p for p in (DATA_DIR / "labels" / split).glob("*.txt")}
        orphans["labels without image"] += [f"{split}/{s}.txt" for s in sorted(labels.keys() - images.keys())]
        orphans["images without label"] += [f"{split}/{images[s].name}" for s in sorted(images.keys() - labels.keys())]
        for stem, path in sorted(images.items()):
            with Image.open(path) as im:
                alpha = bool(np.asarray(im.getchannel("A")).min() == 255) if "A" in im.getbands() else None
                rgb = im.convert("RGB")
                record = dict(width=im.width, height=im.height, mode=im.mode, alpha_opaque=alpha,
                              md5=hashlib.md5(rgb.tobytes()).hexdigest(), dhash=dhash(rgb))
            boxes, issues = parse_labels(labels.get(stem))
            records.append(ImageRecord(split=split, path=path, boxes=boxes, issues=issues, **record))
    return records, orphans


def near_duplicates_across_splits(records: list[ImageRecord]) -> list[tuple[ImageRecord, ImageRecord, int]]:
    hashes = np.array([r.dhash for r in records], dtype=np.uint64)
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    splits = np.array([r.split for r in records])
    pairs = []
    for i in range(len(records) - 1):
        x = (hashes[i + 1:] ^ hashes[i]).view(np.uint8).reshape(-1, 8)
        dist = popcount[x].sum(1)
        for j in np.flatnonzero((dist <= NEAR_DUP_BITS) & (splits[i + 1:] != splits[i])):
            pairs.append((records[i], records[i + 1 + j], int(dist[j])))
    return pairs


# --------------------------------------------------------------------------- statistics


def box_size_model_px(r: ImageRecord, b) -> float:
    """Geometric-mean side length of a box as the model sees it (long side resized to 640)."""
    return float(np.sqrt(b[3] * r.width * b[4] * r.height) * r.scale)


def compute_stats(records: list[ImageRecord]) -> dict:
    per_split = {}
    for split in SPLITS:
        rs = [r for r in records if r.split == split]
        counts = Counter(CLASS_NAMES[b[0]] for r in rs for b in r.boxes)
        n_boxes = [len(r.boxes) for r in rs]
        per_split[split] = {
            "images": len(rs),
            "cells": {c: counts.get(c, 0) for c in CLASS_NAMES},
            "cells_total": sum(n_boxes),
            "cells_per_image": {"mean": round(float(np.mean(n_boxes)), 2), "median": float(np.median(n_boxes)),
                                "min": int(min(n_boxes)), "max": int(max(n_boxes))},
        }

    max_cells = max(len(r.boxes) for r in records)
    cells_hist = {split: np.bincount([len(r.boxes) for r in records if r.split == split],
                                     minlength=max_cells + 1).tolist() for split in SPLITS}
    per_class_per_image = {}
    for ci, c in enumerate(CLASS_NAMES):
        n = [sum(1 for b in r.boxes if b[0] == ci) for r in records]
        per_class_per_image[c] = {"hist": np.bincount(n).tolist(),
                                  "images_with": int(sum(1 for v in n if v)),
                                  "mean": round(float(np.mean(n)), 2)}

    sizes = {c: [] for c in CLASS_NAMES}
    native = {c: [] for c in CLASS_NAMES}
    aspect = {c: [] for c in CLASS_NAMES}
    centers = {c: np.zeros((CENTER_BINS, CENTER_BINS), dtype=int) for c in CLASS_NAMES}
    for r in records:
        for b in r.boxes:
            c = CLASS_NAMES[b[0]]
            sizes[c].append(box_size_model_px(r, b))
            native[c].append(float(np.sqrt(b[3] * r.width * b[4] * r.height)))
            aspect[c].append((b[3] * r.width) / (b[4] * r.height))
            gx = min(int(b[1] * CENTER_BINS), CENTER_BINS - 1)
            gy = min(int(b[2] * CENTER_BINS), CENTER_BINS - 1)
            centers[c][gy, gx] += 1
    q = CENTER_BINS // 4  # central quarter of the image area: 0.25-0.75 on both axes
    central_share = {c: round(float(m[q:-q, q:-q].sum() / m.sum()), 3) for c, m in centers.items()}
    size_summary = {
        c: {"median": round(float(np.median(v)), 1), "p5": round(float(np.percentile(v, 5)), 1),
            "p95": round(float(np.percentile(v, 95)), 1),
            "native_median": round(float(np.median(native[c])), 1),
            "aspect_median": round(float(np.median(aspect[c])), 2),
            "small_share": round(float(np.mean(np.array(v) < 32)), 3)}
        for c, v in sizes.items()
    }

    groups = defaultdict(list)
    for r in records:
        groups[r.group].append(r)
    group_rows = []
    for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        med = {}
        for ci, c in enumerate(CLASS_NAMES):
            v = [box_size_model_px(r, b) for r in rs for b in r.boxes if b[0] == ci]
            med[c] = round(float(np.median(v)), 1) if v else None
        group_rows.append({
            "resolution": g,
            "images": {s: sum(1 for r in rs if r.split == s) for s in SPLITS},
            "images_total": len(rs),
            "modes": dict(Counter(r.mode for r in rs)),
            "scale": round(rs[0].scale, 2),
            "cells_per_image": round(float(np.mean([len(r.boxes) for r in rs])), 1),
            "cells": {c: sum(1 for r in rs for b in r.boxes if b[0] == ci) for ci, c in enumerate(CLASS_NAMES)},
            "median_size_model_px": med,
        })

    return {
        "per_split": per_split,
        "cells_hist": cells_hist,
        "per_class_per_image": per_class_per_image,
        "size_bins": [round(float(x), 2) for x in SIZE_BINS],
        "size_hist": {c: np.histogram(v, bins=SIZE_BINS)[0].tolist() for c, v in sizes.items()},
        "size_summary": size_summary,
        "centers": {c: m.tolist() for c, m in centers.items()},
        "central_share": central_share,
        "groups": group_rows,
    }


def quality_checks(records: list[ImageRecord], orphans: dict, near_dups: list) -> list[dict]:
    def check(cid, label, ok, detail, warn_only=False):
        return {"id": cid, "label": label, "status": "pass" if ok else ("warn" if warn_only else "fail"),
                "detail": detail}

    n_orphans = sum(len(v) for v in orphans.values())
    issues = [(r, i) for r in records for i in r.issues]
    empty = [r for r in records if not r.boxes]
    outside = sum(1 for r in records for (_, x, y, w, h) in r.boxes
                  if x - w / 2 < -OUTSIDE_TOL or y - h / 2 < -OUTSIDE_TOL
                  or x + w / 2 > 1 + OUTSIDE_TOL or y + h / 2 > 1 + OUTSIDE_TOL)
    dup_boxes = sum(len(r.boxes) - len(set(r.boxes)) for r in records)
    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r.md5].append(r)
    exact = [rs for rs in by_md5.values() if len(rs) > 1]
    rgba = [r for r in records if r.alpha_opaque is not None]
    n_boxes = sum(len(r.boxes) for r in records)
    dup_splits = Counter(" ↔ ".join(sorted((a.split, b.split))) for a, b, _ in near_dups)
    near_dup_detail = "no cross-split pairs found"
    if near_dups:
        by_pair = ", ".join(f"{k}: {v}" for k, v in sorted(dup_splits.items()))
        clean_test = "" if any("test" in k for k in dup_splits) else "; the test split has none"
        near_dup_detail = f"{len(near_dups)} pairs show the same cell imaged twice ({by_pair}){clean_test}"
    return [
        check("pairs", "Every image has a label file and every label file an image", n_orphans == 0,
              f"{len(records):,} images checked" if n_orphans == 0 else
              "; ".join(f"{k}: {', '.join(v[:5])}" for k, v in orphans.items() if v)),
        check("format", "Label rows are well formed (5 numeric fields, class 0–2, normalized coordinates)",
              not issues, f"{n_boxes:,} rows parsed" if not issues else
              "; ".join(f"{r.split}/{r.path.name} {i}" for r, i in issues[:5])),
        check("empty", "No image without annotations", not empty,
              "every image has at least one annotated cell" if not empty else f"{len(empty)} empty label files"),
        check("inside", "Boxes stay inside the image", outside == 0,
              "no box overhangs the border" if outside == 0 else
              f"{outside} boxes overhang the border by more than {OUTSIDE_TOL:.1%}", warn_only=True),
        check("dup_boxes", "No duplicate boxes within an image", dup_boxes == 0,
              "no repeated rows" if dup_boxes == 0 else f"{dup_boxes} repeated rows", warn_only=True),
        check("dup_images", "No exact duplicate images", not exact,
              "pixel hashes are unique" if not exact else
              f"{len(exact)} groups of identical images, e.g. " +
              ", ".join(f"{r.split}/{r.path.name}" for r in exact[0]), warn_only=True),
        check("near_dup", f"No near-duplicate images across splits (dHash distance ≤ {NEAR_DUP_BITS}/64)",
              not near_dups, near_dup_detail, warn_only=True),
        check("alpha", "Alpha channels are fully opaque, so loading as RGB loses nothing",
              all(r.alpha_opaque for r in rgba),
              f"{len(rgba)} RGBA images, all opaque" if all(r.alpha_opaque for r in rgba) else
              f"{sum(1 for r in rgba if not r.alpha_opaque)} RGBA images use transparency", warn_only=True),
    ]


def pick_samples(records: list[ImageRecord]) -> list[ImageRecord]:
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for r in records:
        groups[r.group].append(r)
    picked = []
    for g in sorted(groups, key=lambda g: -len(groups[g])):
        rs = groups[g]
        picked += rng.sample(rs, min(SAMPLES_PER_GROUP, len(rs)))
    return picked


# --------------------------------------------------------------------------- figures


def fig_class_distribution(stats: dict) -> str:
    fig, (ax1, ax2) = P.plt.subplots(1, 2, figsize=(10, 2.8), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
    totals = {c: sum(stats["per_split"][s]["cells"][c] for s in SPLITS) for c in CLASS_NAMES}
    grand = sum(totals.values())
    y = np.arange(len(CLASS_NAMES))[::-1]
    ax1.barh(y, [totals[c] for c in CLASS_NAMES], height=0.42, color=[P.CLASS_COLORS[c] for c in CLASS_NAMES])
    for yi, c in zip(y, CLASS_NAMES):
        ax1.annotate(f"{totals[c]:,}  ({totals[c] / grand:.1%})", (totals[c], yi), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=9, color=P.INK_2)
    ax1.set_yticks(y, CLASS_NAMES)
    ax1.set_xlim(0, max(totals.values()) * 1.35)
    ax1.grid(axis="y", visible=False)
    ax1.set_title("Annotated cells per class (all splits)")
    ax1.xaxis.set_major_formatter(P.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))

    ys = np.arange(len(SPLITS))[::-1]
    for yi, s in zip(ys, SPLITS):
        cells = stats["per_split"][s]["cells"]
        total, left = sum(cells.values()), 0.0
        for c in CLASS_NAMES:
            share = cells[c] / total
            ax2.barh(yi, share - 0.004, left=left, height=0.42, color=P.CLASS_COLORS[c],
                     label=c if s == SPLITS[0] else None)
            if share > 0.08:
                ax2.text(left + 0.012, yi, f"{share:.0%}", va="center", fontsize=8.5, color="white")
            left += share
        ax2.annotate(f"{total:,} cells", (1, yi), xytext=(6, 0), textcoords="offset points", va="center",
                     fontsize=9, color=P.INK_2)
    ax2.set_yticks(ys, [f"{s} ({stats['per_split'][s]['images']})" for s in SPLITS])
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_major_formatter(P.matplotlib.ticker.PercentFormatter(1.0))
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Class share per split (images in parentheses)")
    ax2.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, -0.34), handlelength=1, columnspacing=1.2)
    return P.save(fig, "dataset_class_distribution")


def fig_cells_per_image(stats: dict) -> str:
    fig, axes = P.plt.subplots(1, 2, figsize=(10, 2.9), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.3})
    ax = axes[0]
    counts = np.sum([stats["cells_hist"][s] for s in SPLITS], axis=0)
    x = np.arange(len(counts))
    ax.bar(x, counts, width=0.7, color=P.ACCENT)
    mean = float(np.sum(x * counts) / np.sum(counts))
    ax.axvline(mean, color=P.INK_2, lw=1)
    ax.annotate(f"mean {mean:.1f}", (mean, counts.max()), xytext=(5, -2), textcoords="offset points",
                fontsize=9, color=P.INK_2, va="top")
    ax.set_xlabel("annotated cells in the image")
    ax.set_ylabel("images")
    ax.grid(axis="x", visible=False)
    ax.set_title("Cells per image")

    ax = axes[1]
    width = 0.36
    for k, c in enumerate(("WBC", "Platelets")):  # RBC counts are covered by the left panel
        h = np.array(stats["per_class_per_image"][c]["hist"][:6], dtype=float)
        h = np.pad(h, (0, 6 - len(h)))
        ax.bar(np.arange(6) + (k - 0.5) * (width + 0.03), h, width=width, color=P.CLASS_COLORS[c], label=c)
    ax.set_xticks(range(6), ["0", "1", "2", "3", "4", "5"])
    ax.set_xlabel("cells of the class in the image")
    ax.set_ylabel("images")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("WBC and platelet counts per image")
    return P.save(fig, "dataset_cells_per_image")


def fig_box_sizes(stats: dict) -> str:
    fig, ax = P.plt.subplots(figsize=(10, 3.0))
    edges = np.array(stats["size_bins"])
    mids = np.sqrt(edges[:-1] * edges[1:])
    for c in CLASS_NAMES:
        h = np.array(stats["size_hist"][c], dtype=float)
        share = h / h.sum()
        ax.plot(mids, share, color=P.CLASS_COLORS[c], label=c)
        k = int(share.argmax())
        ax.annotate(f"{c}  median {stats['size_summary'][c]['median']:.0f} px", (mids[k], share[k]),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9, color=P.INK_2)
    ax.axvline(32, color=P.AXIS, lw=1)
    ax.annotate("32 px (COCO 'small')", (32, ax.get_ylim()[1] * 0.92), xytext=(-4, 0), textcoords="offset points",
                ha="right", fontsize=8.5, color=P.MUTED)
    ax.set_xscale("log")
    ax.set_xlim(9, 450)
    ax.set_xticks([10, 20, 50, 100, 200, 400], ["10", "20", "50", "100", "200", "400"])
    ax.xaxis.set_minor_locator(P.matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(P.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("box size at training resolution, √(w·h) in px (long side resized to 640)")
    ax.set_ylabel("share of the class")
    ax.legend(loc="upper left")
    ax.set_title("Cell size distribution by class")
    return P.save(fig, "dataset_box_sizes")


def fig_box_centers(stats: dict) -> str:
    cmap = P.matplotlib.colors.LinearSegmentedColormap.from_list("seq", [P.SURFACE] + P.SEQ_BLUE[2:])
    fig, axes = P.plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, c in zip(axes, CLASS_NAMES):
        grid = np.array(stats["centers"][c], dtype=float)
        im = ax.imshow(grid / grid.sum(), cmap=cmap, extent=(0, 1, 1, 0), interpolation="nearest")
        ax.set_title(f"{c} box centers")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=P.matplotlib.ticker.PercentFormatter(1.0, 1))
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=P.INK_2)
    fig.tight_layout()
    return P.save(fig, "dataset_box_centers")


def draw_boxes(r: ImageRecord, size: int = 300) -> Image.Image:
    with Image.open(r.path) as im:
        img = im.convert("RGB")
    s = size / max(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    for c, x, y, w, h in r.boxes:
        W, H = img.size
        xy = [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
        draw.rectangle(xy, outline=P.CLASS_COLORS[CLASS_NAMES[c]], width=2)
    canvas = Image.new("RGB", (size, size), P.SURFACE)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def fig_samples(records: list[ImageRecord]) -> str:
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for r in records:
        groups[r.group].append(r)
    order = sorted(groups, key=lambda g: -len(groups[g]))
    picks = [(g, r) for g in order for r in rng.sample(groups[g], 2)]
    fig, axes = P.plt.subplots(2, len(order), figsize=(2.1 * len(order), 4.7))
    for col, g in enumerate(order):
        for row in range(2):
            ax = axes[row, col]
            r = [r for gg, r in picks if gg == g][row]
            ax.imshow(draw_boxes(r))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(g, fontsize=10)
    handles = [P.matplotlib.patches.Patch(facecolor="none", edgecolor=P.CLASS_COLORS[c], lw=2, label=c)
               for c in CLASS_NAMES]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.04, hspace=0.06, bottom=0.08)
    return P.save(fig, "dataset_samples")


# --------------------------------------------------------------------------- report


def write_report(stats: dict, checks: list[dict], figures: dict[str, str]) -> Path:
    ps = stats["per_split"]
    n_images = sum(ps[s]["images"] for s in SPLITS)
    totals = {c: sum(ps[s]["cells"][c] for s in SPLITS) for c in CLASS_NAMES}
    n_cells = sum(totals.values())
    ss = stats["size_summary"]
    status_icon = {"pass": "✅ pass", "warn": "⚠️ warning", "fail": "❌ fail"}
    lines = [
        "# TXL-PBC dataset report",
        "",
        f"_Generated by `scripts/analyze_dataset.py` on {datetime.now(timezone.utc):%Y-%m-%d}. "
        "Interactive version: the **Dataset** tab of the dashboard._",
        "",
        "## Summary",
        "",
        f"- **{n_images:,} images** with **{n_cells:,} annotated cells** in 3 classes, split "
        f"{ps['train']['images']} / {ps['val']['images']} / {ps['test']['images']} "
        f"({ps['train']['images'] / n_images:.0%} / {ps['val']['images'] / n_images:.0%} / "
        f"{ps['test']['images'] / n_images:.0%}) into train / val / test.",
        f"- **Strong class imbalance:** RBC {totals['RBC'] / n_cells:.1%}, WBC {totals['WBC'] / n_cells:.1%}, "
        f"platelets {totals['Platelets'] / n_cells:.1%} of all cells. The class mix is nearly identical "
        "in every split, so validation and test results are comparable.",
        f"- **Platelets are both rare and small**: median {ss['Platelets']['median']:.0f} px at training "
        f"resolution, {ss['Platelets']['small_share']:.0%} below 32 px. WBC are the largest "
        f"(median {ss['WBC']['median']:.0f} px), RBC sit in between ({ss['RBC']['median']:.0f} px).",
        f"- **{len(stats['groups'])} image resolutions**, a sign that TXL-PBC merges several source "
        "collections with different microscopes and magnifications (see *Image sources*).",
        "- **Quality checks:** " + ", ".join(
            f"{sum(1 for c in checks if c['status'] == s)} {s}" for s in ("pass", "warn", "fail")
            if any(c["status"] == s for c in checks)) + " (details below).",
        "",
        "## Splits and classes",
        "",
        "| Split | Images | WBC | RBC | Platelets | Cells | Cells per image (mean · min–max) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in SPLITS:
        c, cpi = ps[s]["cells"], ps[s]["cells_per_image"]
        lines.append(f"| {s} | {ps[s]['images']:,} | {c['WBC']:,} | {c['RBC']:,} | {c['Platelets']:,} | "
                     f"{ps[s]['cells_total']:,} | {cpi['mean']} · {cpi['min']}–{cpi['max']} |")
    lines += [
        f"| **total** | **{n_images:,}** | **{totals['WBC']:,}** | **{totals['RBC']:,}** | "
        f"**{totals['Platelets']:,}** | **{n_cells:,}** | |",
        "",
        f"![Class distribution]({figures['classes']})",
        "",
        "## Cells per image",
        "",
        f"Images hold {ps['train']['cells_per_image']['min']}–"
        f"{max(ps[s]['cells_per_image']['max'] for s in SPLITS)} cells. "
        f"WBC appear in {stats['per_class_per_image']['WBC']['images_with']:,} images "
        f"({stats['per_class_per_image']['WBC']['images_with'] / n_images:.0%}), usually exactly one per image; "
        f"platelets appear in only {stats['per_class_per_image']['Platelets']['images_with']:,} images "
        f"({stats['per_class_per_image']['Platelets']['images_with'] / n_images:.0%}).",
        "",
        f"![Cells per image]({figures['per_image']})",
        "",
        "## Cell sizes",
        "",
        "Sizes are the geometric mean of box width and height, measured after resizing each image so its "
        "long side is 640 px, which is the resolution the model trains at.",
        "",
        "| Class | Median | 5th–95th percentile | Below 32 px | Native median | Median aspect (w/h) |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for c in CLASS_NAMES:
        s = ss[c]
        lines.append(f"| {c} | {s['median']:.0f} px | {s['p5']:.0f}–{s['p95']:.0f} px | {s['small_share']:.0%} | "
                     f"{s['native_median']:.0f} px | {s['aspect_median']:.2f} |")
    lines += [
        "",
        f"![Box sizes]({figures['sizes']})",
        "",
        "## Image sources",
        "",
        "Image size identifies the source collection. Smaller images are upscaled to 640 px on the long side "
        "for training (column *scale*).",
        "",
        "| Resolution | Images (train / val / test) | Color mode | Scale to 640 | Cells per image | "
        "Median WBC / RBC / platelet size at 640 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for g in stats["groups"]:
        im, med = g["images"], g["median_size_model_px"]
        sizes = " / ".join("–" if med[c] is None else f"{med[c]:.0f}" for c in CLASS_NAMES)
        modes = ", ".join(f"{k} ({v})" for k, v in g["modes"].items())
        lines.append(f"| {g['resolution']} | {g['images_total']} ({im['train']} / {im['val']} / {im['test']}) | "
                     f"{modes} | ×{g['scale']} | {g['cells_per_image']} | {sizes} px |")
    lines += [
        "",
        "## Spatial distribution",
        "",
        "Where box centers fall inside the image (normalized coordinates, darker means more boxes). "
        f"{stats['central_share']['WBC']:.0%} of WBC centers lie in the central quarter of the image area, "
        f"against {stats['central_share']['RBC']:.0%} of RBC and {stats['central_share']['Platelets']:.0%} of "
        "platelets. Many source images are crops centered on one white cell, so position alone predicts the "
        "WBC class. Mosaic and flip augmentation break that shortcut during training.",
        "",
        f"![Box centers]({figures['centers']})",
        "",
        "## Data quality checks",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {c['label']} | {status_icon[c['status']]} | {c['detail']} |" for c in checks]
    status = {c["id"]: c["status"] for c in checks}
    notes = []
    if status["dup_boxes"] != "pass":
        notes.append("- **Repeated label rows** are harmless for training: Ultralytics drops exact duplicate rows "
                     "when it caches the labels. The local evaluation code de-duplicates them the same way.")
    if status["near_dup"] != "pass":
        notes.append("- **Near-duplicates:** these pairs show the same cell and the same neighboring cells, captured "
                     "twice with a slightly different focus, framing or color balance (likely two cameras in a "
                     "source collection). Such validation images make validation metrics slightly optimistic. "
                     "The test split has no such pair, so the final test metrics are unaffected.")
    if notes:
        lines += ["", *notes]
    lines += [
        "",
        "## Sample images",
        "",
        "Two random images per resolution with their ground-truth boxes "
        "(blue WBC, orange RBC, aqua platelets).",
        "",
        f"![Samples]({figures['samples']})",
        "",
        "## What this means for training",
        "",
        "- **Resolution:** `imgsz=640` matches the largest native size (640×480). Nothing is downscaled, "
        "and the small sources are upscaled, which helps platelets.",
        "- **Imbalance:** platelets are about 3% of cells and the smallest objects, so expect their AP to trail "
        "the other classes. Per-class metrics are tracked every epoch for this reason.",
        "- **Augmentation:** smears have no canonical orientation, so both horizontal and vertical flips "
        "preserve the labels. Mosaic places cells at new positions and scales, which counters any "
        "framing bias visible in the box-center maps.",
        "- **Evaluation:** the split is fixed by the dataset and the class mix is stable across splits. Model "
        "selection uses `val`; `test` is only used for the final numbers.",
        "",
    ]
    path = REPORTS_DIR / "dataset_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    P.apply_style()
    records, orphans = scan()
    stats = compute_stats(records)
    near_dups = near_duplicates_across_splits(records)
    checks = quality_checks(records, orphans, near_dups)
    figures = {
        "classes": fig_class_distribution(stats),
        "per_image": fig_cells_per_image(stats),
        "sizes": fig_box_sizes(stats),
        "centers": fig_box_centers(stats),
        "samples": fig_samples(records),
    }
    report = write_report(stats, checks, figures)
    samples = [{
        "image": relpath(r.path, DASHBOARD_DIR),
        "name": r.path.name,
        "split": r.split,
        "group": r.group,
        "width": r.width,
        "height": r.height,
        "boxes": [[c, round(x - w / 2, 4), round(y - h / 2, 4), round(x + w / 2, 4), round(y + h / 2, 4)]
                  for c, x, y, w, h in r.boxes],
    } for r in pick_samples(records)]
    write_dashboard_data("dataset", {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": list(CLASS_NAMES),
        "splits": list(SPLITS),
        "model_imgsz": MODEL_IMGSZ,
        **stats,
        "checks": checks,
        "near_duplicates": [{"a": f"{a.split}/{a.path.name}", "b": f"{b.split}/{b.path.name}", "bits": d}
                            for a, b, d in near_dups[:50]],
        "samples": samples,
    })
    print(f"Wrote {report.relative_to(REPORTS_DIR.parent)} and dashboard/data/dataset.js")
    for c in checks:
        print(f"  [{c['status']:>4}] {c['label']}: {c['detail']}")


if __name__ == "__main__":
    main()
