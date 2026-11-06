"""Find and count cells in a whole microscope picture at any magnification, the way the model saw my smears.

A phone photo through the eyepiece holds hundreds of cells, each tens of pixels wide: shrunk to the model's
640-pixel input they would be a few pixels. So, as scripts/labeling.py did for the training tiles:

1. the model measures the red cells (their median diameter, at a scale where it boxes them reliably);
2. the picture is rescaled so that a red cell is RED_CELL_PX wide;
3. it is cut into 640-pixel tiles overlapping by more than a cell, over the bright field of view;
4. each cell is kept from the tile whose core holds its centre: the cores cover the picture without overlapping,
   and the overlap makes that tile see the cell whole. A last pass drops a cell found twice across a core boundary,
   or under two classes.

Boxes come back in the picture's own pixels. Standalone (numpy, OpenCV and an Ultralytics model) so that the web
app can use it too.
"""

from __future__ import annotations

import numpy as np

RED_CELL_PX = 100   # red cell diameter in the fine-tuning tiles
TILE = 640          # the model's input
OVERLAP = 256       # wider than the largest white cell at that scale (about two red cells)
MIN_FIELD = 0.05    # tiles with less of the field of view are not predicted
DUPLICATE_IOU = 0.7  # boxes overlapping more are the same cell (overlapping red cells stay below)


def resize(img: np.ndarray, scale: float, nearest: bool = False) -> np.ndarray:
    import cv2

    interpolation = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interpolation)


def field_mask(img: np.ndarray) -> np.ndarray:
    """The bright field of view: the eyepiece's circle, or the whole picture when the phone zoomed past it.

    Stray light around the eyepiece can be as bright as the slide but is strongly colored, while the slide, blurred
    over its cells, is pale; saturated red-to-magenta areas are left out.
    """
    import cv2

    h, w = img.shape[:2]
    smooth = cv2.GaussianBlur(img, (0, 0), max(3.0, 0.01 * min(h, w)))
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    p99 = float(np.percentile(gray, 99))
    ch, cw = max(1, h // 10), max(1, w // 10)
    corners = (gray[:ch, :cw], gray[:ch, -cw:], gray[-ch:, :cw], gray[-ch:, -cw:])
    if min(float(np.median(c)) for c in corners) >= 0.3 * p99:
        return np.ones((h, w), dtype=bool)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    s0 = float(np.median(hsv[h // 4:3 * h // 4, w // 4:3 * w // 4, 1]))
    glow = ((hsv[..., 0] < 15) | (hsv[..., 0] > 150)) & (hsv[..., 1] > s0 + 0.2 * 255)
    mask = ((gray > 0.5 * p99) & ~glow).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        mask = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)  # dark clumps inside are field too
    return filled.astype(bool)


def red_cell_size(model, img: np.ndarray, mask: np.ndarray, device=None) -> float | None:
    """Median red cell diameter in the picture's pixels, or None when too few red cells are found.

    The model looks at the field's centre at five scales, each time through a 960-pixel input, and its boxes are
    only trusted at a scale where red cells come out 50 to 180 pixels wide: much larger, it boxes fragments; much
    smaller, its boxes are loose. Of those scales, the one finding the most red cells wins.
    """
    ys, xs = np.nonzero(mask)
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(xs) else (img.shape[0] // 2, img.shape[1] // 2)
    rbc = [k for k, v in model.names.items() if v == "RBC"][0]
    found = []
    for s in (0.25, 0.5, 1.0, 2.0, 3.0):
        half = int(480 / s)
        im = resize(img[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half], s)
        size = int(np.ceil(max(im.shape[:2]) / 32) * 32)
        b = model.predict(im, imgsz=size, conf=0.4, max_det=3000, device=device, verbose=False)[0].boxes.data
        red = b[b[:, 5] == rbc].cpu().numpy()
        if len(red) >= 8:
            native = float(np.median(np.sqrt((red[:, 2] - red[:, 0]) * (red[:, 3] - red[:, 1])))) / s
            found.append((len(red), native, native * s))
    trusted = [f for f in found if 50 <= f[2] <= 180] or found
    return max(trusted, key=lambda f: f[0])[1] if trusted else None


def origins(length: int) -> list[int]:
    """Tile starts along one side: evenly spread, the first and last flush with the picture's edges."""
    if length <= TILE:
        return [0]
    n = int(np.ceil((length - TILE) / (TILE - OVERLAP)))
    return [round(i * (length - TILE) / n) for i in range(n + 1)]


def cores(starts: list[int]) -> list[tuple[float, float]]:
    """Each tile's core along one side: from the middle of its overlap with the previous tile to the next one's."""
    edges = [-np.inf] + [(a + TILE + b) / 2 for a, b in zip(starts, starts[1:])] + [np.inf]
    return list(zip(edges[:-1], edges[1:]))


def drop_duplicates(boxes: np.ndarray) -> np.ndarray:
    """Greedy non-maximum suppression across classes at DUPLICATE_IOU."""
    order = np.argsort(-boxes[:, 4])
    boxes = boxes[order]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = np.ones(len(boxes), bool)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        w = np.clip(np.minimum(boxes[i, 2], boxes[i + 1:, 2]) - np.maximum(boxes[i, 0], boxes[i + 1:, 0]), 0, None)
        h = np.clip(np.minimum(boxes[i, 3], boxes[i + 1:, 3]) - np.maximum(boxes[i, 1], boxes[i + 1:, 1]), 0, None)
        iou = w * h / (area[i] + area[i + 1:] - w * h + 1e-9)
        keep[i + 1:] &= iou <= DUPLICATE_IOU
    return boxes[keep]


def detect(model, img: np.ndarray, conf: float = 0.25, device=None, batch: int = 16,
           red_cell_px: float | None = None, half: bool = False) -> dict:
    """Cells in a BGR picture: boxes as rows of x1, y1, x2, y2, confidence, class in the picture's pixels, with the
    measured red cell size, the scale applied and the number of tiles predicted. red_cell_px skips the measuring
    (a video keeps its magnification); half runs the model in 16-bit floats on a GPU, for speed."""
    import cv2

    field = field_mask(img)
    size = red_cell_px or red_cell_size(model, img, field, device)
    scale = RED_CELL_PX / size if size else 1.0  # nothing to measure: the picture as it is
    big = resize(img, scale)
    big_field = resize(field.astype(np.uint8), scale, nearest=True).astype(bool)
    H, W = big.shape[:2]
    ys, xs = origins(H), origins(W)
    core_y, core_x = cores(ys), cores(xs)
    jobs = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if big_field[y:y + TILE, x:x + TILE].mean() < MIN_FIELD:
                continue
            tile = big[y:y + TILE, x:x + TILE]
            if tile.shape[:2] != (TILE, TILE):  # a picture smaller than a tile: pad it with gray
                tile = cv2.copyMakeBorder(tile, 0, TILE - tile.shape[0], 0, TILE - tile.shape[1],
                                          cv2.BORDER_CONSTANT, value=(114, 114, 114))
            jobs.append((x, y, core_x[j], core_y[i], tile))
    found = []
    for k in range(0, len(jobs), batch):
        chunk = jobs[k:k + batch]
        results = model.predict([job[4] for job in chunk], imgsz=TILE, conf=conf, max_det=1000, device=device,
                                verbose=False, **({"quantize": 16} if half else {}))
        for (x, y, (x0, x1), (y0, y1), _), r in zip(chunk, results):
            b = r.boxes.data.cpu().numpy().copy()
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y
            mx, my = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            found.append(b[(mx >= x0) & (mx < x1) & (my >= y0) & (my < y1)])
    boxes = np.concatenate(found) if found else np.zeros((0, 6), np.float32)
    if len(boxes):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        mx = ((boxes[:, 0] + boxes[:, 2]) / 2).astype(int).clip(0, W - 1)
        my = ((boxes[:, 1] + boxes[:, 3]) / 2).astype(int).clip(0, H - 1)
        boxes = drop_duplicates(boxes[big_field[my, mx]])
        boxes[:, :4] /= scale
    return {"boxes": boxes, "red_cell_px": size, "scale": scale, "tiles": len(jobs)}
