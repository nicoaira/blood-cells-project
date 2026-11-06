// Display-only tracking: no model weights or still-image measurements are changed.
// Scores are temporal averages, rather than independently calibrated probabilities.
class LiveTracker {
  constructor({ holdMs = 400, positionMs = 100, scoreMs = 350, labelMs = 300 } = {}) {
    Object.assign(this, { holdMs, positionMs, scoreMs, labelMs });
    this.reset();
  }

  reset() {
    this.tracks = [];
    this.lastTime = null;
    this.shape = null;
  }

  static iou(a, b) {
    const width = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const height = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const intersection = width * height;
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection || 1);
  }

  update(boxes, now, conf, width, height) {
    const shape = `${width}:${height}`;
    // A new crop, a long pause, or a backwards clock must not inherit labels from the previous view.
    if (this.shape !== shape || (this.lastTime !== null && (now < this.lastTime || now - this.lastTime > 10000))) {
      this.reset();
    }
    this.shape = shape;
    this.lastTime = now;
    const pairs = [];
    for (let t = 0; t < this.tracks.length; t++) for (let b = 0; b < boxes.length; b++) {
      const overlap = LiveTracker.iou(this.tracks[t].lastBox, boxes[b]);
      if (overlap >= 0.3) pairs.push({ t, b, overlap });
    }
    pairs.sort((a, b) => b.overlap - a.overlap);
    const matchedTracks = new Set(), matchedBoxes = new Set();
    const matches = [];
    for (const p of pairs) {
      if (matchedTracks.has(p.t) || matchedBoxes.has(p.b)) continue;
      matchedTracks.add(p.t); matchedBoxes.add(p.b); matches.push(p);
    }
    const confirmed = this.tracks.filter((t) => t.confirmed).length;
    if (confirmed >= 6 && boxes.length >= 6 && matches.length < 0.2 * Math.min(confirmed, boxes.length)) {
      // Most cells no longer overlap: the slide probably moved to another field. Clear held boxes immediately.
      this.tracks = [];
      matches.length = 0; matchedTracks.clear(); matchedBoxes.clear();
    }
    for (const { t: index, b: boxIndex } of matches) {
      const t = this.tracks[index], box = boxes[boxIndex];
      const dt = Math.max(1, now - t.updated);
      const positionAlpha = 1 - Math.exp(-dt / this.positionMs);
      const scoreAlpha = 1 - Math.exp(-dt / this.scoreMs);
      t.box = t.box.map((v, i) => v + positionAlpha * (box[i] - v));
      for (const cls of Object.keys(t.scores)) t.scores[cls] *= 1 - scoreAlpha;
      t.scores[box[5]] = (t.scores[box[5]] || 0) + scoreAlpha * box[4];
      const best = Number(Object.keys(t.scores).reduce((a, b) => t.scores[a] >= t.scores[b] ? a : b));
      // One ambiguous frame cannot immediately change a cell's class and its count.
      const oldClass = t.cls;
      if (t.scores[best] > 1.2 * t.scores[t.cls]) t.cls = best;
      t.seen = now; t.updated = now; t.hits++;
      t.lastBox = box.slice(0, 4);
      if (box[6]) {
        const alpha = 1 - Math.exp(-dt / 600);
        t.subtypeScores = t.subtypeScores
          ? t.subtypeScores.map((p, i) => p + alpha * (box[6][i] - p))
          : [...box[6]];
      }
      const score = t.scores[t.cls];
      if (t.hits >= 2 && score >= conf) t.confirmed = true;
      if (score < Math.max(0, conf - 0.08)) t.confirmed = false;
      if (t.cls !== oldClass || now - t.labelUpdated >= this.labelMs) {
        t.labelScore = score;
        t.subtypeLabelScores = t.subtypeScores ? [...t.subtypeScores] : null;
        t.labelUpdated = now;
      }
    }
    // Unconfirmed, one-frame detections disappear as soon as their next frame fails to match.
    this.tracks = this.tracks.filter((t, i) => matchedTracks.has(i) || (t.confirmed && now - t.seen <= this.holdMs));
    for (let b = 0; b < boxes.length; b++) {
      if (matchedBoxes.has(b)) continue;
      const box = boxes[b];
      this.tracks.push({ box: box.slice(0, 4), lastBox: box.slice(0, 4), cls: box[5], scores: { [box[5]]: box[4] },
        subtypeScores: box[6] ? [...box[6]] : null, subtypeLabelScores: box[6] ? [...box[6]] : null,
        labelScore: box[4], labelUpdated: now, seen: now, updated: now, hits: 1, confirmed: false });
    }
    return this.tracks.filter((t) => t.confirmed).map((t) => [...t.box, t.labelScore, t.cls, t.subtypeLabelScores]);
  }
}
