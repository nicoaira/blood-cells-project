/* Blood cell detector dashboard.
 *
 * Renders the data files written by the project scripts:
 *   data/training.js    live run state (scripts/monitor.py rewrites it every few seconds during a run)
 *   data/evaluation.js  test-set evaluation (scripts/build_reports.py, once the run is over)
 *   data/dataset.js     dataset statistics (scripts/analyze_dataset.py)
 * Each file assigns window.BC_DATA[name]. Loading them with <script> tags instead of fetch() keeps the
 * page working when it is opened straight from disk. Text derived from data is always set with
 * textContent, never parsed as HTML.
 */
"use strict";

(() => {
  const DATA = (window.BC_DATA = window.BC_DATA || {});
  const CLASSES = ["WBC", "RBC", "Platelets"];
  const CLASS_COLOR = { WBC: "var(--c-wbc)", RBC: "var(--c-rbc)", Platelets: "var(--c-plt)" };
  const STAGES = [
    ["queued", "Queued"], ["install", "Install"], ["data", "Data"], ["train", "Train"],
    ["evaluate", "Evaluate"], ["export", "Export"], ["done", "Done"],
  ];
  const LOSS_TITLES = { box_loss: "Box loss", cls_loss: "Classification loss", l1_loss: "L1 box loss", dfl_loss: "DFL loss" };
  const POLL_LIVE_MS = 5000;
  const POLL_IDLE_MS = 30000;
  const STALE_AFTER_S = 90;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const SEQ_LIGHT = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
  const SEQ_DARK = ["#1a1a19", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"];

  const ui = {
    tab: "training",
    split: "test",
    conf: null,
    explorer: null,
    listScroll: 0,
    explorerMode: "errors",
    explorerSort: "errors",
    tables: new Set(),
    hovering: false,
    rendered: {},
  };

  // ------------------------------------------------------------------ DOM helpers

  function append(el, children) {
    for (const c of children.flat(Infinity)) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    return append(el, children);
  }

  function s(tag, attrs, ...children) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
    return append(el, children);
  }

  const ICON_PATHS = {
    check: "M3.5 8.5l3 3 6-7",
    cross: "M4.5 4.5l7 7M11.5 4.5l-7 7",
    warn: "M8 2.5l6 11H2zM8 7v3M8 12v.01",
    dot: "M8 5.5a2.5 2.5 0 1 1 0 5a2.5 2.5 0 1 1 0-5",
    clock: "M8 2.5a5.5 5.5 0 1 1 0 11a5.5 5.5 0 1 1 0-11M8 5v3.2l2 1.3",
    external: "M9 3h4v4M13 3L7.5 8.5M11 9.5V13H3V5h3.5",
  };

  function icon(name, cls = "ico") {
    return s("svg", { class: cls, viewBox: "0 0 16 16", "aria-hidden": "true" },
      s("path", { d: ICON_PATHS[name], fill: "none", stroke: "currentColor", "stroke-width": 1.8,
        "stroke-linecap": "round", "stroke-linejoin": "round" }));
  }

  // ------------------------------------------------------------------ formatting

  const nfCache = {};
  const nf = (d) => (nfCache[d] ||= new Intl.NumberFormat("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
  const ok = (v) => typeof v === "number" && Number.isFinite(v);
  const fmt = {
    int: (v) => (ok(v) ? Math.round(v).toLocaleString("en-US") : "–"),
    num: (v, d = 3) => (ok(v) ? nf(d).format(v) : "–"),
    pct: (v, d = 0) => (ok(v) ? `${nf(d).format(v * 100)}%` : "–"),
    signed: (v, d = 2) => (ok(v) ? `${v > 0 ? "+" : ""}${nf(d).format(v)}` : "–"),
    dur(sec) {
      if (!ok(sec)) return "–";
      sec = Math.max(0, Math.round(sec));
      const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
      if (hh) return `${hh} h ${String(mm).padStart(2, "0")} min`;
      if (mm) return `${mm} min ${String(ss).padStart(2, "0")} s`;
      return `${ss} s`;
    },
    clock: (t) => (ok(t) ? new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "–"),
    date: (t) => (ok(t) ? new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "–"),
    compact: (v) => (Math.abs(v) >= 10000 ? `${nf(0).format(v / 1000)}K` : nf(0).format(v)),
  };
  const nowS = () => Date.now() / 1000;

  // ------------------------------------------------------------------ tooltip

  const tooltip = document.getElementById("tooltip");

  function keyEl(kind, color) {
    const cls = kind === "rect" ? "key-rect" : kind === "box" ? "key-box" : "key-line";
    return h("span", { class: cls, style: kind === "box" ? { borderColor: color } : { background: color } });
  }

  function showTip(x, y, title, rows) {
    tooltip.replaceChildren();
    if (title) tooltip.append(h("div", { class: "tt-title" }, title));
    for (const r of rows) {
      tooltip.append(h("div", { class: "tt-row" }, r.color ? keyEl(r.kind || "line", r.color) : null,
        h("strong", null, r.value), h("span", null, r.label || "")));
    }
    tooltip.style.display = "block";
    const pad = 14, w = tooltip.offsetWidth, ht = tooltip.offsetHeight;
    let left = x + pad, top = y + pad;
    if (left + w > innerWidth - 8) left = x - w - pad;
    if (top + ht > innerHeight - 8) top = y - ht - pad;
    tooltip.style.left = `${Math.max(8, left)}px`;
    tooltip.style.top = `${Math.max(8, top)}px`;
  }

  const hideTip = () => { tooltip.style.display = "none"; };

  function tipFor(el, getContent) {
    const show = (ev) => {
      const [title, rows] = getContent();
      const r = el.getBoundingClientRect();
      showTip(ev.clientX ?? r.right, ev.clientY ?? r.top, title, rows);
    };
    el.addEventListener("pointermove", show);
    el.addEventListener("pointerleave", hideTip);
    el.addEventListener("focus", show);
    el.addEventListener("blur", hideTip);
    if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "0");
    return el;
  }

  // ------------------------------------------------------------------ chart plumbing

  const drawers = new WeakMap();
  const resizer = new ResizeObserver((entries) => {
    for (const e of entries) {
      const draw = drawers.get(e.target);
      const w = Math.round(e.contentRect.width);
      if (draw && w && w !== draw.lastWidth) { draw.lastWidth = w; draw(); }
    }
  });

  function observe(el, draw) {
    drawers.set(el, draw);
    resizer.observe(el);
    el.addEventListener("pointerenter", () => { ui.hovering = true; });
    el.addEventListener("pointerleave", () => { ui.hovering = false; });
  }

  function linear(d0, d1, r0, r1) {
    const k = (r1 - r0) / (d1 - d0 || 1);
    const f = (v) => r0 + (v - d0) * k;
    f.invert = (p) => d0 + (p - r0) / k;
    return f;
  }

  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || Math.abs(hi) || 1;
    const raw = span / Math.max(1, n);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const err = raw / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const ticks = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-6; v += step) ticks.push(+v.toPrecision(12));
    return ticks;
  }

  function nearestIndex(xs, x) {
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] < x) lo = mid; else hi = mid;
    }
    return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
  }

  function isDark() {
    const t = document.documentElement.dataset.theme;
    return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function seqColor(t) {
    const stops = isDark() ? SEQ_DARK : SEQ_LIGHT;
    const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const a = stops[i].match(/\w\w/g).map((v) => parseInt(v, 16));
    const b = stops[i + 1].match(/\w\w/g).map((v) => parseInt(v, 16));
    const f = x - i;
    const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
    const lum = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
    return { fill: `rgb(${c.join(",")})`, text: lum < 0.5 ? "#ffffff" : "#0b0b0b" };
  }

  // ------------------------------------------------------------------ line chart

  function drawEndLabels(g, items, W, H) {
    const gap = 13;
    items.sort((a, b) => a.y - b.y);
    let prev = -Infinity;
    for (const it of items) { it.ly = Math.max(it.y, prev + gap); prev = it.ly; }
    let limit = H;
    for (let k = items.length - 1; k >= 0; k--) { items[k].ly = Math.min(items[k].ly, limit); limit = items[k].ly - gap; }
    for (const it of items) {
      if (Math.abs(it.ly - it.y) > 3) g.append(s("path", { class: "marker-line", d: `M${it.x + 4},${it.y}L${W + 6},${it.ly}` }));
      g.append(s("text", { x: W + 9, y: it.ly, "dominant-baseline": "middle" }, it.label));
    }
  }

  function lineChart(opts) {
    const {
      xs, series, height = 180, yDomain, yFormat = (v) => fmt.num(v, 3), yTickFormat, xFormat = (v) => String(v),
      xTicks, markers = [], bands = [], endLabels = true, label = "", tipTitle = (x) => `Epoch ${x}`,
    } = opts;
    const root = h("div", { class: "chart", style: { minHeight: `${height}px` } });
    function draw() {
      const width = Math.floor(root.clientWidth);
      if (!width) return;
      root.replaceChildren();
      const values = series.flatMap((se) => se.values).filter(ok);
      if (!xs.length || !values.length) {
        root.append(h("div", { class: "empty" }, "Waiting for the first data points"));
        return;
      }
      const endText = (se) => String(se.endLabel ?? se.label);
      const labelled = series.filter((se) => se.endLabel !== false);
      const right = endLabels && labelled.length ? Math.min(110, 14 + 6.4 * Math.max(...labelled.map((se) => endText(se).length))) : 12;
      const m = { top: 10, right, bottom: 28, left: 42 };
      const W = Math.max(40, width - m.left - m.right), H = height - m.top - m.bottom;
      let lo, hi;
      if (yDomain) [lo, hi] = yDomain;
      else {
        lo = Math.min(...values); hi = Math.max(...values);
        const pad = (hi - lo) * 0.1 || Math.abs(hi) * 0.1 || 0.5;
        lo -= pad; hi += pad;
      }
      const yt = niceTicks(lo, hi, Math.max(2, Math.round(H / 38)));
      if (!yDomain) { lo = Math.min(lo, yt[0]); hi = Math.max(hi, yt[yt.length - 1]); }
      const step = yt.length > 1 ? yt[1] - yt[0] : 1;
      const autoTick = (v) => fmt.num(v, Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9))));
      const x0 = xs[0], x1 = xs.length > 1 ? xs[xs.length - 1] : xs[0] + 1;
      const X = linear(x0, x1, 0, W), Y = linear(lo, hi, H, 0);
      const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, height, tabindex: "0", role: "img", "aria-label": label });
      const g = s("g", { transform: `translate(${m.left},${m.top})` });
      svg.append(g);
      for (const b of bands) {
        const a = Math.max(x0, b.from), z = Math.min(x1, b.to);
        if (z > a) g.append(s("rect", { class: "marker-band", x: X(a), y: 0, width: X(z) - X(a), height: H }));
      }
      const grid = s("g", { class: "grid" });
      g.append(grid);
      for (const t of yt) {
        if (t < lo - 1e-9 || t > hi + 1e-9) continue;
        grid.append(s("line", { x1: 0, x2: W, y1: Y(t), y2: Y(t) }));
        g.append(s("g", { class: "tick" }, s("text", { x: -8, y: Y(t), "text-anchor": "end", "dominant-baseline": "middle" },
          (yTickFormat || autoTick)(t))));
      }
      g.append(s("line", { class: "axis-line", x1: 0, x2: W, y1: H, y2: H }));
      const intX = xs.every(Number.isInteger);  // epochs: whole-number ticks only
      const xt = xTicks || niceTicks(x0, x1, Math.min(intX ? Math.max(1, x1 - x0) : Infinity, Math.max(3, Math.floor(W / 55))))
        .filter((v) => v >= x0 && v <= x1 && (!intX || Number.isInteger(v)));
      for (const t of xt) {
        g.append(s("g", { class: "tick" }, s("text", { x: X(t), y: H + 18, "text-anchor": "middle" }, xFormat(t))));
      }
      for (const mk of markers) {
        if (!ok(mk.x) || mk.x < x0 || mk.x > x1) continue;
        g.append(s("line", { class: "marker-line", x1: X(mk.x), x2: X(mk.x), y1: 0, y2: H }));
        if (mk.label) {
          // Top by default; "bottom" keeps the label off curves that run along the top of the plot.
          const bottom = mk.at === "bottom";
          const leftOfLine = bottom ? X(mk.x) > W / 2 : X(mk.x) > W - 70;
          g.append(s("text", { class: "marker-label", x: X(mk.x) + (leftOfLine ? -4 : 4), y: bottom ? H - 6 : 9,
            "text-anchor": leftOfLine ? "end" : "start" }, mk.label));
        }
      }
      const ends = [];
      for (const se of series) {
        let d = "", pen = false, last = -1;
        se.values.forEach((v, i) => {
          if (!ok(v)) { pen = false; return; }
          d += `${pen ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`;
          pen = true; last = i;
        });
        g.append(s("path", { class: "series", d, style: `stroke:${se.color};stroke-width:${se.width || 2}px` }));
        if (endLabels && last >= 0 && se.endLabel !== false) ends.push({ x: X(xs[last]), y: Y(se.values[last]), label: endText(se) });
      }
      if (endLabels) drawEndLabels(g, ends, W, H);
      const cross = s("line", { class: "crosshair", y1: 0, y2: H, visibility: "hidden" });
      const dots = s("g");
      const hit = s("rect", { class: "hit", x: 0, y: 0, width: W, height: H });
      g.append(cross, dots, hit);
      let idx = null;
      const show = (i, cx, cy) => {
        idx = i;
        const px = X(xs[i]);
        cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
        dots.replaceChildren();
        const rows = [];
        for (const se of series) {
          const v = se.values[i];
          if (!ok(v)) continue;
          dots.append(s("circle", { class: "dot", cx: px, cy: Y(v), r: 4, style: `fill:${se.color}` }));
          rows.push({ value: yFormat(v), label: se.label, color: se.color });
        }
        const r = svg.getBoundingClientRect();
        showTip(cx ?? r.left + (px + m.left) * (r.width / width), cy ?? r.top + m.top, tipTitle(xs[i]), rows);
      };
      const hide = () => { idx = null; cross.setAttribute("visibility", "hidden"); dots.replaceChildren(); hideTip(); };
      hit.addEventListener("pointermove", (ev) => {
        const r = svg.getBoundingClientRect();
        const px = (ev.clientX - r.left) * (width / r.width) - m.left;
        show(nearestIndex(xs, X.invert(px)), ev.clientX, ev.clientY);
      });
      hit.addEventListener("pointerleave", hide);
      svg.addEventListener("blur", hide);
      svg.addEventListener("keydown", (ev) => {
        if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
          ev.preventDefault();
          const next = idx == null ? xs.length - 1 : Math.max(0, Math.min(xs.length - 1, idx + (ev.key === "ArrowRight" ? 1 : -1)));
          show(next);
        } else if (ev.key === "Escape") hide();
      });
      root.append(svg);
    }
    observe(root, draw);
    return root;
  }

  // ------------------------------------------------------------------ bar figures

  function columnChart({ labels, groups, height = 170, valueFormat = fmt.int, tipTitle = (l) => l, labelEvery = 1, refs = [] }) {
    const root = h("div", { class: "chart", style: { minHeight: `${height}px` } });
    function draw() {
      const width = Math.floor(root.clientWidth);
      if (!width) return;
      root.replaceChildren();
      const m = { top: refs.length ? 20 : 10, right: 10, bottom: 28, left: 42 };
      const W = width - m.left - m.right, H = height - m.top - m.bottom;
      const max = Math.max(1, ...groups.flatMap((gr) => gr.values));
      const yt = niceTicks(0, max, Math.max(2, Math.round(H / 38)));
      const top = Math.max(max, yt[yt.length - 1]);
      const Y = linear(0, top, H, 0);
      const band = W / labels.length;
      const inner = Math.min(band * 0.78, 24 * groups.length + 2 * (groups.length - 1));
      const bw = (inner - 2 * (groups.length - 1)) / groups.length;
      const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img" });
      const g = s("g", { transform: `translate(${m.left},${m.top})` });
      svg.append(g);
      const grid = s("g", { class: "grid" });
      g.append(grid);
      for (const t of yt) {
        grid.append(s("line", { x1: 0, x2: W, y1: Y(t), y2: Y(t) }));
        g.append(s("g", { class: "tick" }, s("text", { x: -8, y: Y(t), "text-anchor": "end", "dominant-baseline": "middle" }, fmt.compact(t))));
      }
      labels.forEach((lab, i) => {
        const x0 = i * band + (band - inner) / 2;
        groups.forEach((gr, k) => {
          const v = gr.values[i] || 0;
          const x = x0 + k * (bw + 2), y = Y(v), hgt = H - y;
          if (hgt > 0.5) {
            const r = Math.min(4, bw / 2, hgt);
            g.append(s("path", {
              d: `M${x},${H}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${H}Z`,
              style: `fill:${gr.color}`,
            }));
          }
        });
        const hit = s("rect", { class: "bar-hit", x: i * band, y: 0, width: band, height: H });
        tipFor(hit, () => [tipTitle(lab), groups.map((gr) => ({ value: valueFormat(gr.values[i] || 0), label: gr.label, color: gr.color, kind: "rect" }))]);
        g.append(hit);
        if (i % labelEvery === 0) g.append(s("g", { class: "tick" }, s("text", { x: i * band + band / 2, y: H + 18, "text-anchor": "middle" }, lab)));
      });
      for (const ref of refs) {
        const x = ref.index * band + band / 2;
        g.append(s("line", { class: "crosshair", x1: x, x2: x, y1: -6, y2: H }));
        g.append(s("text", { x: x + 5, y: -8 }, ref.label));
      }
      g.append(s("line", { class: "axis-line", x1: 0, x2: W, y1: H, y2: H }));
      root.append(svg);
    }
    observe(root, draw);
    return root;
  }

  function barList(items, { max = 1, format = (v) => fmt.num(v, 3), ref, refLabel = "all classes" } = {}) {
    const wrap = h("div", { class: "bars" });
    for (const it of items) {
      const frac = Math.max(0, Math.min(1, it.value / max));
      const track = h("div", { class: "bar-track" },
        h("div", { class: "bar-fill", style: { width: `${frac * 85}%`, background: it.color } }),
        h("span", { class: "bar-value", style: { left: `${frac * 85}%` } }, format(it.value)));
      if (ok(ref)) track.append(h("span", { class: "bar-ref", style: { left: `${(ref / max) * 85}%` } }));
      tipFor(track, () => [it.label, [{ value: format(it.value), label: it.tip || "", color: it.color, kind: "rect" },
        ...(ok(ref) ? [{ value: format(ref), label: refLabel }] : [])]]);
      wrap.append(h("div", { class: "bar-row" }, h("span", { class: "name" }, it.label), track));
    }
    return wrap;
  }

  function segBar(parts) {
    const total = parts.reduce((a, p) => a + p.value, 0) || 1;
    const bar = h("div", { class: "segbar", role: "img", "aria-label": parts.map((p) => `${p.label} ${fmt.pct(p.value / total, 1)}`).join(", ") });
    for (const p of parts) {
      const share = p.value / total;
      const seg = h("span", { style: { flex: `${share} 1 0`, background: p.color } }, share >= 0.12 ? fmt.pct(share) : "");
      tipFor(seg, () => [p.label, [{ value: fmt.int(p.value), label: `${fmt.pct(share, 1)} of cells`, color: p.color, kind: "rect" }]]);
      bar.append(seg);
    }
    return bar;
  }

  function heatmap({ rows, cols, values, shares, rowTitle = "ground truth", colTitle = "predicted", diagonalStrong = true }) {
    const root = h("div", { class: "chart" });
    function draw() {
      const width = Math.floor(root.clientWidth);
      if (!width) return;
      root.replaceChildren();
      const m = { top: 24, right: 4, bottom: 8, left: 86 };
      const cell = Math.min(78, (width - m.left - m.right) / cols.length);
      const W = cell * cols.length, H = cell * 0.62 * rows.length, ch = cell * 0.62;
      const height = H + m.top + m.bottom;
      const svg = s("svg", { viewBox: `0 0 ${W + m.left + m.right} ${height}`, height, role: "img", style: `max-width:${W + m.left + m.right}px` });
      const g = s("g", { transform: `translate(${m.left},${m.top})` });
      svg.append(g);
      cols.forEach((c, j) => g.append(s("text", { x: j * cell + cell / 2, y: -8, "text-anchor": "middle" }, c)));
      rows.forEach((r, i) => {
        g.append(s("text", { x: -8, y: i * ch + ch / 2, "text-anchor": "end", "dominant-baseline": "middle" }, r));
        cols.forEach((c, j) => {
          const v = values[i][j];
          if (v == null) return;
          const col = seqColor(shares[i][j]);
          const rect = s("rect", { x: j * cell + 1, y: i * ch + 1, width: cell - 2, height: ch - 2, rx: 4, style: `fill:${col.fill}` });
          tipFor(rect, () => [`${r} → ${c}`, [{ value: fmt.int(v), label: `${fmt.pct(shares[i][j], 1)} of the ${rowTitle} row` }]]);
          g.append(rect);
          g.append(s("text", { x: j * cell + cell / 2, y: i * ch + ch / 2, "text-anchor": "middle", "dominant-baseline": "middle",
            style: `fill:${col.text};font-weight:${diagonalStrong && i === j ? 650 : 400};pointer-events:none;font-size:12px` }, fmt.int(v)));
        });
      });
      root.append(h("div", { class: "legend" }, `Rows: ${rowTitle} · columns: ${colTitle} · shading: share of the row`), svg);
    }
    observe(root, draw);
    return root;
  }

  function legend(items) {
    return h("div", { class: "legend" }, items.map((it) => h("span", null, keyEl(it.kind || "line", it.color), it.label)));
  }

  function table(columns, rows, { totalRow } = {}) {
    const thead = h("thead", null, h("tr", null, columns.map((c) => h("th", { class: c.num ? "num" : null }, c.label))));
    const mk = (r, cls) => h("tr", { class: cls }, columns.map((c) => {
      const v = c.value(r);
      return h("td", { class: c.num ? "num" : null }, c.swatch ? h("span", { class: "swatch", style: { background: c.swatch(r) } }) : null, v);
    }));
    const tbody = h("tbody", null, rows.map((r) => mk(r)), totalRow ? mk(totalRow, "total") : null);
    return h("div", { class: "table-wrap" }, h("table", null, thead, tbody));
  }

  function card({ id, title, sub, body, tableView, tools, cls }) {
    const showTable = tableView && ui.tables.has(id);
    const content = h("div");
    content.append(showTable ? tableView() : body());
    const toolsEl = h("div", { class: "card-tools" }, tools || null);
    if (tableView) {
      const btn = h("button", { class: "btn", type: "button", "aria-pressed": String(showTable) }, showTable ? "Chart" : "Table");
      btn.addEventListener("click", () => {
        const on = !ui.tables.has(id);
        on ? ui.tables.add(id) : ui.tables.delete(id);
        btn.textContent = on ? "Chart" : "Table";
        btn.setAttribute("aria-pressed", String(on));
        content.replaceChildren(on ? tableView() : body());
      });
      toolsEl.append(btn);
    }
    return h("section", { class: `card ${cls || ""}`, id: id ? `card-${id}` : null },
      h("div", { class: "card-head" }, h("div", { class: "card-title" }, h("h2", null, title), sub ? h("p", { class: "sub" }, sub) : null), toolsEl),
      content);
  }

  function tile({ label, value, unit, note, hero, meter }) {
    return h("div", { class: `tile ${hero ? "hero" : ""}` },
      h("div", { class: "label" }, label),
      h("div", { class: "value" }, value, unit ? h("small", null, ` ${unit}`) : null),
      meter != null ? h("div", { class: "meter", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": Math.round(meter * 100) },
        h("span", { style: { width: `${Math.max(0, Math.min(1, meter)) * 100}%` } })) : null,
      note ? h("div", { class: "note" }, note) : null);
  }

  function emptyCard(title, text) {
    return h("section", { class: "card" }, h("div", { class: "empty" }, h("strong", null, title), h("span", null, text)));
  }

  function smallMultiples(items) {
    return h("div", { class: "small-multiples" }, items.map((it) => h("div", null, h("p", { class: "sm-title" }, it.title), it.chart)));
  }

  // ------------------------------------------------------------------ training view

  function runInfo(t) {
    const epochs = t.epochs || [];
    const last = epochs[epochs.length - 1];
    const bestEpoch = t.train_end?.best_epoch || last?.best_epoch;
    const best = epochs.find((e) => e.epoch === bestEpoch) || last;
    const planned = t.train_end?.epochs_planned || last?.epochs || t.train?.epochs || t.config?.train?.epochs;
    const close = t.config?.train?.close_mosaic;
    return { epochs, last, best, bestEpoch, planned, mosaicOff: close && planned ? planned - close + 1 : null };
  }

  function stageStates(t) {
    const order = STAGES.map(([k]) => k);
    const failed = t.stage === "error" || t.kaggle_status === "error";
    let current = t.stage;
    if (!t.started_at) current = "queued";
    if (failed) current = t.stages?.length ? t.stages[t.stages.length - 1].stage : "queued";
    const ci = order.indexOf(current);
    return STAGES.map(([k, label], i) => {
      let state = i < ci ? "done" : i === ci ? "active" : "todo";
      if (current === "done" && i === ci) state = "done";
      if (failed && i === ci) state = "failed";
      return { key: k, label, state };
    });
  }

  function etaSeconds(t, info) {
    const { epochs, planned } = info;
    if (!epochs.length || !planned) return null;
    const recent = epochs.slice(-5).map((e) => e.epoch_time).filter(ok);
    const per = recent.reduce((a, b) => a + b, 0) / Math.max(1, recent.length);
    return Math.max(0, planned - epochs[epochs.length - 1].epoch) * per + 120;
  }

  function renderTraining(t) {
    if (!t) return [emptyCard("No run data yet", "Push a run with `make train`, then follow it with `make monitor`.")];
    const info = runInfo(t);
    const { epochs, last, best, bestEpoch, planned } = info;
    const running = t.kaggle_status === "running";
    const out = [];

    // Status and stage stepper
    const states = stageStates(t);
    const stepper = h("div", { class: "stepper" }, states.map((st) => h("div", { class: "step", "data-state": st.state },
      icon(st.state === "done" ? "check" : st.state === "failed" ? "cross" : st.state === "active" ? "dot" : "clock"), st.label)));
    const p = t.progress;
    const statusBody = [stepper];
    let msg = t.stage_message || "";
    if (t.kaggle_status === "queued") msg = "Waiting for Kaggle to assign a GPU session";
    if (t.error) msg = `Run failed: ${t.error.message}`;
    else if (t.failure_message) msg = `Kaggle: ${t.failure_message}`;
    statusBody.push(h("div", { class: "stage-msg" }, msg));
    if (running && p && t.stage === "train") {
      const frac = p.phase === "validate" || p.phase === "epoch_done" ? 1 : (p.batch || 0) / (p.batches || 1);
      const what = p.phase === "validate" ? "validating" : p.phase === "epoch_done" ? "epoch finished" : `batch ${p.batch}/${p.batches}`;
      statusBody.push(h("div", { class: "meter", role: "progressbar", "aria-valuenow": Math.round(frac * 100), "aria-valuemin": 0, "aria-valuemax": 100 },
        h("span", { style: { width: `${frac * 100}%` } })),
      h("div", { class: "meter-row" }, h("span", null, `Epoch ${p.epoch} of ${p.epochs}: ${what}`),
        p.loss ? h("span", null, Object.entries(p.loss).map(([k, v]) => `${k.replace("_loss", "")} ${fmt.num(v, 3)}`).join(" · ")) : null));
    }
    out.push(card({ id: "status", title: "Run status", sub: t.run?.kernel ? `Kaggle kernel ${t.run.kernel} · version ${t.run.version}` : null, body: () => h("div", null, statusBody) }));

    // KPI row
    const start = t.started_at;
    const end = t.finished_at || (running ? nowS() : t.monitor?.last_event_at);
    const sinceBest = last && bestEpoch ? last.epoch - bestEpoch : null;
    const patience = last?.patience || t.config?.train?.patience;
    out.push(h("div", { class: "kpis" },
      tile({ hero: true, label: "Best validation mAP50-95", value: best ? fmt.num(best.metrics["mAP50-95"], 3) : "–",
        note: best ? `epoch ${best.epoch} · mAP50 ${fmt.num(best.metrics.mAP50, 3)}` : "after the first epoch" }),
      tile({ label: "Epoch", value: last ? String(last.epoch) : "–", unit: planned ? `/ ${planned}` : null,
        meter: last && planned ? last.epoch / planned : 0,
        note: t.train_end ? (t.train_end.early_stopped ? "stopped early" : "all epochs done") : last ? `${fmt.num(last.epoch_time, 1)} s per epoch` : null }),
      tile({ label: "Elapsed", value: fmt.dur(start ? end - start : null),
        note: running && t.stage === "train" ? `about ${fmt.dur(etaSeconds(t, info))} left` : t.finished_at ? `finished ${fmt.clock(t.finished_at)}` : null }),
      tile({ label: "Early stopping", value: sinceBest == null ? "–" : String(sinceBest), unit: patience ? `/ ${patience}` : null,
        note: "epochs without improvement", meter: sinceBest != null && patience ? sinceBest / patience : null }),
      tile({ label: "GPU", value: t.env?.gpu || "–", note: gpuNote(t, last) }),
    ));

    if (!epochs.length) {
      out.push(emptyCard("Charts appear after the first epoch",
        !running ? "The run has not reached training yet."
          : t.stage === "train" ? "Training has started; each epoch takes about 35 s on a T4."
          : `${t.stage_message || "Setting up"}; training starts next.`));
    } else {
      const xs = epochs.map((e) => e.epoch);
      const markers = bestEpoch ? [{ x: bestEpoch, label: `best ${bestEpoch}` }] : [];
      const bands = info.mosaicOff && info.mosaicOff <= xs[xs.length - 1] ? [{ from: info.mosaicOff - 0.5, to: xs[xs.length - 1] + 0.5 }] : [];
      const lossKeys = Object.keys(last.train_loss);
      const lossTable = () => table(
        [{ label: "Epoch", value: (e) => e.epoch, num: true },
          ...lossKeys.flatMap((k) => [{ label: `train ${k}`, value: (e) => fmt.num(e.train_loss[k], 4), num: true },
            { label: `val ${k}`, value: (e) => fmt.num(e.val_loss[k], 4), num: true }])],
        epochs.slice().reverse());
      out.push(card({
        id: "losses", title: "Losses",
        sub: "Training loss (gray) against validation loss (violet). The shaded band marks the final epochs without mosaic augmentation.",
        body: () => h("div", null, legend([{ label: "train", color: "var(--c-context)" }, { label: "val", color: "var(--c-accent)" }]),
          smallMultiples(lossKeys.map((k) => ({
            title: LOSS_TITLES[k] || k,
            chart: lineChart({ xs, markers, bands, label: LOSS_TITLES[k] || k, yFormat: (v) => fmt.num(v, 3),
              series: [{ label: "train", color: "var(--c-context)", values: epochs.map((e) => e.train_loss[k]) },
                { label: "val", color: "var(--c-accent)", values: epochs.map((e) => e.val_loss[k]) }] }),
          })))),
        tableView: lossTable,
      }));
      const metricKeys = ["precision", "recall", "mAP50", "mAP50-95"];
      out.push(card({
        id: "metrics", title: "Validation metrics", sub: "Evaluated on the 252 validation images after every epoch.",
        body: () => smallMultiples(metricKeys.map((k) => ({
          title: k,
          chart: lineChart({ xs, markers, label: `Validation ${k}`, endLabels: true,
            series: [{ label: k, endLabel: fmt.num(last.metrics[k], 3), color: "var(--c-accent)", values: epochs.map((e) => e.metrics[k]) }] }),
        }))),
        tableView: () => table([{ label: "Epoch", value: (e) => e.epoch, num: true },
          ...metricKeys.map((k) => ({ label: k, value: (e) => fmt.num(e.metrics[k], 4), num: true }))], epochs.slice().reverse()),
      }));
      const perClass = (k) => lineChart({
        xs, markers, label: `Per-class ${k}`, yFormat: (v) => fmt.num(v, 3),
        series: [...CLASSES.map((c) => ({ label: c, endLabel: `${c} ${fmt.num(last.per_class?.[c]?.[k], 2)}`, color: CLASS_COLOR[c], values: epochs.map((e) => e.per_class?.[c]?.[k]) })),
          { label: "all classes", endLabel: false, color: "var(--ink)", width: 1.25, values: epochs.map((e) => e.metrics[k]) }],
      });
      out.push(card({
        id: "per-class", title: "Per-class validation metrics", sub: "The thin dark line is the mean over classes.",
        body: () => h("div", null, legend([...CLASSES.map((c) => ({ label: c, color: CLASS_COLOR[c] })), { label: "all classes", color: "var(--ink)" }]),
          smallMultiples([{ title: "mAP50-95", chart: perClass("mAP50-95") }, { title: "mAP50", chart: perClass("mAP50") }])),
        tableView: () => table([{ label: "Epoch", value: (e) => e.epoch, num: true },
          ...CLASSES.flatMap((c) => ["mAP50", "mAP50-95"].map((k) => ({ label: `${c} ${k}`, value: (e) => fmt.num(e.per_class?.[c]?.[k], 4), num: true })))],
        epochs.slice().reverse()),
      }));
      out.push(h("div", { class: "row halves" },
        card({
          id: "lr", title: "Learning rate", sub: "Weight parameters; warm-up, then linear decay.",
          body: () => lineChart({ xs, label: "Learning rate", yFormat: (v) => v.toExponential(2), yTickFormat: (v) => v.toExponential(0), endLabels: false,
            series: [{ label: "learning rate", color: "var(--c-accent)", values: epochs.map((e) => e.lr?.pg0) }] }),
          tableView: () => table([{ label: "Epoch", value: (e) => e.epoch, num: true }, { label: "lr", value: (e) => (ok(e.lr?.pg0) ? e.lr.pg0.toExponential(3) : "–"), num: true }], epochs.slice().reverse()),
        }),
        card({
          id: "epoch-time", title: "Epoch duration", sub: "Training plus validation, in seconds.",
          body: () => lineChart({ xs, label: "Epoch duration", yFormat: (v) => `${fmt.num(v, 1)} s`, yTickFormat: (v) => fmt.num(v, 0), endLabels: false,
            series: [{ label: "seconds", color: "var(--c-accent)", values: epochs.map((e) => e.epoch_time) }] }),
          tableView: () => table([{ label: "Epoch", value: (e) => e.epoch, num: true }, { label: "seconds", value: (e) => fmt.num(e.epoch_time, 1), num: true }], epochs.slice().reverse()),
        })));
    }

    // Console and run details
    const consoleEl = h("pre", { class: "console", tabindex: "0" }, (t.console || []).join("\n") || "No log lines yet.");
    out.push(h("div", { class: "row wide-left" },
      card({ id: "console", title: "Kaggle log", sub: "Latest lines of the session log, streamed live.", body: () => consoleEl }),
      card({ id: "details", title: "Run details", body: () => runDetails(t) })));
    requestAnimationFrame(() => {
      if (ui.consolePinned !== false) consoleEl.scrollTop = consoleEl.scrollHeight;
      consoleEl.addEventListener("scroll", () => {
        ui.consolePinned = consoleEl.scrollTop + consoleEl.clientHeight >= consoleEl.scrollHeight - 8;
      });
    });
    return out;
  }

  function gpuNote(t, last) {
    const used = t.progress?.gpu_mem_gb || last?.gpu_mem_gb;
    const total = t.env?.gpu_mem_gb;
    if (ok(used) && total) return `${fmt.num(used, 1)} of ${total} GB reserved`;
    return total ? `${total} GB` : null;
  }

  function runDetails(t) {
    const c = t.config || {}, tr = c.train || {}, env = t.env || {}, train = t.train || {};
    const rows = [
      ["Model", c.model ? `${c.model}${train.params ? ` · ${fmt.num(train.params / 1e6, 1)} M params · ${fmt.num(train.gflops, 0)} GFLOPs` : ""}` : "–"],
      ["Schedule", tr.epochs ? `${tr.epochs} epochs, patience ${tr.patience}, batch ${tr.batch}, ${tr.imgsz} px` : "–"],
      ["Optimizer", tr.optimizer ? `${tr.optimizer}, lr0 ${tr.lr0}, final lr ×${tr.lrf}, warm-up ${tr.warmup_epochs} epochs` : "–"],
      ["Augmentation", tr.mosaic != null ? `mosaic ${tr.mosaic} (off last ${tr.close_mosaic}), flips ${tr.fliplr}/${tr.flipud}, scale ${tr.scale}` : "–"],
      ["Hardware", env.gpu ? `${env.gpu} (${env.gpu_mem_gb} GB), ${env.cpu_count} vCPUs` : "–"],
      ["Software", env.torch ? `PyTorch ${env.torch}, CUDA ${env.cuda}, Ultralytics ${env.ultralytics}, Python ${env.python}` : "–"],
      ["Pushed", fmt.date(t.run?.pushed_at ? Date.parse(t.run.pushed_at) / 1000 : null)],
      ["Started", fmt.date(t.started_at)],
    ];
    const dl = h("dl", { class: "kv" }, rows.map(([k, v]) => [h("dt", null, k), h("dd", null, v)]));
    if (t.run?.url) dl.append(h("dt", null, "Kernel"), h("dd", null, h("a", { href: t.run.url, target: "_blank", rel: "noopener" }, t.run.kernel)));
    return dl;
  }

  // ------------------------------------------------------------------ performance view

  function boxIoU(a, b) {
    const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = iw * ih;
    const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return union > 0 ? inter / union : 0;
  }

  /* Same greedy matching as scripts/build_reports.py: highest confidence first, best unclaimed box. */
  function greedyMatch(pred, gt, iouThr, classAware) {
    const order = pred.map((_, i) => i).sort((i, j) => pred[j][1] - pred[i][1]);
    const claimed = new Array(gt.length).fill(false);
    const pairs = [], fps = [];
    for (const i of order) {
      let best = -1, bj = -1;
      for (let j = 0; j < gt.length; j++) {
        if (claimed[j] || (classAware && gt[j][0] !== pred[i][0])) continue;
        const v = boxIoU(pred[i].slice(2), gt[j].slice(1));
        if (v > best) { best = v; bj = j; }
      }
      if (bj >= 0 && best >= iouThr) { claimed[bj] = true; pairs.push([i, bj]); } else fps.push(i);
    }
    const fns = gt.map((_, j) => j).filter((j) => !claimed[j]);
    return { pairs, fps, fns };
  }

  const thresholdCache = new Map();
  function analyzeAt(ev, conf) {
    const key = String(conf);
    if (thresholdCache.has(key)) return thresholdCache.get(key);
    const nc = CLASSES.length;
    const tp = Array(nc).fill(0), fp = Array(nc).fill(0), fn = Array(nc).fill(0);
    const cm = Array.from({ length: nc + 1 }, () => Array(nc + 1).fill(0));
    const images = ev.explorer.map((im) => {
      const pred = im.pred.filter((b) => b[1] >= conf);
      const m = greedyMatch(pred, im.gt, ev.iou_match, true);
      m.pairs.forEach(([, j]) => tp[im.gt[j][0]]++);
      m.fps.forEach((i) => fp[pred[i][0]]++);
      m.fns.forEach((j) => fn[im.gt[j][0]]++);
      const a = greedyMatch(pred, im.gt, ev.iou_match, false);
      a.pairs.forEach(([i, j]) => cm[im.gt[j][0]][pred[i][0]]++);
      a.fps.forEach((i) => cm[nc][pred[i][0]]++);
      a.fns.forEach((j) => cm[im.gt[j][0]][nc]++);
      const count = (boxes, k) => boxes.filter((b) => b[0] === k).length;
      return { name: im.name, match: m, pred, errors: m.fps.length + m.fns.length,
        gtCount: CLASSES.map((_, k) => count(im.gt, k)), predCount: CLASSES.map((_, k) => count(pred, k)) };
    });
    const perClass = CLASSES.map((c, k) => {
      const pr = tp[k] + fp[k] ? tp[k] / (tp[k] + fp[k]) : 0, rc = tp[k] + fn[k] ? tp[k] / (tp[k] + fn[k]) : 0;
      const g = images.map((im) => im.gtCount[k]), q = images.map((im) => im.predCount[k]);
      const err = g.map((v, i) => q[i] - v);
      const rel = g.flatMap((v, i) => (v > 0 ? [Math.abs(err[i]) / v] : []));
      return { class: c, tp: tp[k], fp: fp[k], fn: fn[k], precision: pr, recall: rc, f1: pr + rc ? (2 * pr * rc) / (pr + rc) : 0,
        trueTotal: g.reduce((a, b) => a + b, 0), predTotal: q.reduce((a, b) => a + b, 0),
        mae: err.reduce((a, e) => a + Math.abs(e), 0) / err.length, bias: err.reduce((a, e) => a + e, 0) / err.length,
        exact: err.filter((e) => e === 0).length / err.length, mape: rel.length ? rel.reduce((a, b) => a + b, 0) / rel.length : null };
    });
    const res = { conf, images, perClass, cm };
    thresholdCache.set(key, res);
    return res;
  }

  function renderPerformance(ev, t) {
    if (!ev) {
      const info = t ? runInfo(t) : null;
      const text = t?.kaggle_status === "running" ?
        `Training is at epoch ${info?.last?.epoch ?? 0} of ${info?.planned ?? "?"}. Test-set results, curves and the prediction explorer appear here when the run finishes.` :
        "Run `make reports` after a finished run to build the evaluation data.";
      return [emptyCard("Evaluation not available yet", text)];
    }
    if (ui.conf == null) ui.conf = ev.operating_conf;
    const split = ev.splits[ui.split] ? ui.split : "test";
    const se = ev.splits[split];
    const o = se.overall;
    const out = [];

    const splitControl = h("div", { class: "controls" },
      h("span", { class: "control" }, "Split for the metric summary",
        h("span", { class: "segmented" }, ["test", "val"].map((sp) => h("button", { type: "button", "aria-pressed": String(sp === split),
          onclick: () => { ui.split = sp; rerender(); } }, sp)))),
      h("span", { class: "badge" }, icon("check"), `${ev.run?.version ? `Kaggle run v${ev.run.version} · ` : ""}best.pt from epoch ` +
        `${ev.best_epoch ?? t?.train_end?.best_epoch ?? "?"} · ${ev.model}`));
    out.push(splitControl);

    out.push(h("div", { class: "kpis" },
      tile({ hero: true, label: `${split} mAP50-95`, value: fmt.num(o["mAP50-95"], 3), note: `averaged over IoU 0.50–0.95 · ${CLASSES.length} classes` }),
      tile({ label: `${split} mAP50`, value: fmt.num(o.mAP50, 3), note: `mAP75 ${fmt.num(o.mAP75, 3)}` }),
      tile({ label: "Precision", value: fmt.num(o.precision, 3), note: "at the F1-optimal confidence" }),
      tile({ label: "Recall", value: fmt.num(o.recall, 3), note: "at the F1-optimal confidence" }),
      tile({ label: "Inference", value: fmt.num(se.speed_ms?.inference, 1), unit: "ms/img",
        note: `${ev.environment?.gpu || "GPU"} · +${fmt.num((se.speed_ms?.preprocess || 0) + (se.speed_ms?.postprocess || 0), 1)} ms pre/post` }),
    ));

    const rows = se.per_class;
    out.push(h("div", { class: "row halves" },
      card({
        id: "per-class-metrics", title: `Per-class metrics (${split})`, sub: "Bars per class; the dark tick is the mean over classes.",
        body: () => smallMultiples(["precision", "recall", "mAP50", "mAP50-95"].map((k) => ({
          title: k, chart: barList(rows.map((r) => ({ label: r.class, value: r[k], color: CLASS_COLOR[r.class], tip: k })), { ref: o[k] }),
        }))),
        tableView: () => table([
          { label: "Class", value: (r) => r.class, swatch: (r) => CLASS_COLOR[r.class] },
          { label: "Images", value: (r) => fmt.int(r.images), num: true }, { label: "Cells", value: (r) => fmt.int(r.instances), num: true },
          ...["precision", "recall", "f1", "mAP50", "mAP75", "mAP50-95"].map((k) => ({ label: k, value: (r) => fmt.num(r[k], 3), num: true }))], rows),
      }),
      se.curves ? card({
        id: "pr-curves", title: `Precision-recall curves (${split})`, sub: "IoU 0.5. The legend gives AP50 per class.",
        body: () => h("div", null, legend(rows.map((r) => ({ label: `${r.class} · AP50 ${fmt.num(r.mAP50, 3)}`, color: CLASS_COLOR[r.class] }))),
          lineChart({ xs: se.curves.x, height: 230, yDomain: [0, 1.02], label: "Precision-recall", endLabels: false, tipTitle: (x) => `Recall ${fmt.num(x, 2)}`,
            xFormat: (v) => fmt.num(v, 1), yFormat: (v) => fmt.num(v, 3), yTickFormat: (v) => fmt.num(v, 1),
            series: CLASSES.map((c) => ({ label: c, color: CLASS_COLOR[c], values: se.curves.precision_recall[c] })) })),
        tableView: () => table([{ label: "Recall", value: (r) => fmt.num(r.x, 2), num: true },
          ...CLASSES.map((c) => ({ label: `${c} precision`, value: (r) => fmt.num(r[c], 3), num: true }))],
        se.curves.x.map((x, i) => ({ x, ...Object.fromEntries(CLASSES.map((c) => [c, se.curves.precision_recall[c][i]])) })).filter((_, i) => i % 5 === 0)),
      }) : emptyCard("No curves for this split", "")));

    // Operating threshold section (test split, interactive)
    out.push(h("h3", { class: "section-title" }, "Operating threshold · test split"));
    const thresholdHost = h("div", { class: "view" });
    out.push(thresholdHost);
    renderThresholdSection(thresholdHost, ev);

    if (ev.splits.test_nms_free) {
      const e2e = ev.splits.test_nms_free, d = ev.splits.test;
      const comp = [
        { head: "One-to-many head + NMS (default)", ...d.overall, ms: (d.speed_ms?.inference || 0) + (d.speed_ms?.postprocess || 0) },
        { head: "One-to-one head, NMS-free", ...e2e.overall, ms: (e2e.speed_ms?.inference || 0) + (e2e.speed_ms?.postprocess || 0) },
      ];
      out.push(card({
        id: "nms", title: "NMS-free head (test)", sub: "YOLO26 can skip non-maximum suppression with its one-to-one head (nms=False).",
        body: () => table([{ label: "Head", value: (r) => r.head }, ...["precision", "recall", "mAP50", "mAP50-95"].map((k) => ({ label: k, value: (r) => fmt.num(r[k], 3), num: true })),
          { label: "inference + post (ms/img)", value: (r) => fmt.num(r.ms, 1), num: true }], comp),
      }));
    }
    return out;
  }

  function renderThresholdSection(host, ev) {
    const res = analyzeAt(ev, ui.conf);
    const slider = h("input", { type: "range", min: Math.max(0.05, ev.conf_min), max: 0.95, step: 0.01, value: ui.conf, "aria-label": "Confidence threshold" });
    const output = h("output", null, fmt.num(ui.conf, 2));
    slider.addEventListener("input", () => {
      output.textContent = fmt.num(+slider.value, 2);
      clearTimeout(slider._t);
      slider._t = setTimeout(() => { ui.conf = +slider.value; renderThresholdSection(host, ev); }, 60);
    });
    const reset = h("button", { class: "btn", type: "button", onclick: () => { ui.conf = ev.operating_conf; renderThresholdSection(host, ev); } },
      `Reset to ${fmt.num(ev.operating_conf, 2)}`);
    const controls = h("div", { class: "controls" },
      h("label", { class: "control" }, "Confidence threshold", slider, output), reset,
      h("span", { class: "badge" }, `${fmt.num(ev.operating_conf, 2)} maximizes F1 on the validation split · match IoU ≥ ${ev.iou_match}`));

    const nc = CLASSES.length;
    const labelsRows = [...CLASSES, "no cell"], labelsCols = [...CLASSES, "missed"];
    const values = res.cm.map((r, i) => r.map((v, j) => (i === nc && j === nc ? null : v)));
    const shares = res.cm.map((r) => { const tot = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => v / tot); });
    const pc = res.perClass;
    const f1Curve = ev.splits.test.curves;
    const children = [controls,
      h("div", { class: "row halves" },
        card({
          id: "confusion", title: "Confusion matrix", sub: "Class-agnostic matching. ‘No cell’ = detection without a matching annotation; ‘missed’ = annotated cell without a detection.",
          body: () => heatmap({ rows: labelsRows, cols: labelsCols, values, shares }),
          tableView: () => table([{ label: "ground truth \\ predicted", value: (r) => r.label }, ...labelsCols.map((c, j) => ({ label: c, value: (r) => (r.values[j] == null ? "–" : fmt.int(r.values[j])), num: true }))],
            labelsRows.map((label, i) => ({ label, values: values[i] }))),
        }),
        card({
          id: "f1-conf", title: "F1 against confidence (test)", sub: "The vertical line is the threshold chosen above.",
          body: () => h("div", null, legend(CLASSES.map((c) => ({ label: c, color: CLASS_COLOR[c] }))),
            lineChart({ xs: f1Curve.x, height: 230, yDomain: [0, 1.02], endLabels: false, label: "F1 against confidence",
              markers: [{ x: ui.conf, label: `threshold ${fmt.num(ui.conf, 2)}`, at: "bottom" }], tipTitle: (x) => `Confidence ${fmt.num(x, 2)}`,
              xFormat: (v) => fmt.num(v, 1), yTickFormat: (v) => fmt.num(v, 1),
              series: CLASSES.map((c) => ({ label: c, color: CLASS_COLOR[c], values: f1Curve.f1_confidence[c] })) })),
          tableView: () => table([{ label: "Confidence", value: (r) => fmt.num(r.x, 2), num: true }, ...CLASSES.map((c) => ({ label: `${c} F1`, value: (r) => fmt.num(r[c], 3), num: true }))],
            f1Curve.x.map((x, i) => ({ x, ...Object.fromEntries(CLASSES.map((c) => [c, f1Curve.f1_confidence[c][i]])) })).filter((_, i) => i % 5 === 0)),
        })),
      h("div", { class: "row halves" },
        card({
          id: "detections", title: "Detections at this threshold",
          body: () => table([
            { label: "Class", value: (r) => r.class, swatch: (r) => CLASS_COLOR[r.class] },
            { label: "Correct", value: (r) => fmt.int(r.tp), num: true }, { label: "False", value: (r) => fmt.int(r.fp), num: true },
            { label: "Missed", value: (r) => fmt.int(r.fn), num: true }, { label: "Precision", value: (r) => fmt.num(r.precision, 3), num: true },
            { label: "Recall", value: (r) => fmt.num(r.recall, 3), num: true }, { label: "F1", value: (r) => fmt.num(r.f1, 3), num: true }], pc),
        }),
        card({
          id: "counting", title: "Cell counts per image", sub: `Detected against annotated, over ${res.images.length} test images.`,
          body: () => table([
            { label: "Class", value: (r) => r.class, swatch: (r) => CLASS_COLOR[r.class] },
            { label: "Annotated", value: (r) => fmt.int(r.trueTotal), num: true }, { label: "Detected", value: (r) => fmt.int(r.predTotal), num: true },
            { label: "MAE", value: (r) => fmt.num(r.mae, 2), num: true }, { label: "Bias", value: (r) => fmt.signed(r.bias, 2), num: true },
            { label: "Relative error", value: (r) => fmt.pct(r.mape, 1), num: true }, { label: "Exact", value: (r) => fmt.pct(r.exact), num: true }], pc),
        })),
      explorerCard(ev, res)];
    host.replaceChildren(...children);
    const list = host.querySelector(".explorer-list");
    if (list) {
      list.scrollTop = ui.listScroll;
      const current = list.querySelector('[aria-current="true"]');
      if (current && (current.offsetTop < list.scrollTop || current.offsetTop + current.offsetHeight > list.scrollTop + list.clientHeight)) {
        list.scrollTop = current.offsetTop - list.clientHeight / 2;
      }
      list.addEventListener("scroll", () => { ui.listScroll = list.scrollTop; });
    }
  }

  function imageFigure(src, w, ht, boxes, alt) {
    const img = h("img", { src, alt, loading: "lazy", width: w, height: ht });
    const fig = h("div", { class: "figure-img" }, img);
    img.addEventListener("error", () => fig.append(h("div", { class: "missing" }, "Image not available: put the TXL-PBC dataset in data/TXL-PBC (see the README).")));
    const svg = s("svg", { viewBox: `0 0 ${w} ${ht}`, preserveAspectRatio: "none" });
    for (const b of boxes) {
      const r = s("rect", { x: b.x1 * w, y: b.y1 * ht, width: (b.x2 - b.x1) * w, height: (b.y2 - b.y1) * ht,
        style: `stroke:${b.color};stroke-width:${b.width || 2}px${b.dash ? ";stroke-dasharray:4 3" : ""}` });
      svg.append(r);
    }
    fig.append(svg);
    return fig;
  }

  function explorerCard(ev, res) {
    const order = res.images.map((im, i) => i);
    if (ui.explorerSort === "errors") order.sort((a, b) => res.images[b].errors - res.images[a].errors || a - b);
    if (ui.explorer == null || ui.explorer >= ev.explorer.length) ui.explorer = order[0];
    const im = ev.explorer[ui.explorer], r = res.images[ui.explorer];
    const pos = order.indexOf(ui.explorer);
    const go = (delta) => { ui.explorer = order[(pos + delta + order.length) % order.length]; rerenderThreshold(); };

    const list = h("div", { class: "explorer-list", role: "listbox", "aria-label": "Test images" }, order.map((i) => {
      const x = res.images[i];
      return h("button", { type: "button", "aria-current": String(i === ui.explorer), onclick: () => { ui.explorer = i; rerenderThreshold(); } },
        h("span", null, x.name.slice(0, 12)), h("span", { class: `errs ${x.errors ? "has" : ""}` }, x.errors ? `${x.errors} error${x.errors > 1 ? "s" : ""}` : "no errors"));
    }));

    const boxes = [];
    if (ui.explorerMode === "errors") {
      r.match.pairs.forEach(([i]) => { const b = r.pred[i]; boxes.push({ x1: b[2], y1: b[3], x2: b[4], y2: b[5], color: "var(--s-good)" }); });
      r.match.fns.forEach((j) => { const b = im.gt[j]; boxes.push({ x1: b[1], y1: b[2], x2: b[3], y2: b[4], color: "var(--s-warning)", width: 3 }); });
      r.match.fps.forEach((i) => { const b = r.pred[i]; boxes.push({ x1: b[2], y1: b[3], x2: b[4], y2: b[5], color: "var(--s-critical)", width: 3 }); });
    } else if (ui.explorerMode === "pred") {
      r.pred.forEach((b) => boxes.push({ x1: b[2], y1: b[3], x2: b[4], y2: b[5], color: CLASS_COLOR[CLASSES[b[0]]] }));
    } else {
      im.gt.forEach((b) => boxes.push({ x1: b[1], y1: b[2], x2: b[3], y2: b[4], color: CLASS_COLOR[CLASSES[b[0]]] }));
    }
    const modes = [["errors", "Errors"], ["pred", "Predictions"], ["gt", "Ground truth"]];
    const modeControl = h("span", { class: "segmented" }, modes.map(([k, label]) => h("button", { type: "button", "aria-pressed": String(ui.explorerMode === k),
      onclick: () => { ui.explorerMode = k; rerenderThreshold(); } }, label)));
    const sortControl = h("span", { class: "segmented" }, [["errors", "Most errors"], ["name", "Name"]].map(([k, label]) => h("button", { type: "button",
      "aria-pressed": String(ui.explorerSort === k), onclick: () => { ui.explorerSort = k; ui.explorer = null; ui.listScroll = 0; rerenderThreshold(); } }, label)));
    const key = ui.explorerMode === "errors" ?
      legend([{ label: "✓ correct", color: "var(--s-good)", kind: "box" }, { label: "✗ false detection", color: "var(--s-critical)", kind: "box" }, { label: "! missed cell", color: "var(--s-warning)", kind: "box" }]) :
      legend(CLASSES.map((c) => ({ label: c, color: CLASS_COLOR[c], kind: "box" })));
    const stage = h("div", { class: "explorer-stage" }, key, imageFigure(im.image, im.width, im.height, boxes, `Test image ${im.name}`),
      h("div", { class: "explorer-nav" },
        h("button", { class: "btn", type: "button", onclick: () => go(-1) }, "← Previous"),
        h("span", null, `${pos + 1} of ${order.length} · ${im.width}×${im.height}`),
        h("button", { class: "btn", type: "button", onclick: () => go(1) }, "Next →")));
    const side = h("div", null,
      table([{ label: "Class", value: (x) => x.c, swatch: (x) => CLASS_COLOR[x.c] }, { label: "Annotated", value: (x) => x.g, num: true }, { label: "Detected", value: (x) => x.p, num: true }],
        CLASSES.map((c, k) => ({ c, g: r.gtCount[k], p: r.predCount[k] }))),
      h("dl", { class: "kv", style: { marginTop: "14px" } },
        h("dt", null, "Correct"), h("dd", null, String(r.match.pairs.length)),
        h("dt", null, "False detections"), h("dd", null, String(r.match.fps.length)),
        h("dt", null, "Missed cells"), h("dd", null, String(r.match.fns.length)),
        h("dt", null, "Image"), h("dd", null, im.name)));
    return card({
      id: "explorer", title: "Prediction explorer",
      sub: `All ${ev.explorer.length} test images at the threshold above. Use the arrow keys to step through images.`,
      tools: [h("span", { class: "control" }, "Show", modeControl), h("span", { class: "control" }, "Sort", sortControl)],
      body: () => {
        const wrap = h("div", { class: "explorer", tabindex: "0" }, list, stage, side);
        wrap.addEventListener("keydown", (e) => {
          if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); go(1); }
          if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); go(-1); }
        });
        return wrap;
      },
    });
  }

  function rerenderThreshold() {
    const host = document.querySelector("#card-explorer")?.parentElement;
    if (host && DATA.evaluation) {
      const focused = document.activeElement?.closest?.(".explorer") != null;
      renderThresholdSection(host, DATA.evaluation);
      if (focused) document.querySelector(".explorer")?.focus();
    }
  }

  // ------------------------------------------------------------------ dataset view

  function renderDataset(d) {
    if (!d) return [emptyCard("Dataset statistics not built yet", "Run `make dataset-report`.")];
    const totals = Object.fromEntries(CLASSES.map((c) => [c, d.splits.reduce((a, sp) => a + d.per_split[sp].cells[c], 0)]));
    const nCells = Object.values(totals).reduce((a, b) => a + b, 0);
    const nImages = d.splits.reduce((a, sp) => a + d.per_split[sp].images, 0);
    const out = [];
    out.push(h("div", { class: "kpis" },
      tile({ hero: true, label: "Annotated cells", value: fmt.int(nCells), note: `${fmt.num(nCells / nImages, 1)} per image on average` }),
      tile({ label: "Images", value: fmt.int(nImages), note: d.splits.map((sp) => `${d.per_split[sp].images} ${sp}`).join(" · ") }),
      tile({ label: "Classes", value: String(CLASSES.length), note: CLASSES.join(", ") }),
      tile({ label: "Image sources", value: String(d.groups.length), note: "distinct resolutions" }),
      tile({ label: "Quality checks", value: `${d.checks.filter((c) => c.status === "pass").length} / ${d.checks.length}`, note: "passed, see below" })));

    out.push(h("div", { class: "row halves" },
      card({
        id: "ds-classes", title: "Class balance", sub: "Annotated cells per class over all splits, and the class share within each split.",
        body: () => h("div", null,
          barList(CLASSES.map((c) => ({ label: c, value: totals[c], color: CLASS_COLOR[c], tip: `${fmt.pct(totals[c] / nCells, 1)} of all cells` })),
            { max: Math.max(...Object.values(totals)), format: (v) => `${fmt.int(v)} · ${fmt.pct(v / nCells, 1)}` }),
          h("div", { style: { height: "16px" } }),
          legend(CLASSES.map((c) => ({ label: c, color: CLASS_COLOR[c], kind: "rect" }))),
          d.splits.map((sp) => h("div", { class: "seg-row" }, h("span", null, `${sp} (${d.per_split[sp].images} images)`),
            segBar(CLASSES.map((c) => ({ label: c, value: d.per_split[sp].cells[c], color: CLASS_COLOR[c] }))),
            h("span", { class: "total" }, `${fmt.int(d.per_split[sp].cells_total)} cells`)))),
        tableView: () => table([{ label: "Split", value: (r) => r.sp }, { label: "Images", value: (r) => fmt.int(d.per_split[r.sp].images), num: true },
          ...CLASSES.map((c) => ({ label: c, value: (r) => fmt.int(d.per_split[r.sp].cells[c]), num: true })),
          { label: "Cells", value: (r) => fmt.int(d.per_split[r.sp].cells_total), num: true }], d.splits.map((sp) => ({ sp }))),
      }),
      card({
        id: "ds-per-image", title: "Cells per image", sub: "Number of annotated cells in each image, all splits.",
        body: () => {
          const counts = d.cells_hist.train.map((_, i) => d.splits.reduce((a, sp) => a + (d.cells_hist[sp][i] || 0), 0));
          const mean = counts.reduce((a, v, i) => a + v * i, 0) / counts.reduce((a, v) => a + v, 0);
          return columnChart({ labels: counts.map((_, i) => String(i)), labelEvery: 5, tipTitle: (l) => `${l} cells in the image`,
            groups: [{ label: "images", color: "var(--c-accent)", values: counts }], refs: [{ index: mean, label: `mean ${fmt.num(mean, 1)}` }] });
        },
        tableView: () => table([{ label: "Cells in image", value: (r) => r.i, num: true }, ...d.splits.map((sp) => ({ label: `${sp} images`, value: (r) => fmt.int(d.cells_hist[sp][r.i] || 0), num: true }))],
          d.cells_hist.train.map((_, i) => ({ i })).filter((r) => d.splits.some((sp) => d.cells_hist[sp][r.i]))),
      })));

    const mids = d.size_bins.slice(0, -1).map((e, i) => Math.sqrt(e * d.size_bins[i + 1]));
    const logX = mids.map((v) => Math.log10(v));
    out.push(h("div", { class: "row halves" },
      card({
        id: "ds-sizes", title: "Cell size by class", sub: `Box size √(w·h) after resizing to ${d.model_imgsz} px, as the model sees it. Log scale.`,
        body: () => h("div", null, legend(CLASSES.map((c) => ({ label: `${c} · median ${fmt.num(d.size_summary[c].median, 0)} px`, color: CLASS_COLOR[c] }))),
          lineChart({ xs: logX, height: 220, endLabels: false, label: "Cell size distribution",
            xTicks: [10, 20, 50, 100, 200, 400].map(Math.log10), xFormat: (v) => fmt.num(10 ** v, 0),
            tipTitle: (x) => `≈ ${fmt.num(10 ** x, 0)} px`, yFormat: (v) => fmt.pct(v, 1), yTickFormat: (v) => fmt.pct(v),
            series: CLASSES.map((c) => { const hs = d.size_hist[c], tot = hs.reduce((a, b) => a + b, 0) || 1; return { label: c, color: CLASS_COLOR[c], values: hs.map((v) => v / tot) }; }) })),
        tableView: () => table([{ label: "Class", value: (r) => r, swatch: (r) => CLASS_COLOR[r] },
          { label: "Median px", value: (r) => fmt.num(d.size_summary[r].median, 0), num: true },
          { label: "5th–95th pct", value: (r) => `${fmt.num(d.size_summary[r].p5, 0)}–${fmt.num(d.size_summary[r].p95, 0)}`, num: true },
          { label: "Below 32 px", value: (r) => fmt.pct(d.size_summary[r].small_share), num: true },
          { label: "Native median px", value: (r) => fmt.num(d.size_summary[r].native_median, 0), num: true }], CLASSES),
      }),
      card({
        id: "ds-wbc-plt", title: "WBC and platelets per image", sub: "Most images hold exactly one white cell; most hold no platelet.",
        body: () => {
          const labels = ["0", "1", "2", "3", "4", "5+"];
          const bucket = (hist) => labels.map((_, i) => (i < 5 ? hist[i] || 0 : hist.slice(5).reduce((a, b) => a + b, 0)));
          return h("div", null, legend(["WBC", "Platelets"].map((c) => ({ label: c, color: CLASS_COLOR[c], kind: "rect" }))),
            columnChart({ labels, tipTitle: (l) => `${l} of the class in the image`,
              groups: ["WBC", "Platelets"].map((c) => ({ label: c, color: CLASS_COLOR[c], values: bucket(d.per_class_per_image[c].hist) })) }));
        },
        tableView: () => table([{ label: "Class", value: (r) => r, swatch: (r) => CLASS_COLOR[r] },
          { label: "Images with ≥1", value: (r) => fmt.int(d.per_class_per_image[r].images_with), num: true },
          { label: "Mean per image", value: (r) => fmt.num(d.per_class_per_image[r].mean, 2), num: true }], CLASSES),
      })));

    out.push(h("div", { class: "row halves" },
      card({
        id: "ds-sources", title: "Image sources", sub: `Resolution identifies the source collection. Smaller images are upscaled to ${d.model_imgsz} px.`,
        body: () => table([
          { label: "Resolution", value: (g) => g.resolution }, { label: "Images", value: (g) => fmt.int(g.images_total), num: true },
          { label: "train / val / test", value: (g) => `${g.images.train} / ${g.images.val} / ${g.images.test}`, num: true },
          { label: "Scale", value: (g) => `×${g.scale}`, num: true }, { label: "Cells / image", value: (g) => fmt.num(g.cells_per_image, 1), num: true },
          { label: "Median WBC / RBC / PLT px", value: (g) => CLASSES.map((c) => (g.median_size_model_px[c] == null ? "–" : fmt.num(g.median_size_model_px[c], 0))).join(" / "), num: true }], d.groups),
      }),
      card({
        id: "ds-checks", title: "Data quality checks",
        body: () => h("ul", { class: "checks" }, d.checks.map((c) => h("li", { "data-status": c.status },
          icon(c.status === "pass" ? "check" : c.status === "warn" ? "warn" : "cross"),
          h("div", null, h("div", null, h("span", { class: "verdict" }, c.status === "pass" ? "Pass" : c.status === "warn" ? "Warning" : "Fail"), c.label),
            h("div", { class: "detail" }, c.detail))))),
      })));

    out.push(card({
      id: "ds-samples", title: "Sample images", sub: "Random images from every source with their ground-truth boxes.",
      body: () => h("div", null, legend(CLASSES.map((c) => ({ label: c, color: CLASS_COLOR[c], kind: "box" }))),
        h("div", { class: "gallery" }, d.samples.map((smp) => h("figure", { style: { margin: 0 } },
          imageFigure(smp.image, smp.width, smp.height, smp.boxes.map((b) => ({ x1: b[1], y1: b[2], x2: b[3], y2: b[4], color: CLASS_COLOR[CLASSES[b[0]]] })), `${smp.split} image ${smp.name}`),
          h("figcaption", null, h("span", null, `${smp.group} · ${smp.split}`), h("span", null, `${smp.boxes.length} cells`)))))),
    }));
    return out;
  }

  // ------------------------------------------------------------------ app shell

  const view = document.getElementById("view");
  const liveRegion = document.getElementById("live-region");

  function headerStatus() {
    const t = DATA.training;
    const pill = document.getElementById("status-pill");
    const text = pill.lastElementChild;
    let tone = "neutral", label = "No run";
    if (t) {
      const st = t.kaggle_status, phase = t.monitor?.phase;
      const info = runInfo(t);
      if (st === "queued") { tone = "warning"; label = "Queued on Kaggle"; }
      else if (st === "running") { tone = "running"; label = t.stage === "train" && info.last ? `Training · epoch ${info.last.epoch}/${info.planned}` : `Running · ${t.stage}`; }
      else if (st === "complete" && phase === "finished") { tone = "good"; label = "✓ Complete"; }
      else if (st === "complete") { tone = "running"; label = phase === "downloading" ? "Downloading outputs" : "Building reports"; }
      else if (st === "error" || phase === "failed") { tone = "critical"; label = "✗ Failed"; }
      else if (st === "cancel_acknowledged") { tone = "critical"; label = "Cancelled"; }
      else { label = st || "Unknown"; }
    }
    if (text.textContent !== label) { text.textContent = label; liveRegion.textContent = `Run status: ${label}`; }
    pill.dataset.tone = tone;
    const link = document.getElementById("kaggle-link");
    if (t?.run?.url) { link.href = t.run.url; link.hidden = false; }
    updateClock();
  }

  function updateClock() {
    const t = DATA.training, el = document.getElementById("updated");
    if (!t?.monitor?.updated) { el.textContent = ""; return; }
    const age = nowS() - Date.parse(t.monitor.updated) / 1000;
    const live = ["queued", "running"].includes(t.kaggle_status);
    if (live && age > STALE_AFTER_S) el.textContent = `⚠ last update ${fmt.dur(age)} ago: is make monitor running?`;
    else el.textContent = `${live ? "Live · " : ""}updated ${age < 60 ? `${Math.max(0, Math.round(age))} s` : fmt.dur(age)} ago`;
  }

  function keyFor(tab) {
    if (tab === "training") return DATA.training?.monitor?.updated || "none";
    if (tab === "performance") return `${DATA.evaluation?.generated || "none"}|${ui.split}|${DATA.training?.monitor?.updated && !DATA.evaluation ? DATA.training.monitor.updated : ""}`;
    return DATA.dataset?.generated || "none";
  }

  function render(force = false) {
    const key = keyFor(ui.tab);
    if (!force && ui.rendered[ui.tab] === key) return;
    ui.rendered = { [ui.tab]: key };
    const content = ui.tab === "training" ? renderTraining(DATA.training) :
      ui.tab === "performance" ? renderPerformance(DATA.evaluation, DATA.training) : renderDataset(DATA.dataset);
    hideTip();
    view.replaceChildren(...content);
  }

  function rerender() { render(true); }

  function selectTab(tab, push = true) {
    if (!["training", "performance", "dataset"].includes(tab)) tab = "training";
    ui.tab = tab;
    for (const b of document.querySelectorAll("nav.tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
    if (push) history.replaceState(null, "", `#${tab}`);
    render(true);
  }

  function loadData(name) {
    return new Promise((resolve) => {
      const el = document.createElement("script");
      el.src = `data/${name}.js?t=${Date.now()}`;
      el.onload = () => { el.remove(); resolve(true); };
      el.onerror = () => { el.remove(); resolve(false); };
      document.head.append(el);
    });
  }

  let deferred = 0;
  async function tick() {
    await loadData("training");
    const t = DATA.training;
    const finished = t && ["complete", "error", "cancel_acknowledged"].includes(t.kaggle_status) && ["finished", "failed"].includes(t.monitor?.phase);
    // Optional files are retried only when they can have changed, so a missing file does not log every tick.
    if (finished || !ui.triedEvaluation) { ui.triedEvaluation = true; await loadData("evaluation"); }
    if (!DATA.dataset && !ui.triedDataset) { ui.triedDataset = true; await loadData("dataset"); }
    headerStatus();
    if (ui.hovering && deferred < 6) deferred++;
    else { deferred = 0; render(); }
    setTimeout(tick, finished && DATA.evaluation ? POLL_IDLE_MS : POLL_LIVE_MS);
  }

  // Theme toggle: auto -> light -> dark -> auto. Charts that compute colors re-render on change.
  const themeBtn = document.getElementById("theme-btn");
  function applyTheme(theme) {
    if (theme) document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
    try { theme ? localStorage.setItem("bc-theme", theme) : localStorage.removeItem("bc-theme"); } catch (e) { /* storage unavailable */ }
    themeBtn.textContent = `Theme: ${theme || "auto"}`;
    rerender();
  }
  themeBtn.textContent = `Theme: ${document.documentElement.dataset.theme || "auto"}`;
  themeBtn.addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme;
    applyTheme(cur === undefined ? "light" : cur === "light" ? "dark" : null);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => rerender());

  for (const b of document.querySelectorAll("nav.tabs button")) b.addEventListener("click", () => selectTab(b.dataset.tab));
  window.addEventListener("hashchange", () => selectTab(location.hash.slice(1), false));
  setInterval(updateClock, 1000);
  selectTab(location.hash.slice(1) || "training", false);
  tick();
})();
