"""Matplotlib styling for report figures, matching the dashboard's light theme.

Color roles follow the dashboard: classes take categorical slots 1-3 (validated
for every pair, light and dark), non-class single series use the violet accent,
and context series (e.g. training loss next to validation loss) use muted gray.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES_DIR  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#6250d6"
CLASS_COLORS = {"WBC": "#2a78d6", "RBC": "#eb6834", "Platelets": "#1baf7a"}
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
    })


def save(fig, name: str, directory=FIGURES_DIR) -> str:
    """Save a figure as PNG and return its path relative to the reports directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{path.name}"


def end_labels(ax, points: list[tuple[float, float, str]], dx: float = 6, gap: float = 11) -> None:
    """Direct labels at line end points, in text ink (never the series color).

    points holds (x, y, text) per series. Labels are spread vertically to stay `gap` points
    apart and inside the axes, and a label that had to move gets a thin leader line to its
    line's end. The spacing is computed in screen units, so call this once the layout is
    final (after tight_layout).
    """
    pt_per_px = 72 / ax.figure.dpi
    anchors = [ax.transData.transform((x, y))[1] * pt_per_px for x, y, _ in points]
    order = sorted(range(len(points)), key=lambda i: anchors[i])
    pos = [anchors[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    top = ax.bbox.y1 * pt_per_px - gap / 2
    for k in reversed(range(len(pos))):  # pull a stack that overflows the top back inside
        pos[k] = min(pos[k], top if k == len(pos) - 1 else pos[k + 1] - gap)
    for k, i in enumerate(order):
        x, y, text = points[i]
        dy = pos[k] - anchors[i]
        leader = dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=1, shrinkB=2) if abs(dy) > 2 else None
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=9,
                    color=INK_2, arrowprops=leader)
