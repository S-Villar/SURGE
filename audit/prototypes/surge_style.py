"""SURGE visual system — prototype theme module.

One place that defines typography, spacing, color, and output settings for
every SURGE figure. Charts import roles ("series 1", "good", "grid"), never
raw hex, so the palette can be swapped once without touching plot code.

Design rules encoded here (see audit/AUDIT_REPORT.md §11):
  * categorical hues in fixed slot order, never cycled past 8 — fold to "Other"
  * sequential = one hue light->dark; diverging = blue<->red with gray midpoint
  * status colors (pass/fail) are reserved and never used as series colors
  * text wears ink colors, never series colors
  * one axis per chart — never twin y-scales
  * deterministic output: fixed dpi, no timestamps in metadata, svg hashsalt

The palette is a brand-neutral, CVD-validated default (validated with the
dataviz palette validator: adjacent-pair CVD dE >= 8, normal-vision dE >= 15,
in light and dark modes). Swap PALETTES values to rebrand; keep the order.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------- color

PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        # categorical slots — fixed order, never cycled
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        # sequential ramp (blue, light->dark) for density/heatmaps
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                "#256abf", "#184f95", "#0d366b"],
        # status — reserved, never series colors
        "good": "#0ca30c", "warning": "#fab219",
        "serious": "#ec835a", "critical": "#d03b3b",
        "good_text": "#006300",
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
        "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5",
                "#6da7ec", "#9ec5f4", "#cde2fb"],
        "good": "#0ca30c", "warning": "#fab219",
        "serious": "#ec835a", "critical": "#d03b3b",
        "good_text": "#0ca30c",
    },
}

# ---------------------------------------------------------------- typography

_FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

_BASE = 9.0          # pt — publication figures read at column width
_DPI_SCREEN = 150
_DPI_PRINT = 300


def rc_params(mode: str = "light") -> dict:
    p = PALETTES[mode]
    return {
        # figure
        "figure.facecolor": p["surface"],
        "figure.dpi": _DPI_SCREEN,
        "savefig.dpi": _DPI_PRINT,
        "savefig.facecolor": p["surface"],
        "figure.constrained_layout.use": True,
        # axes
        "axes.facecolor": p["surface"],
        "axes.edgecolor": p["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": _BASE + 2,
        "axes.titleweight": "bold",
        "axes.titlecolor": p["ink"],
        "axes.labelsize": _BASE,
        "axes.labelcolor": p["ink2"],
        "axes.prop_cycle": mpl.cycler(color=p["series"]),
        # grid: recessive hairlines
        "grid.color": p["grid"],
        "grid.linewidth": 0.6,
        # ticks
        "xtick.color": p["muted"], "ytick.color": p["muted"],
        "xtick.labelsize": _BASE - 1, "ytick.labelsize": _BASE - 1,
        "xtick.major.size": 0, "ytick.major.size": 0,
        # lines & markers: thin marks
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "patch.linewidth": 0,
        # legend: hairline box, small
        "legend.frameon": False,
        "legend.fontsize": _BASE - 1,
        # text
        "font.family": "sans-serif",
        "font.sans-serif": _FONT_STACK,
        "font.size": _BASE,
        "text.color": p["ink"],
        # deterministic SVG output
        "svg.hashsalt": "surge",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }


@contextmanager
def surge_theme(mode: str = "light"):
    """Context manager: all figures created inside use the SURGE theme."""
    with mpl.rc_context(rc_params(mode)):
        yield PALETTES[mode]


def save_figure(fig, out_stem: Path, formats=("png", "svg", "pdf")) -> list[Path]:
    """Deterministic multi-format export: identical input -> identical bytes.

    Strips creation dates from SVG/PDF/PNG metadata so artifact diffs are
    meaningful in CI and reports.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    meta = {
        "png": {"Software": "SURGE", "Creation Time": ""},
        "svg": {"Date": None},
        "pdf": {"CreationDate": None, "ModDate": None, "Creator": "SURGE"},
    }
    for fmt in formats:
        path = out_stem.with_suffix("." + fmt)
        fig.savefig(path, format=fmt, metadata=meta.get(fmt))
        written.append(path)
    return written


def fmt_metric(value: float, kind: str = "r2") -> str:
    """Consistent metric formatting across CLI, plots, and reports."""
    if value is None:
        return "—"
    if kind in ("r2", "accuracy", "auroc", "f1"):
        return f"{value:.3f}"
    if kind in ("rmse", "mae", "nrmse", "rel_l2"):
        return f"{value:.3g}"
    if kind == "runtime":
        return f"{value:.1f}s" if value >= 1 else f"{value*1000:.0f}ms"
    if kind == "pm":  # mean ± std
        return f"{value[0]:.3f} ± {value[1]:.3f}"
    return f"{value:.4g}"
