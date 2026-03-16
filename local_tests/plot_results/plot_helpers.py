"""
plot_helpers.py — Shared matplotlib style helpers for all plot scripts.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALGO_COLORS = {
    "induced":        "#4C9BE8",
    "non_induced":    "#F4A261",
    "paths":          "#2A9D8F",
    "pattern_finder": "#E76F51",
}
ALGO_LABELS = {
    "induced":        "Motif Induced",
    "non_induced":    "Motif Non-Induced",
    "paths":          "Path Finder",
    "pattern_finder": "Pattern Finder",
}
ALGORITHMS = ["induced", "non_induced", "paths", "pattern_finder"]

FONT_SIZE_TITLE  = 15
FONT_SIZE_AXIS   = 13
FONT_SIZE_TICK   = 11
FONT_SIZE_LEGEND = 12


def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  "#1a1a2e",
        "axes.facecolor":    "#16213e",
        "axes.edgecolor":    "#aaaaaa",
        "axes.labelcolor":   "white",
        "xtick.color":       "white",
        "ytick.color":       "white",
        "text.color":        "white",
        "grid.color":        "#333355",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.size":         FONT_SIZE_TICK,
    })


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {path}")


def na_or_zero(val):
    """Return 0 for plots when val is None (missing data)."""
    return 0 if val is None else val


def is_missing(val):
    return val is None
