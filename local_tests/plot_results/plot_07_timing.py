"""
plot_07_timing.py
Generates: Updated 3-panel timing chart (preprocessing/G-only/total) for
equal-density experiment. All 4 algorithms, larger clearer legend.
Output: local_tests/plot_results/plots/plot_07_timing.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from parse_logs import (ALGORITHMS, EQUAL_DEG_DENSITIES, COLOR_DISTRIBUTIONS,
                        load_equal_deg_timing)
from plot_helpers import (apply_dark_style, save_fig, ALGO_COLORS, ALGO_LABELS,
                          na_or_zero, is_missing, FONT_SIZE_TITLE,
                          FONT_SIZE_AXIS, FONT_SIZE_LEGEND)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    apply_dark_style()
    timing = {algo: load_equal_deg_timing(algo) for algo in ALGORITHMS}

    labels = [f"{c[:3]}\nd{d}" for d in EQUAL_DEG_DENSITIES
              for c in COLOR_DISTRIBUTIONS]
    x = np.arange(len(labels))
    width = 0.2
    offsets = np.linspace(-(len(ALGORITHMS)-1)/2*width,
                           (len(ALGORITHMS)-1)/2*width, len(ALGORITHMS))

    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Algorithm Timing Comparison — Equal-Density Experiment",
                 fontsize=FONT_SIZE_TITLE + 2, color="white", y=0.98)

    panels = [
        ("A — Preprocessing / Pattern-build Time",
         lambda t: t.get("preprocess", t.get("G_time", 0))),
        ("B — G Search Time (without preprocessing)",
         lambda t: t.get("G_time", 0)),
        ("C — Total Time (preprocessing + search)",
         lambda t: t.get("total_time", 0)),
    ]

    for ax, (title, extractor) in zip(axes, panels):
        ax.set_facecolor("#16213e")
        has_missing = False
        for k, algo in enumerate(ALGORITHMS):
            vals = []
            for d in EQUAL_DEG_DENSITIES:
                for c in COLOR_DISTRIBUTIONS:
                    t = timing[algo].get((c, d))
                    vals.append(None if t is None else extractor(t))

            heights = [na_or_zero(v) for v in vals]
            bars = ax.bar(x + offsets[k], heights, width,
                          label=ALGO_LABELS[algo],
                          color=ALGO_COLORS[algo], alpha=0.85)
            for bar, v in zip(bars, vals):
                if is_missing(v):
                    has_missing = True
                    bar.set_hatch("/"); bar.set_edgecolor("white")

        ax.set_title(title, fontsize=13, color="white", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, color="white")
        ax.set_ylabel("Seconds", fontsize=FONT_SIZE_AXIS)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left",
                  framealpha=0.4, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="#aaaaaa",
                  ncol=2)
        if has_missing:
            ax.text(0.99, 0.99, "Hatched = missing data",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color="#ffcc44")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, os.path.join(OUT_DIR, "plot_07_timing.png"))

if __name__ == "__main__":
    main()
