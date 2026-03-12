"""
plot_05_fp_vs_density.py
Generates: Line plot of FP rate vs graph density, one line per algorithm per
color distribution. Uses embedded den-3 experiment (averaged over 10 graphs).
Output: local_tests/plot_results/plots/plot_05_fp_vs_density.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, COLOR_DISTRIBUTIONS,
                        load_embedded_fp)
from plot_helpers import (apply_dark_style, save_fig, ALGO_COLORS, ALGO_LABELS,
                          na_or_zero, is_missing, FONT_SIZE_TITLE,
                          FONT_SIZE_AXIS, FONT_SIZE_LEGEND)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_LINESTYLES = {"uniform": "-", "average": "--", "rare": ":"}
COLOR_MARKERS    = {"uniform": "o",  "average": "s",  "rare": "^"}

def main():
    apply_dark_style()
    data = {algo: load_embedded_fp(algo, 3) for algo in ALGORITHMS}

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("False Positives vs Graph Density — Embedded Den-3",
                 fontsize=FONT_SIZE_TITLE, color="white")

    for ax, algo in zip(axes, ALGORITHMS):
        ax.set_facecolor("#16213e")
        for color in COLOR_DISTRIBUTIONS:
            vals = [data[algo].get((g, color)) for g in GRAPH_DENSITIES_DEN3]
            ys   = [na_or_zero(v) for v in vals]
            missing_mask = [is_missing(v) for v in vals]
            ax.plot(GRAPH_DENSITIES_DEN3, ys,
                    linestyle=COLOR_LINESTYLES[color],
                    marker=COLOR_MARKERS[color],
                    color="white" if color == "uniform" else
                          "#F4A261" if color == "average" else "#2A9D8F",
                    linewidth=2, markersize=7,
                    label=color.capitalize())
            # Mark missing points
            for x, y, m in zip(GRAPH_DENSITIES_DEN3, ys, missing_mask):
                if m:
                    ax.plot(x, y, "x", color="#ffcc44", markersize=10, zorder=5)

        ax.set_title(ALGO_LABELS[algo], fontsize=12, color="white", pad=8)
        ax.set_xlabel("Graph Density", fontsize=FONT_SIZE_AXIS)
        ax.set_xticks(GRAPH_DENSITIES_DEN3)
        ax.grid(alpha=0.3)
        ax.tick_params(colors="white")

    axes[0].set_ylabel("Avg False Positives (out of 90)", fontsize=FONT_SIZE_AXIS)
    axes[0].legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.3,
                   labelcolor="white", facecolor="#1a1a2e")
    fig.text(0.5, -0.02, "× = missing data (shown as 0)", ha="center",
             fontsize=9, color="#ffcc44")
    plt.tight_layout()
    save_fig(fig, os.path.join(OUT_DIR, "plot_05_fp_vs_density.png"))

if __name__ == "__main__":
    main()
