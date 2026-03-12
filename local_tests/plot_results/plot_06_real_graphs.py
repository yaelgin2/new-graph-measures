"""
plot_06_real_graphs.py
Generates: Grouped bar chart of false positive count per algorithm for
Mutagenicity and DHFR-MD real graphs.
Output: local_tests/plot_results/plots/plot_06_real_graphs.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from parse_logs import ALGORITHMS, REAL_GRAPHS, load_real_graph_fp
from plot_helpers import (apply_dark_style, save_fig, ALGO_COLORS, ALGO_LABELS,
                          na_or_zero, is_missing, FONT_SIZE_TITLE,
                          FONT_SIZE_AXIS, FONT_SIZE_LEGEND)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    apply_dark_style()
    fp_data = {algo: load_real_graph_fp(algo) for algo in ALGORITHMS}

    x = np.arange(len(REAL_GRAPHS))
    width = 0.2
    offsets = np.linspace(-(len(ALGORITHMS)-1)/2*width,
                           (len(ALGORITHMS)-1)/2*width, len(ALGORITHMS))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    has_missing = False
    for k, algo in enumerate(ALGORITHMS):
        vals    = [fp_data[algo].get(g) for g in REAL_GRAPHS]
        heights = [na_or_zero(v) for v in vals]
        bars = ax.bar(x + offsets[k], heights, width,
                      label=ALGO_LABELS[algo],
                      color=ALGO_COLORS[algo], alpha=0.85)
        for bar, v in zip(bars, vals):
            if is_missing(v):
                has_missing = True
                bar.set_hatch("/"); bar.set_edgecolor("white")
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                        str(int(h)), ha="center", va="bottom",
                        fontsize=11, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(REAL_GRAPHS, fontsize=13, color="white")
    ax.set_ylabel("False Positives (out of 1000)", fontsize=FONT_SIZE_AXIS)
    ax.set_title("False Positives on Real Graphs (NCI109 Subgraphs)",
                 fontsize=FONT_SIZE_TITLE, color="white", pad=12)
    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.3,
              labelcolor="white", facecolor="#1a1a2e", edgecolor="#aaaaaa")
    ax.grid(axis="y", alpha=0.3)
    if has_missing:
        ax.text(0.99, 0.99, "Hatched = missing data (shown as 0)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#ffcc44")
    save_fig(fig, os.path.join(OUT_DIR, "plot_06_real_graphs.png"))

if __name__ == "__main__":
    main()
