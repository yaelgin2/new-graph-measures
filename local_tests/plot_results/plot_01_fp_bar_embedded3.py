"""
plot_01_fp_bar_embedded3.py
Generates: Updated grouped bar chart — false positive count per graph config,
all 4 algorithms side by side, for embedded density-3 subgraphs (exp 1).
Averaged over 10 graph versions. X-axis: graph_density × color_distribution.
Output: local_tests/plot_results/plots/plot_01_fp_bar_embedded3.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, COLOR_DISTRIBUTIONS,
                        load_embedded_fp,
                        run_script_main)
from plot_helpers import (apply_dark_style, save_fig, ALGO_COLORS, ALGO_LABELS,
                          na_or_zero, is_missing, FONT_SIZE_TITLE,
                          FONT_SIZE_AXIS, FONT_SIZE_LEGEND)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# pattern_finder ran 1000-S not 100-S — excluded from this plot
PLOT_ALGOS = ["induced", "non_induced", "paths"]

def main():
    apply_dark_style()
    data = {algo: load_embedded_fp(algo, 3) for algo in PLOT_ALGOS}

    labels = [f"den{g}\n{c[:3]}" for g in GRAPH_DENSITIES_DEN3
              for c in COLOR_DISTRIBUTIONS]
    x = np.arange(len(labels))
    width = 0.2
    n = len(PLOT_ALGOS)
    offsets = np.linspace(-(n-1)/2*width, (n-1)/2*width, n)

    fig, ax = plt.subplots(figsize=(20, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    has_missing = False
    for k, algo in enumerate(PLOT_ALGOS):
        vals, hatches = [], []
        for g in GRAPH_DENSITIES_DEN3:
            for c in COLOR_DISTRIBUTIONS:
                v = data[algo].get((g, c))
                if is_missing(v):
                    has_missing = True
                vals.append(na_or_zero(v))
                hatches.append("/" if is_missing(v) else "")

        bars = ax.bar(x + offsets[k], vals, width,
                      label=ALGO_LABELS[algo],
                      color=ALGO_COLORS[algo], alpha=0.85)
        # Hatch missing bars
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
                bar.set_edgecolor("white")

        # Value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(int(round(v))), ha="center", va="bottom",
                        fontsize=8, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color="white")
    ax.set_xlabel("Graph density × Color distribution", fontsize=FONT_SIZE_AXIS, color="white")
    ax.set_ylabel("Avg False Positives (out of 90, S_11..S_100)", fontsize=FONT_SIZE_AXIS, color="white")
    ax.set_title("False Positives — Embedded Den-3 Subgraphs (10 graphs × 100 S)",
                 fontsize=FONT_SIZE_TITLE, color="white", pad=15)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left",
              framealpha=0.3, labelcolor="white",
              facecolor="#1a1a2e", edgecolor="#aaaaaa")
    ax.grid(axis="y", alpha=0.3)
    if has_missing:
        ax.text(0.99, 0.99, "Hatched = missing data (shown as 0)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#ffcc44")

    save_fig(fig, os.path.join(OUT_DIR, "plot_01_fp_bar_embedded3.png"))

if __name__ == "__main__":
    run_script_main(main)