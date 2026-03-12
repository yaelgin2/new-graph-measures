"""
plot_03_fp_heatmap.py
Generates: 4-panel heatmap (one per algorithm) — rows=graph density,
cols=color distribution, cell=avg FP rate. Uses embedded den-3 experiment.
Output: local_tests/plot_results/plots/plot_03_fp_heatmap.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, COLOR_DISTRIBUTIONS,
                        load_embedded_fp)
from plot_helpers import (apply_dark_style, save_fig, ALGO_LABELS,
                          FONT_SIZE_TITLE, FONT_SIZE_AXIS)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    apply_dark_style()
    data = {algo: load_embedded_fp(algo, 3) for algo in ALGORITHMS}

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("False Positive Heatmap — Embedded Den-3 (avg over 10 graphs)",
                 fontsize=FONT_SIZE_TITLE, color="white", y=1.02)

    for ax, algo in zip(axes, ALGORITHMS):
        matrix = np.full((len(GRAPH_DENSITIES_DEN3), len(COLOR_DISTRIBUTIONS)), np.nan)
        for i, g in enumerate(GRAPH_DENSITIES_DEN3):
            for j, c in enumerate(COLOR_DISTRIBUTIONS):
                v = data[algo].get((g, c))
                if v is not None:
                    matrix[i, j] = v

        # Use a masked array so NaN shows as grey
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="#444466")

        im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=90)
        ax.set_xticks(range(len(COLOR_DISTRIBUTIONS)))
        ax.set_xticklabels(COLOR_DISTRIBUTIONS, fontsize=11, color="white")
        ax.set_yticks(range(len(GRAPH_DENSITIES_DEN3)))
        ax.set_yticklabels([f"den {g}" for g in GRAPH_DENSITIES_DEN3],
                           fontsize=11, color="white")
        ax.set_title(ALGO_LABELS[algo], fontsize=12, color="white", pad=8)
        ax.set_facecolor("#16213e")

        # Cell annotations
        for i in range(len(GRAPH_DENSITIES_DEN3)):
            for j in range(len(COLOR_DISTRIBUTIONS)):
                v = matrix[i, j]
                txt = f"{v:.1f}" if not np.isnan(v) else "N/A"
                color = "black" if (not np.isnan(v) and v > 45) else "white"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="white")

    plt.tight_layout()
    save_fig(fig, os.path.join(OUT_DIR, "plot_03_fp_heatmap.png"))

if __name__ == "__main__":
    main()
