"""
plot_03_fp_heatmap.py
Generates two 5-panel heatmaps (one per embedded density):
  panels 1-4: FP count per algorithm (1 graph × 1000 S experiment)
  panel 5:    S's missed by ALL 4 algorithms simultaneously
Output: local_tests/plot_results/plots/plot_03_fp_heatmap_den3.png
        local_tests/plot_results/plots/plot_03_fp_heatmap_den5.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, GRAPH_DENSITIES_DEN5,
                        COLOR_DISTRIBUTIONS, EMBEDDED_SKIP, NUM_S_TIMED,
                        load_timed_embedded_fp,
                        load_per_s_results_for_layering,
                        run_script_main)
from plot_helpers import (apply_dark_style, save_fig, ALGO_LABELS,
                          FONT_SIZE_TITLE, FONT_SIZE_AXIS)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_S = NUM_S_TIMED - EMBEDDED_SKIP   # 990


def compute_all_missed(embedded_den, densities):
    """
    Returns dict: {(g_den, color): count of S's missed by ALL 4 algos}
    """
    s_range = range(EMBEDDED_SKIP + 1, NUM_S_TIMED + 1)
    result  = {}
    for g_den in densities:
        for color in COLOR_DISTRIBUTIONS:
            per_algo = load_per_s_results_for_layering(embedded_den, g_den, color)
            avail    = [a for a in ALGORITHMS if per_algo.get(a) is not None]
            if not avail:
                result[(g_den, color)] = None
                continue
            missed = sum(
                1 for i in s_range
                if all(per_algo[a].get(i, False) for a in avail)
            )
            result[(g_den, color)] = missed
    return result


def make_heatmap(embedded_den, densities, out_path):
    apply_dark_style()
    fp_data     = {algo: load_timed_embedded_fp(algo, embedded_den) for algo in ALGORITHMS}
    missed_data = compute_all_missed(embedded_den, densities)

    panels      = ALGORITHMS + ["all_missed"]
    panel_titles = {a: ALGO_LABELS[a] for a in ALGORITHMS}
    panel_titles["all_missed"] = "Missed by ALL 4"

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"False Positive Heatmap — Embedded Den-{embedded_den} (1 graph × 1000 S)",
        fontsize=FONT_SIZE_TITLE, color="white", y=1.02
    )

    # Build all matrices first so we can compute all_missed's own vmax
    matrices = {}
    for panel in panels:
        m   = np.full((len(densities), len(COLOR_DISTRIBUTIONS)), np.nan)
        src = fp_data[panel] if panel != "all_missed" else missed_data
        for i, g in enumerate(densities):
            for j, c in enumerate(COLOR_DISTRIBUTIONS):
                v = src.get((g, c))
                if v is not None:
                    m[i, j] = v
        matrices[panel] = m

    for ax, panel in zip(axes, panels):
        matrix = matrices[panel]
        masked = np.ma.masked_invalid(matrix)

        # Same colormap and scale (0..990) for all panels
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="#444466")

        im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=MAX_S)
        ax.set_xticks(range(len(COLOR_DISTRIBUTIONS)))
        ax.set_xticklabels(COLOR_DISTRIBUTIONS, fontsize=11, color="white")
        ax.set_yticks(range(len(densities)))
        ax.set_yticklabels([f"den {g}" for g in densities], fontsize=11, color="white")
        ax.set_title(panel_titles[panel], fontsize=12, color="white", pad=8)
        ax.set_facecolor("#16213e")

        for i in range(len(densities)):
            for j in range(len(COLOR_DISTRIBUTIONS)):
                v = matrix[i, j]
                txt   = f"{v:.0f}" if not np.isnan(v) else "N/A"
                color = "black" if (not np.isnan(v) and v > MAX_S / 2) else "white"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="white")

    plt.tight_layout()
    save_fig(fig, out_path)
    print(f"Written: {out_path}")


def main():
    make_heatmap(3, GRAPH_DENSITIES_DEN3,
                 os.path.join(OUT_DIR, "plot_03_fp_heatmap_den3.png"))
    make_heatmap(5, GRAPH_DENSITIES_DEN5,
                 os.path.join(OUT_DIR, "plot_03_fp_heatmap_den5.png"))

if __name__ == "__main__":
    run_script_main(main)