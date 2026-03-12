"""
plot_10_efficiency_frontier.py
Generates: Scatter plot — x=total_time, y=false_positives_caught (990-FP),
one point per algorithm per config (equal-density experiment).
Shows which algorithm gives best detection per second.
Output: local_tests/plot_results/plots/plot_10_efficiency_frontier.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from parse_logs import (ALGORITHMS, EQUAL_DEG_DENSITIES, COLOR_DISTRIBUTIONS,
                        load_equal_deg_fp, load_equal_deg_timing)
from plot_helpers import (apply_dark_style, save_fig, ALGO_COLORS, ALGO_LABELS,
                          FONT_SIZE_TITLE, FONT_SIZE_AXIS, FONT_SIZE_LEGEND)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

MARKERS = {"induced": "o", "non_induced": "s", "paths": "^", "pattern_finder": "D"}

def main():
    apply_dark_style()
    fp_data  = {algo: load_equal_deg_fp(algo)     for algo in ALGORITHMS}
    t_data   = {algo: load_equal_deg_timing(algo) for algo in ALGORITHMS}

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for algo in ALGORITHMS:
        xs, ys, labels_pt = [], [], []
        for d in EQUAL_DEG_DENSITIES:
            for c in COLOR_DISTRIBUTIONS:
                fp = fp_data[algo].get((c, d))
                t  = t_data[algo].get((c, d))
                if fp is None or t is None:
                    continue
                total_t = t.get("total_time")
                if total_t is None:
                    continue
                caught = 990 - fp  # 990 = NUM_S_TIMED - EMBEDDED_SKIP
                xs.append(total_t)
                ys.append(caught)
                labels_pt.append(f"d{d}/{c[:3]}")

        sc = ax.scatter(xs, ys, c=ALGO_COLORS[algo], marker=MARKERS[algo],
                        s=100, alpha=0.8, label=ALGO_LABELS[algo],
                        edgecolors="white", linewidth=0.5)
        for x, y, lbl in zip(xs, ys, labels_pt):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color="white", alpha=0.7)

    ax.set_xlabel("Total Time (seconds)", fontsize=FONT_SIZE_AXIS)
    ax.set_ylabel("S's Correctly Rejected (out of 990)", fontsize=FONT_SIZE_AXIS)
    ax.set_title("Efficiency Frontier — Detection Power vs Compute Time",
                 fontsize=FONT_SIZE_TITLE, color="white", pad=12)
    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.3,
              labelcolor="white", facecolor="#1a1a2e", edgecolor="#aaaaaa",
              markerscale=1.5)
    ax.grid(alpha=0.3)
    save_fig(fig, os.path.join(OUT_DIR, "plot_10_efficiency_frontier.png"))

if __name__ == "__main__":
    main()
