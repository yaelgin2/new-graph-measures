"""
plot_04_layered.py
Generates: Stacked bar chart showing layered detection improvement.
For each config (emb_den × graph_den × color), shows:
  - caught only by induced / only non_induced / only paths / only pattern_finder
  - caught by 2+ algorithms
  - missed by all (remaining FP)
Output: local_tests/plot_results/plots/plot_04_layered.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, GRAPH_DENSITIES_DEN5,
                        COLOR_DISTRIBUTIONS, EMBEDDED_SKIP, NUM_S_TIMED,
                        load_per_s_results_for_layering)
from plot_helpers import (apply_dark_style, save_fig, ALGO_COLORS, ALGO_LABELS,
                          FONT_SIZE_TITLE, FONT_SIZE_AXIS, FONT_SIZE_LEGEND)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

ONLY_COLORS = {**ALGO_COLORS, "multi": "#9B5DE5", "missed": "#555577"}

def main():
    apply_dark_style()
    s_range = range(EMBEDDED_SKIP + 1, NUM_S_TIMED + 1)

    configs = ([(3, g, c) for g in GRAPH_DENSITIES_DEN3 for c in COLOR_DISTRIBUTIONS] +
               [(5, g, c) for g in GRAPH_DENSITIES_DEN5 for c in COLOR_DISTRIBUTIONS])

    labels   = []
    stacks   = {a: [] for a in ALGORITHMS}
    multi    = []
    missed   = []
    has_missing = False

    for emb_den, g_den, color in configs:
        labels.append(f"e{emb_den}g{g_den}\n{color[:3]}")
        per_algo = load_per_s_results_for_layering(emb_den, g_den, color)
        avail = [a for a in ALGORITHMS if per_algo.get(a) is not None]
        if not avail:
            has_missing = True
            for a in ALGORITHMS:
                stacks[a].append(0)
            multi.append(0); missed.append(0)
            continue

        only  = {a: 0 for a in ALGORITHMS}
        multi_cnt  = 0
        missed_cnt = 0

        for i in s_range:
            caught = [a for a in avail if not per_algo[a].get(i, True)]
            if len(caught) == 0:
                missed_cnt += 1
            elif len(caught) == 1:
                only[caught[0]] += 1
            else:
                multi_cnt += 1

        for a in ALGORITHMS:
            stacks[a].append(only[a])
        multi.append(multi_cnt)
        missed.append(missed_cnt)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(20, len(labels)//2), 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bottom = np.zeros(len(labels))
    for algo in ALGORITHMS:
        vals = np.array(stacks[algo])
        ax.bar(x, vals, bottom=bottom, label=f"Only {ALGO_LABELS[algo]}",
               color=ALGO_COLORS[algo], alpha=0.85)
        bottom += vals

    ax.bar(x, multi,  bottom=bottom, label="Caught by 2+ algos",
           color=ONLY_COLORS["multi"],  alpha=0.85)
    bottom += np.array(multi)
    ax.bar(x, missed, bottom=bottom, label="Missed by all (FP)",
           color=ONLY_COLORS["missed"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color="white")
    ax.set_xlabel("Embedded den × Graph den × Color", fontsize=FONT_SIZE_AXIS)
    ax.set_ylabel("Number of S_11..S_1000", fontsize=FONT_SIZE_AXIS)
    ax.set_title("Layered Detection — How Many S's Does Each Algorithm Uniquely Catch?",
                 fontsize=FONT_SIZE_TITLE, color="white", pad=12)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper right",
              framealpha=0.3, labelcolor="white",
              facecolor="#1a1a2e", edgecolor="#aaaaaa")
    ax.grid(axis="y", alpha=0.3)
    if has_missing:
        ax.text(0.01, 0.99, "Some configs missing data (shown as 0)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color="#ffcc44")

    save_fig(fig, os.path.join(OUT_DIR, "plot_04_layered.png"))

if __name__ == "__main__":
    main()
