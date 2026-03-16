"""
plot_04_layered.py
Grouped bar chart: for each config (emb_den × graph_den × color), 5 bars showing:
  - How many S's each algorithm MISSED (false negatives among non-embedded S's)
  - How many S's ALL 4 algorithms missed simultaneously
Output: local_tests/plot_results/plots/plot_04_layered.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, GRAPH_DENSITIES_DEN5,
                        COLOR_DISTRIBUTIONS, EMBEDDED_SKIP, NUM_S_TIMED,
                        load_per_s_results_for_layering,
                        run_script_main)
from plot_helpers import (apply_dark_style, save_fig, ALGO_COLORS, ALGO_LABELS,
                          FONT_SIZE_TITLE, FONT_SIZE_AXIS, FONT_SIZE_LEGEND)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

ALL_MISSED_COLOR = "#9B5DE5"
BARS      = ALGORITHMS + ["all_missed"]
BAR_LABEL = {a: f"{ALGO_LABELS[a]} missed" for a in ALGORITHMS}
BAR_LABEL["all_missed"] = "All 4 missed"
BAR_COLOR = {**ALGO_COLORS, "all_missed": ALL_MISSED_COLOR}


def main():
    apply_dark_style()
    s_range = range(EMBEDDED_SKIP + 1, NUM_S_TIMED + 1)

    configs = ([(3, g, c) for g in GRAPH_DENSITIES_DEN3 for c in COLOR_DISTRIBUTIONS] +
               [(5, g, c) for g in GRAPH_DENSITIES_DEN5 for c in COLOR_DISTRIBUTIONS])

    labels      = []
    bar_vals    = {b: [] for b in BARS}
    has_missing = False

    for emb_den, g_den, color in configs:
        labels.append(f"e{emb_den}g{g_den}\n{color[:3]}")
        per_algo = load_per_s_results_for_layering(emb_den, g_den, color)
        avail    = [a for a in ALGORITHMS if per_algo.get(a) is not None]

        if not avail:
            has_missing = True
            for b in BARS:
                bar_vals[b].append(0)
            continue

        # For each algo: count S's it MISSED (did not catch = PASS = FP)
        # A "miss" = algo said PASS (feasible) but S is not embedded → false positive
        for a in ALGORITHMS:
            if a in avail:
                missed = sum(1 for i in s_range if per_algo[a].get(i, False))
            else:
                missed = 0
            bar_vals[a].append(missed)

        # All-4-missed: S's where every available algo said PASS
        all_missed = sum(
            1 for i in s_range
            if all(per_algo[a].get(i, False) for a in avail)
        )
        bar_vals["all_missed"].append(all_missed)

    n      = len(BARS)
    width  = 0.15
    x      = np.arange(len(labels))
    offsets = np.linspace(-(n-1)/2*width, (n-1)/2*width, n)

    fig, ax = plt.subplots(figsize=(max(24, len(labels) // 2), 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for k, bar in enumerate(BARS):
        vals = np.array(bar_vals[bar], dtype=float)
        bars = ax.bar(x + offsets[k], vals, width,
                      label=BAR_LABEL[bar],
                      color=BAR_COLOR[bar], alpha=0.85)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                        str(int(v)), ha="center", va="bottom",
                        fontsize=6, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, color="white")
    ax.set_xlabel("Embedded den × Graph den × Color", fontsize=FONT_SIZE_AXIS)
    ax.set_ylabel("Number of S's missed (false positives)", fontsize=FONT_SIZE_AXIS)
    ax.set_title("Missed S's per Algorithm and All-4-Missed (S_11..S_1000)",
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
    run_script_main(main)