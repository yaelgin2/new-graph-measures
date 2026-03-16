"""
plot_09_s_timing.py
Generates: Grouped bar chart of S computation time vs density per algorithm.
Averages over first 100 S's for consistency across all density levels.
Output: local_tests/plot_results/plots/plot_09_s_timing.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from parse_logs import EQUAL_DEG_DENSITIES, COLOR_DISTRIBUTIONS, load_s_timing, run_script_main
from plot_helpers import (apply_dark_style, save_fig, ALGO_COLORS, ALGO_LABELS,
                          na_or_zero, is_missing, FONT_SIZE_TITLE,
                          FONT_SIZE_AXIS, FONT_SIZE_LEGEND)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

S_ALGOS = ["induced", "non_induced", "paths"]

def main():
    apply_dark_style()
    labels = [f"d{d}\n{c[:3]}" for d in EQUAL_DEG_DENSITIES for c in COLOR_DISTRIBUTIONS]
    x = np.arange(len(labels))
    width = 0.25
    offsets = np.linspace(-(len(S_ALGOS)-1)/2*width,
                           (len(S_ALGOS)-1)/2*width, len(S_ALGOS))

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    has_missing = False
    for k, algo in enumerate(S_ALGOS):
        vals = []
        for d in EQUAL_DEG_DENSITIES:
            timing = load_s_timing(algo, d)
            for c in COLOR_DISTRIBUTIONS:
                run = f"color_{c}_deg_{d}"
                t   = timing.get(run)
                vals.append(None if t is None else t.get("avg_s_time"))

        heights = [na_or_zero(v) for v in vals]
        bars = ax.bar(x + offsets[k], heights, width,
                      label=ALGO_LABELS[algo],
                      color=ALGO_COLORS[algo], alpha=0.85)
        for bar, v in zip(bars, vals):
            if is_missing(v):
                has_missing = True
                bar.set_hatch("/"); bar.set_edgecolor("white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color="white")
    ax.set_xlabel("Density × Color Distribution", fontsize=FONT_SIZE_AXIS)
    ax.set_ylabel("Avg Time per S (seconds)", fontsize=FONT_SIZE_AXIS)
    ax.set_title("Avg S Computation Time by Density and Color Distribution",
                 fontsize=FONT_SIZE_TITLE, color="white", pad=12)
    ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.3,
              labelcolor="white", facecolor="#1a1a2e", edgecolor="#aaaaaa")
    ax.grid(axis="y", alpha=0.3)
    if has_missing:
        ax.text(0.99, 0.99, "Hatched = missing data (shown as 0)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#ffcc44")
    save_fig(fig, os.path.join(OUT_DIR, "plot_09_s_timing.png"))

if __name__ == "__main__":
    run_script_main(main)