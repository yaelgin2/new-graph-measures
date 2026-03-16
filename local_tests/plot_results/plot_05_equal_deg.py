"""
plot_05_equal_deg.py
Two-panel figure for the equal-density experiment (G and S have the same density).
Densities: 3, 5, 8, 15 — Color distributions: uniform, average, rare — 1000 S's.

Top panel:    Grouped bar chart — false positive count per algo
              X-axis: density × color   Bars: one per algorithm
Bottom panel: Grouped bar chart — total runtime per algo (seconds)
              Same x-axis layout

Output: local_tests/plot_results/plots/plot_05_equal_deg.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from parse_logs import (ALGORITHMS, EQUAL_DEG_DENSITIES, COLOR_DISTRIBUTIONS,
                        NUM_S_TIMED, EMBEDDED_SKIP,
                        load_equal_deg_fp, load_equal_deg_timing,
                        run_script_main)
from plot_helpers import (apply_dark_style, save_fig, ALGO_COLORS, ALGO_LABELS,
                          na_or_zero, is_missing,
                          FONT_SIZE_TITLE, FONT_SIZE_AXIS, FONT_SIZE_LEGEND)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_FP = NUM_S_TIMED - EMBEDDED_SKIP   # 990


def main():
    apply_dark_style()

    fp_data     = {a: load_equal_deg_fp(a)     for a in ALGORITHMS}
    timing_data = {a: load_equal_deg_timing(a) for a in ALGORITHMS}

    labels  = [f"deg{d}\n{c[:3]}" for d in EQUAL_DEG_DENSITIES
                                   for c in COLOR_DISTRIBUTIONS]
    keys    = [(c, d) for d in EQUAL_DEG_DENSITIES for c in COLOR_DISTRIBUTIONS]
    x       = np.arange(len(labels))
    n       = len(ALGORITHMS)
    width   = 0.18
    offsets = np.linspace(-(n-1)/2*width, (n-1)/2*width, n)

    fig, (ax_fp, ax_t) = plt.subplots(2, 1, figsize=(20, 12), constrained_layout=True)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax_fp, ax_t):
        ax.set_facecolor("#16213e")

    fig.suptitle("Equal-Density Experiment — G and S Same Density",
                 fontsize=FONT_SIZE_TITLE, color="white")

    # ── Top panel: False Positives ────────────────────────────────────────────
    has_missing_fp = False
    for k, algo in enumerate(ALGORITHMS):
        vals = [fp_data[algo].get(key) for key in keys]
        heights = [na_or_zero(v) for v in vals]
        bars = ax_fp.bar(x + offsets[k], heights, width,
                         label=ALGO_LABELS[algo],
                         color=ALGO_COLORS[algo], alpha=0.85)
        for bar, v in zip(bars, vals):
            if is_missing(v):
                has_missing_fp = True
                bar.set_hatch("/"); bar.set_edgecolor("white")
            elif v > 0:
                ax_fp.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + 2,
                           str(int(round(v))), ha="center", va="bottom",
                           fontsize=7, color="white")

    ax_fp.set_xticks(x)
    ax_fp.set_xticklabels(labels, fontsize=9, color="white")
    ax_fp.set_ylabel(f"False Positives (out of {MAX_FP})",
                     fontsize=FONT_SIZE_AXIS, color="white")
    ax_fp.set_title("False Positive Count", fontsize=12, color="white", pad=8)
    ax_fp.set_ylim(0, MAX_FP * 1.12)
    ax_fp.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left",
                 framealpha=0.3, labelcolor="white",
                 facecolor="#1a1a2e", edgecolor="#aaaaaa")
    ax_fp.grid(axis="y", alpha=0.3)
    if has_missing_fp:
        ax_fp.text(0.99, 0.99, "Hatched = missing data (shown as 0)",
                   transform=ax_fp.transAxes, ha="right", va="top",
                   fontsize=9, color="#ffcc44")

    # ── Bottom panel: Timing ──────────────────────────────────────────────────
    has_missing_t = False
    for k, algo in enumerate(ALGORITHMS):
        vals = []
        for key in keys:
            t = timing_data[algo].get(key)
            vals.append(t["total_time"] if t else None)

        heights = [na_or_zero(v) for v in vals]
        bars = ax_t.bar(x + offsets[k], heights, width,
                        label=ALGO_LABELS[algo],
                        color=ALGO_COLORS[algo], alpha=0.85)
        for bar, v in zip(bars, vals):
            if is_missing(v):
                has_missing_t = True
                bar.set_hatch("/"); bar.set_edgecolor("white")

    ax_t.set_xticks(x)
    ax_t.set_xticklabels(labels, fontsize=9, color="white")
    ax_t.set_xlabel("Density × Color Distribution",
                    fontsize=FONT_SIZE_AXIS, color="white")
    ax_t.set_ylabel("Total Runtime (seconds)",
                    fontsize=FONT_SIZE_AXIS, color="white")
    ax_t.set_title("Total Runtime", fontsize=12, color="white", pad=8)
    ax_t.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left",
                framealpha=0.3, labelcolor="white",
                facecolor="#1a1a2e", edgecolor="#aaaaaa")
    ax_t.grid(axis="y", alpha=0.3)
    if has_missing_t:
        ax_t.text(0.99, 0.99, "Hatched = missing data (shown as 0)",
                  transform=ax_t.transAxes, ha="right", va="top",
                  fontsize=9, color="#ffcc44")

    save_fig(fig, os.path.join(OUT_DIR, "plot_05_equal_deg.png"))

if __name__ == "__main__":
    run_script_main(main)