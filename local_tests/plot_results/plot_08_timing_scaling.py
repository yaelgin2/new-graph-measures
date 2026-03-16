"""
plot_08_timing_scaling.py
Generates: 2-panel line chart — G time and total time vs graph density,
one line per algorithm. Uses embedded experiments (den-3 and den-5, _0 graph).
Output: local_tests/plot_results/plots/plot_08_timing_scaling.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, GRAPH_DENSITIES_DEN5,
                        load_embedded_timing)
from plot_helpers import (apply_dark_style, save_fig, ALGO_COLORS, ALGO_LABELS,
                          na_or_zero, is_missing, FONT_SIZE_TITLE,
                          FONT_SIZE_AXIS, FONT_SIZE_LEGEND)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    apply_dark_style()
    # Average over color distributions for each graph density
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Timing Scaling vs Graph Density", fontsize=FONT_SIZE_TITLE+2,
                 color="white")

    for row_idx, (emb_den, graph_dens) in enumerate([(3, GRAPH_DENSITIES_DEN3),
                                                      (5, GRAPH_DENSITIES_DEN5)]):
        timing = {algo: load_embedded_timing(algo, emb_den) for algo in ALGORITHMS}

        for col_idx, time_key in enumerate(["G_time", "total_time"]):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("#16213e")
            has_missing = False

            for algo in ALGORITHMS:
                ys = []
                for g in graph_dens:
                    # Average over color distributions
                    vals = [timing[algo].get((g, c)) for c in
                            ["uniform", "average", "rare"]]
                    nums = [v[time_key] for v in vals if v is not None and time_key in v]
                    if nums:
                        ys.append(sum(nums) / len(nums))
                    else:
                        has_missing = True
                        ys.append(0)

                ax.plot(graph_dens, ys, marker="o", linewidth=2.5,
                        markersize=8, label=ALGO_LABELS[algo],
                        color=ALGO_COLORS[algo])

            title_part = "G Computation Time" if time_key == "G_time" else "Total Time"
            ax.set_title(f"Emb-den {emb_den} — {title_part} (avg over colors)",
                         fontsize=12, color="white", pad=8)
            ax.set_xlabel("Graph Density", fontsize=FONT_SIZE_AXIS)
            ax.set_ylabel("Seconds", fontsize=FONT_SIZE_AXIS)
            ax.set_xticks(graph_dens)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=FONT_SIZE_LEGEND, framealpha=0.3,
                      labelcolor="white", facecolor="#1a1a2e")
            if has_missing:
                ax.text(0.01, 0.99, "Some data missing (shown as 0)",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=9, color="#ffcc44")

    plt.tight_layout()
    save_fig(fig, os.path.join(OUT_DIR, "plot_08_timing_scaling.png"))

if __name__ == "__main__":
    main()
