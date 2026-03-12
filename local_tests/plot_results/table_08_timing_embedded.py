"""
table_08_timing_embedded.py
Generates: CSV of G computation time and total time for timed embedded experiments
(exp 4 = embedded den-3, exp 5 = embedded den-5). One graph (_0) per config.
Output: local_tests/plot_results/csv/table_08_timing_embedded.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, GRAPH_DENSITIES_DEN5,
                        COLOR_DISTRIBUTIONS, load_embedded_timing)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    rows = []
    for emb_den, graph_dens in [(3, GRAPH_DENSITIES_DEN3), (5, GRAPH_DENSITIES_DEN5)]:
        timing = {algo: load_embedded_timing(algo, emb_den) for algo in ALGORITHMS}
        for g_den in graph_dens:
            for color in COLOR_DISTRIBUTIONS:
                row = {"emb_density": emb_den, "graph_density": g_den, "color": color}
                for algo in ALGORITHMS:
                    t = timing[algo].get((g_den, color))
                    row[f"G_time_{algo}"]     = f"{t['G_time']:.2f}"     if t else "N/A"
                    row[f"total_time_{algo}"] = f"{t['total_time']:.2f}" if t else "N/A"
                rows.append(row)

    fieldnames = ["emb_density","graph_density","color"]
    for algo in ALGORITHMS:
        fieldnames += [f"G_time_{algo}", f"total_time_{algo}"]

    out_path = os.path.join(OUT_DIR, "table_08_timing_embedded.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
