"""
table_09_s_timing.py
Generates: CSV of S computation timing (exp 7) — time to compute motifs/paths
for S graphs of different densities. Averages over first 100 S's for consistency.
Output: local_tests/plot_results/csv/table_09_s_timing.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from parse_logs import (COLOR_DISTRIBUTIONS, EQUAL_DEG_DENSITIES,
                        load_s_timing)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT_DIR, exist_ok=True)

S_ALGOS = ["induced", "non_induced", "paths"]

def main():
    rows = []
    for density in EQUAL_DEG_DENSITIES:
        for color in COLOR_DISTRIBUTIONS:
            run = f"color_{color}_deg_{density}"
            row = {"density": density, "color": color}
            for algo in S_ALGOS:
                timing = load_s_timing(algo, density)
                t = timing.get(run)
                row[f"total_time_{algo}"] = f"{t['total_time']:.2f}" if t else "N/A"
                row[f"G_time_{algo}"]     = f"{t['G_time']:.2f}"     if t else "N/A"
            rows.append(row)

    fieldnames = ["density","color"]
    for algo in S_ALGOS:
        fieldnames += [f"G_time_{algo}", f"total_time_{algo}"]

    out_path = os.path.join(OUT_DIR, "table_09_s_timing.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
