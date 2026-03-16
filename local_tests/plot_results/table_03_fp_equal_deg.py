"""
table_03_fp_equal_deg.py
Generates: CSV of false positive counts for equal-density experiment (exp 3).
1 graph, 1000 S, graph density = subgraph density.
Output: local_tests/plot_results/csv/table_03_fp_equal_deg.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from parse_logs import (ALGORITHMS, EQUAL_DEG_DENSITIES, COLOR_DISTRIBUTIONS,
                        load_equal_deg_fp)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    data = {algo: load_equal_deg_fp(algo) for algo in ALGORITHMS}

    rows = []
    for density in EQUAL_DEG_DENSITIES:
        for color in COLOR_DISTRIBUTIONS:
            row = {"density": density, "color_distribution": color}
            for algo in ALGORITHMS:
                val = data[algo].get((color, density))
                row[algo] = val if val is not None else "N/A"
            rows.append(row)

    out_path = os.path.join(OUT_DIR, "table_03_fp_equal_deg.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["density","color_distribution"] + ALGORITHMS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
