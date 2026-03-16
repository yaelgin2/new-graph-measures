"""
table_01_fp_embedded3.py
Generates: CSV table of average false positive rates across 10 graph versions
for experiments 1 (embedded density-3 subgraphs, 100 S per graph).
Rows: graph density × color distribution. Columns: algorithm.
Output: local_tests/plot_results/csv/table_01_fp_embedded3.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, COLOR_DISTRIBUTIONS,
                        load_embedded_fp)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    data = {algo: load_embedded_fp(algo, 3) for algo in ALGORITHMS}

    rows = []
    for g_den in GRAPH_DENSITIES_DEN3:
        for color in COLOR_DISTRIBUTIONS:
            row = {"graph_density": g_den, "color_distribution": color}
            for algo in ALGORITHMS:
                val = data[algo].get((g_den, color))
                row[algo] = f"{val:.2f}" if val is not None else "N/A"
            rows.append(row)

    out_path = os.path.join(OUT_DIR, "table_01_fp_embedded3.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["graph_density","color_distribution"] + ALGORITHMS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
