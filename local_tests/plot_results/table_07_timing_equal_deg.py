"""
table_07_timing_equal_deg.py
Generates: CSV of G computation time and total time for equal-density experiment.
Output: local_tests/plot_results/csv/table_07_timing_equal_deg.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from parse_logs import (ALGORITHMS, EQUAL_DEG_DENSITIES, COLOR_DISTRIBUTIONS,
                        load_equal_deg_timing)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    timing = {algo: load_equal_deg_timing(algo) for algo in ALGORITHMS}

    rows = []
    for density in EQUAL_DEG_DENSITIES:
        for color in COLOR_DISTRIBUTIONS:
            row = {"density": density, "color": color}
            for algo in ALGORITHMS:
                t = timing[algo].get((color, density))
                row[f"G_time_{algo}"]     = f"{t['G_time']:.2f}"     if t else "N/A"
                row[f"total_time_{algo}"] = f"{t['total_time']:.2f}" if t else "N/A"
                if algo == "pattern_finder" and t:
                    row[f"preprocess_{algo}"] = f"{t['preprocess']:.2f}"
                elif algo == "pattern_finder":
                    row[f"preprocess_{algo}"] = "N/A"
            rows.append(row)

    fieldnames = ["density","color"]
    for algo in ALGORITHMS:
        fieldnames += [f"G_time_{algo}", f"total_time_{algo}"]
        if algo == "pattern_finder":
            fieldnames.append(f"preprocess_{algo}")

    out_path = os.path.join(OUT_DIR, "table_07_timing_equal_deg.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
