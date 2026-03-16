"""
table_06_real_graphs.py
Generates: CSV of false positive counts and timing for real graphs
(Mutagenicity, DHFR-MD) for all algorithms.
Output: local_tests/plot_results/csv/table_06_real_graphs.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from parse_logs import ALGORITHMS, REAL_GRAPHS, load_real_graph_fp, load_real_graph_timing

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    fp_data     = {algo: load_real_graph_fp(algo)     for algo in ALGORITHMS}
    timing_data = {algo: load_real_graph_timing(algo) for algo in ALGORITHMS}

    rows = []
    for gname in REAL_GRAPHS:
        row = {"graph": gname}
        for algo in ALGORITHMS:
            fp  = fp_data[algo].get(gname)
            row[f"fp_{algo}"] = fp if fp is not None else "N/A"
            t   = timing_data[algo].get(gname)
            row[f"G_time_{algo}"]     = f"{t['G_time']:.2f}"     if t else "N/A"
            row[f"total_time_{algo}"] = f"{t['total_time']:.2f}" if t else "N/A"
        rows.append(row)

    fieldnames = ["graph"]
    for algo in ALGORITHMS:
        fieldnames += [f"fp_{algo}", f"G_time_{algo}", f"total_time_{algo}"]

    out_path = os.path.join(OUT_DIR, "table_06_real_graphs.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
