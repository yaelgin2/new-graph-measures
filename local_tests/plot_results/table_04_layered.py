"""
table_04_layered.py
Generates: CSV of layered false positive counts — how many S's pass ALL algorithms
vs. are caught by at least one. Uses timed embedded experiments (_0 graphs, 1000 S).
Output: local_tests/plot_results/csv/table_04_layered.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, GRAPH_DENSITIES_DEN5,
                        COLOR_DISTRIBUTIONS, EMBEDDED_SKIP, NUM_S_TIMED,
                        load_per_s_results_for_layering)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    rows = []
    configs = (
        [(3, g, c) for g in GRAPH_DENSITIES_DEN3 for c in COLOR_DISTRIBUTIONS] +
        [(5, g, c) for g in GRAPH_DENSITIES_DEN5 for c in COLOR_DISTRIBUTIONS]
    )

    for emb_den, g_den, color in configs:
        per_algo = load_per_s_results_for_layering(emb_den, g_den, color, j=0)

        # Only consider S_11..S_1000
        s_range = range(EMBEDDED_SKIP + 1, NUM_S_TIMED + 1)

        # FP per algo
        fp_each = {}
        for algo in ALGORITHMS:
            res = per_algo.get(algo)
            if res is None:
                fp_each[algo] = None
            else:
                fp_each[algo] = sum(1 for i in s_range if res.get(i, False))

        # Layered: passes ALL algorithms (worst case FP)
        available = [a for a in ALGORITHMS if per_algo.get(a) is not None]
        if available:
            fp_all = sum(
                1 for i in s_range
                if all(per_algo[a].get(i, False) for a in available)
            )
            fp_any_caught = sum(
                1 for i in s_range
                if any(not per_algo[a].get(i, True) for a in available)
            )
        else:
            fp_all = None
            fp_any_caught = None

        row = {
            "emb_density": emb_den,
            "graph_density": g_den,
            "color": color,
            "fp_all_pass": fp_all,
            "caught_by_at_least_one": fp_any_caught,
        }
        for algo in ALGORITHMS:
            row[f"fp_{algo}"] = fp_each.get(algo, "N/A")
        rows.append(row)

    fieldnames = (["emb_density","graph_density","color","fp_all_pass",
                   "caught_by_at_least_one"] +
                  [f"fp_{a}" for a in ALGORITHMS])
    out_path = os.path.join(OUT_DIR, "table_04_layered.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
