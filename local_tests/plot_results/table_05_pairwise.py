"""
table_05_pairwise.py
Generates: CSV of pairwise overlap — for each pair of algorithms, how many S's
does each uniquely catch that the other misses. Uses timed embedded _0 graphs.
Output: local_tests/plot_results/csv/table_05_pairwise.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from itertools import combinations
from parse_logs import (ALGORITHMS, GRAPH_DENSITIES_DEN3, GRAPH_DENSITIES_DEN5,
                        COLOR_DISTRIBUTIONS, EMBEDDED_SKIP, NUM_S_TIMED,
                        load_per_s_results_for_layering)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    # Aggregate over all configs
    pair_stats = {(a, b): {"only_a": 0, "only_b": 0, "both": 0, "neither": 0}
                  for a, b in combinations(ALGORITHMS, 2)}

    configs = (
        [(3, g, c) for g in GRAPH_DENSITIES_DEN3 for c in COLOR_DISTRIBUTIONS] +
        [(5, g, c) for g in GRAPH_DENSITIES_DEN5 for c in COLOR_DISTRIBUTIONS]
    )

    s_range = range(EMBEDDED_SKIP + 1, NUM_S_TIMED + 1)

    for emb_den, g_den, color in configs:
        per_algo = load_per_s_results_for_layering(emb_den, g_den, color, j=0)

        for a, b in combinations(ALGORITHMS, 2):
            ra = per_algo.get(a)
            rb = per_algo.get(b)
            if ra is None or rb is None:
                continue
            for i in s_range:
                # "caught" = algorithm says NOT in G = FAIL = False
                caught_a = not ra.get(i, True)
                caught_b = not rb.get(i, True)
                if caught_a and caught_b:
                    pair_stats[(a, b)]["both"] += 1
                elif caught_a:
                    pair_stats[(a, b)]["only_a"] += 1
                elif caught_b:
                    pair_stats[(a, b)]["only_b"] += 1
                else:
                    pair_stats[(a, b)]["neither"] += 1

    rows = []
    for (a, b), stats in pair_stats.items():
        rows.append({"algo_a": a, "algo_b": b, **stats})

    out_path = os.path.join(OUT_DIR, "table_05_pairwise.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo_a","algo_b","only_a","only_b","both","neither"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
