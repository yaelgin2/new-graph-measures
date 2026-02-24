import json
import os
import pickle
import logging
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
import itertools

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger

# ---------------- CONFIG ---------------- #

BASE_DIR = os.path.join(os.getcwd(), "local_tests")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs_by_density_3")
PICKLE_DIR = os.path.join(BASE_DIR, "induced", "cache")
LOG_DIR = os.path.join(BASE_DIR, "induced", "logs")
FANMOD_DIR = os.path.join(BASE_DIR, "other_algorithms_tests", "fanmod_plus")

os.makedirs(PICKLE_DIR, exist_ok=True)

CONFIGURATION = {
    "colored_directed_variations_3": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_directed_colored.pkl",
    "colored_undirected_variations_3": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_undirected_colored.pkl",
    "colored_directed_variations_4": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_directed_colored.pkl",
    "colored_undirected_variations_4": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_undirected_colored.pkl",
}

MOTIF_SIZE = 4


# ---------------- HELPERS ---------------- #

def read_graph_file(filename):
    graph = nx.Graph()
    with open(filename) as f:
        graph_json = json.load(f)

    for node in graph_json["nodes"]:
        graph.add_node(node["id"], color=node["color"])

    for edge in graph_json["links"]:
        graph.add_edge(edge["source"], edge["target"])

    return graph


def load_fanmod_results(graph_file_name):
    """Load FANMOD+ results JSON if it exists. Returns dict {int motif_id: count} or None."""
    path = os.path.join(FANMOD_DIR, f"{graph_file_name}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def compare_with_fanmod(graph_file_name, vdmc_sum, fanmod):
    """
    Compare VDMC motif sum against FANMOD+ results.
    Prints mismatches and returns False if any mismatch found.
    """
    all_motifs = set(vdmc_sum.keys()) | set(fanmod.keys())
    mismatches = []
    for mid in sorted(all_motifs):
        vc = vdmc_sum.get(mid, 0)
        fc = fanmod.get(mid, 0)
        if vc != fc:
            mismatches.append((mid, fc, vc))

    if mismatches:
        print(f"\n  ✗ FANMOD+ vs VDMC MISMATCH in {graph_file_name}:")
        print(f"  {'Motif ID':<20} {'FANMOD+':>10} {'VDMC':>10} {'Diff':>10}")
        print(f"  {'-'*52}")
        for mid, fc, vc in mismatches[:20]:
            print(f"  {mid:<20} {fc:>10} {vc:>10} {fc - vc:>+10}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
        return False

    print(f"  ✓ FANMOD+ match confirmed for {graph_file_name}")
    return True


# ---------------- MAIN ---------------- #

def main():
    SUMMARY_LOG = os.path.join(LOG_DIR, "summary_induced_different_distributions_3.log")

    summary_logger = logging.getLogger("summary_induced_different_distributions_3")
    summary_logger.setLevel(logging.INFO)
    summary_handler = logging.FileHandler(SUMMARY_LOG)
    summary_logger.addHandler(summary_handler)

    for graph_avg_neighs in [5, 8, 10, 13, 15]:
        for color_distribution in ['uniform', 'average', 'rare']:

            INPUT_DIR = os.path.join(BASE_DIR, f"input_color_{color_distribution}_deg_3")

            avg_false_positives = 0

            for j in range(10):
                # ---------------- LOGGING ---------------- #

                graph_file_name = f'g_den_{graph_avg_neighs}_embedded_den_3_{color_distribution}_{j}'

                LOG_FILE = os.path.join(LOG_DIR, f"{graph_file_name}.log")

                logging.basicConfig(
                    filename=LOG_FILE,
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s",
                )

                G_PICKLE = os.path.join(PICKLE_DIR, f"{graph_file_name}.pkl")

                # ----- Load or compute G motifs -----
                if os.path.exists(G_PICKLE):
                    with open(G_PICKLE, "rb") as f:
                        g_motifs = pickle.load(f)
                    print(f"Loaded cached G motifs for {graph_file_name}")
                else:
                    G = read_graph_file(os.path.join(GRAPH_DIR, f"{graph_file_name}.json"))
                    g_calc = MotifsNodeCalculator(
                        graph=G,
                        colores_loaded=True,
                        configuration=CONFIGURATION,
                        level=MOTIF_SIZE,
                        calc_nodes=False,
                        calc_edges=False,
                        count_motifs=True,
                        logger=PrintLogger(),
                    )

                    g_motifs = g_calc.build()

                    with open(G_PICKLE, "wb") as f:
                        pickle.dump(g_motifs, f)

                    print(f"Computed and cached G motifs for {graph_file_name}")

                g_sum = g_motifs.get(MotifsNodeCalculator.MOTIF_SUM_KEY)
                g_motifs.pop(MotifsNodeCalculator.MOTIF_SUM_KEY)

                # ----- Compare with FANMOD+ if available -----
                fanmod = load_fanmod_results(graph_file_name)
                if fanmod is not None:
                    ok = compare_with_fanmod(graph_file_name, g_sum, fanmod)
                    if not ok:
                        print(f"\nStopping due to mismatch in {graph_file_name}.")
                        return
                else:
                    print(f"  (No FANMOD+ results found for {graph_file_name}, skipping comparison)")

                false_pos_sum_only = 0

                # ----- Process S graphs -----
                for i in range(1, 101):
                    S = read_graph_file(os.path.join(INPUT_DIR, f"S_{i}.json"))

                    s_calc = MotifsNodeCalculator(
                        graph=S,
                        colores_loaded=True,
                        configuration=CONFIGURATION,
                        level=MOTIF_SIZE,
                        calc_nodes=False,
                        calc_edges=False,
                        count_motifs=True,
                    )
                    s_motifs = s_calc.build()[MotifsNodeCalculator.MOTIF_SUM_KEY]

                    # ---------- Stage 1: motif sum check ----------
                    feasible_sum = True
                    for m, cnt in s_motifs.items():
                        if g_sum.get(m, 0) < cnt:
                            feasible_sum = False
                            break

                    if i > 10 and feasible_sum:
                        false_pos_sum_only += 1

                    if not feasible_sum:
                        logging.info(f"SUM FAIL S_{i}")
                        continue

                    print(f"Done S_{i}")

                summary_logger.info(f"{graph_file_name} | sum_only={false_pos_sum_only}")
                avg_false_positives += false_pos_sum_only

            avg_false_positives /= 10
            summary_logger.info(f"g_den_{graph_avg_neighs}_embedded_den_5_{color_distribution} | AVERAGE sum_only={avg_false_positives}")


if __name__ == "__main__":
    main()