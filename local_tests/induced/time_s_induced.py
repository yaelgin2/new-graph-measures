import json
import os
import pickle
import logging
import time

import networkx as nx

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger
from local_tests.induced.s_motif_cache import get_cached_S_motifs

# ---------------- CONFIG ---------------- #

BASE_DIR        = os.path.join(os.getcwd(), "local_tests")
PICKLE_DIR      = os.path.join(BASE_DIR, "induced", "cache")
LOG_DIR         = os.path.join(BASE_DIR, "induced", "logs")
COMPARE_LOG_DIR = os.path.join(LOG_DIR, "compare_results")

os.makedirs(PICKLE_DIR, exist_ok=True)
os.makedirs(COMPARE_LOG_DIR, exist_ok=True)

CONFIGURATION = {
    "colored_directed_variations_3":   "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_directed_colored.pkl",
    "colored_undirected_variations_3": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_undirected_colored.pkl",
    "colored_directed_variations_4":   "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_directed_colored.pkl",
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


def compute_s_motifs_fresh(s_path):
    """Compute S motifs without cache - for accurate timing."""
    S = read_graph_file(s_path)
    calc = MotifsNodeCalculator(
        graph=S,
        colores_loaded=True,
        configuration=CONFIGURATION,
        level=MOTIF_SIZE,
        calc_nodes=False,
        calc_edges=False,
        count_motifs=True,
    )
    return calc.build()


# ---------------- MAIN ---------------- #

def main():
    TIMES_LOG   = os.path.join(COMPARE_LOG_DIR, "s_times_induced_5.log")

    times_logger = logging.getLogger("s_times_induced")
    times_logger.setLevel(logging.INFO)
    if not times_logger.handlers:
        h = logging.FileHandler(TIMES_LOG)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        times_logger.addHandler(h)

    for color_distribution in ["uniform", "average", "rare"]:

        INPUT_DIR = os.path.join(BASE_DIR, f"input_color_{color_distribution}_deg_5")

        for i in range(1, 1001):
            S_path = os.path.join(INPUT_DIR, f"S_{i}.json")

            start  = time.perf_counter()
            compute_s_motifs_fresh(S_path)
            S_time = time.perf_counter() - start

            times_logger.info(f"{INPUT_DIR} | S_{i} time={S_time:.4f}s")
            print(f"Done {INPUT_DIR} S_{i}")


if __name__ == "__main__":
    main()