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
LOG_DIR         = os.path.join(BASE_DIR, "non_induced", "logs")
COMPARE_LOG_DIR = os.path.join(LOG_DIR, "compare_results")

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

def preprocess_motifs_for_non_induced(motif_size, motifs, motif_graph):
    keys_to_add = {}
    for motif in motifs:
        motif_number = motif >> (8 * motif_size)
        colors_bits  = motif % (1 << (8 * motif_size))
        color_array  = [((colors_bits >> (8 * (motif_size - 1 - i))) % (1 << 8)) for i in range(motif_size)]

        for _, v, data in motif_graph.out_edges(motif_number, data=True):
            for permutation in data["permutations"]:
                color_perm = 0
                for i in range(len(permutation)):
                    color_perm += color_array[i] << ((motif_size - 1 - permutation[i]) * 8)
                perm_motif_num = (v << (8 * motif_size)) + color_perm

                if perm_motif_num not in motifs:
                    keys_to_add[perm_motif_num] = keys_to_add.get(perm_motif_num, 0) + 1
                else:
                    motifs[perm_motif_num] += motifs[motif]
    motifs.update(keys_to_add)
    
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
    return calc.build().get(MotifsNodeCalculator.MOTIF_SUM_KEY, 0)


# ---------------- MAIN ---------------- #

def main():
    TIMES_LOG   = os.path.join(COMPARE_LOG_DIR, "s_times_non_induced_5.log")

    times_logger = logging.getLogger("s_times_induced")
    times_logger.setLevel(logging.INFO)
    if not times_logger.handlers:
        h = logging.FileHandler(TIMES_LOG)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        times_logger.addHandler(h)

    with open(f"local_tests/non_induced/create_inclusion_motifs_dag/{MOTIF_SIZE}_undirected_colored_dag", "rb") as f:
        motif_graph = pickle.load(f)
        
    for color_distribution in ["uniform", "average", "rare"]:

        INPUT_DIR = os.path.join(BASE_DIR, f"input_color_{color_distribution}_deg_5")

        for i in range(1, 1001):
            S_path = os.path.join(INPUT_DIR, f"S_{i}.json")

            start  = time.perf_counter()
            s_motifs = compute_s_motifs_fresh(S_path)
            preprocess_motifs_for_non_induced(MOTIF_SIZE, s_motifs, motif_graph)
            S_time = time.perf_counter() - start

            times_logger.info(f"{INPUT_DIR} | S_{i} time={S_time:.4f}s")
            print(f"Done {INPUT_DIR} S_{i}")


if __name__ == "__main__":
    main()