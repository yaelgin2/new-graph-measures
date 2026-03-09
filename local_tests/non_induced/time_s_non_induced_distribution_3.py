import json
import os
import pickle
import logging
import time

import networkx as nx

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger

# ---------------- CONFIG ---------------- #

BASE_DIR   = os.path.join(os.getcwd(), "local_tests")
GRAPH_DIR  = os.path.join(BASE_DIR, "graphs_by_density_3")
PICKLE_DIR = os.path.join(BASE_DIR, "non_induced", "cache")
LOG_DIR    = os.path.join(BASE_DIR, "non_induced", "logs")
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

def compute_motifs(path):
    G = read_graph_file(path)
    calc = MotifsNodeCalculator(
        graph=G,
        colores_loaded=True,
        configuration=CONFIGURATION,
        level=MOTIF_SIZE,
        calc_nodes=False,
        calc_edges=False,
        count_motifs=True,
        logger=PrintLogger(),
    )
    return calc.build()


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


# ---------------- MAIN ---------------- #

def main():
    TIMES_LOG_FILE   = os.path.join(COMPARE_LOG_DIR, "times_3.log")
    SUMMARY_LOG_FILE = os.path.join(COMPARE_LOG_DIR, "summary_3.log")

    summary_logger = logging.getLogger("non_induced_summary")
    summary_logger.setLevel(logging.INFO)
    if not summary_logger.handlers:
        h = logging.FileHandler(SUMMARY_LOG_FILE)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        summary_logger.addHandler(h)

    times_logger = logging.getLogger("non_induced_times")
    times_logger.setLevel(logging.INFO)
    if not times_logger.handlers:
        h = logging.FileHandler(TIMES_LOG_FILE)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        times_logger.addHandler(h)

    with open(f"local_tests/non_induced/create_inclusion_motifs_dag/{MOTIF_SIZE}_undirected_colored_dag", "rb") as f:
        motif_graph = pickle.load(f)

    for graph_avg_neighs in [5, 8, 10, 13, 15]:
        for color_distribution in ["uniform", "average", "rare"]:

            INPUT_DIR = os.path.join(BASE_DIR, f"input_color_{color_distribution}_deg_3")

            graph_file_name = f"g_den_{graph_avg_neighs}_embedded_den_3_{color_distribution}_0"
            LOG_FILE = os.path.join(COMPARE_LOG_DIR, f"{graph_file_name}.log")

            logger = logging.getLogger(graph_file_name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                h = logging.FileHandler(LOG_FILE)
                h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
                logger.addHandler(h)

            total_G_start = time.perf_counter()

            G_PICKLE = os.path.join(PICKLE_DIR, f"{graph_file_name}.pkl")

            G = read_graph_file(os.path.join(GRAPH_DIR, f"{graph_file_name}.json"))
            start = time.perf_counter()
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
            g_sum = g_calc.build()[MotifsNodeCalculator.MOTIF_SUM_KEY]
            preprocess_motifs_for_non_induced(MOTIF_SIZE, g_sum, motif_graph)
            G_time = time.perf_counter() - start
                
            with open(G_PICKLE, "wb") as f:
                pickle.dump(g_sum, f)

            times_logger.info(f"{graph_file_name} | G_compute_time={G_time:.4f}s")

            false_pos_sum_only = 0

            for i in range(1, 1001):
                S_path = os.path.join(INPUT_DIR, f"S_{i}.json")

                s_motifs = compute_motifs(S_path)
                s_motifs = s_motifs[MotifsNodeCalculator.MOTIF_SUM_KEY]
                preprocess_motifs_for_non_induced(MOTIF_SIZE, s_motifs, motif_graph)

                feasible_sum = all(g_sum.get(m, 0) >= cnt for m, cnt in s_motifs.items())

                if i > 10 and feasible_sum:
                    false_pos_sum_only += 1

                logger.info(f"S_{i} {'PASS' if feasible_sum else 'FAIL'}")

            total_time = time.perf_counter() - total_G_start
            times_logger.info(f"{graph_file_name} | TOTAL_BATCH_TIME={total_time:.4f}s")
            summary_logger.info(f"{graph_file_name} | sum_only={false_pos_sum_only}")


if __name__ == "__main__":
    main()