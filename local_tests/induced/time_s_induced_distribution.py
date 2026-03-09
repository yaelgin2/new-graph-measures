import json
import os
import logging
import time

import networkx as nx

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger

# ---------------- CONFIG ---------------- #

BASE_DIR        = os.path.join(os.getcwd(), "local_tests")
GRAPH_DIR       = os.path.join(BASE_DIR, "graphs_by_density_3")
LOG_ROOT        = os.path.join(BASE_DIR, "induced", "logs")
COMPARE_LOG_DIR = os.path.join(LOG_ROOT, "compare_results")

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


# ---------------- MAIN ---------------- #

def main():
    TIMES_LOG_FILE   = os.path.join(COMPARE_LOG_DIR, "times.log")
    SUMMARY_LOG_FILE = os.path.join(COMPARE_LOG_DIR, "summary.log")

    summary_logger = logging.getLogger("induced_summary")
    summary_logger.setLevel(logging.INFO)
    if not summary_logger.handlers:
        h = logging.FileHandler(SUMMARY_LOG_FILE)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        summary_logger.addHandler(h)

    times_logger = logging.getLogger("induced_times")
    times_logger.setLevel(logging.INFO)
    if not times_logger.handlers:
        h = logging.FileHandler(TIMES_LOG_FILE)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        times_logger.addHandler(h)

    for graph_avg_neighs in [5, 8, 10, 13, 15]:
        for color_distribution in ["uniform", "average", "rare"]:

            INPUT_DIR       = os.path.join(BASE_DIR, f"input_color_{color_distribution}_deg_3")
            graph_file_name = f"g_den_{graph_avg_neighs}_embedded_den_3_{color_distribution}_0"
            LOG_FILE        = os.path.join(COMPARE_LOG_DIR, f"{graph_file_name}.log")

            logger = logging.getLogger(graph_file_name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                h = logging.FileHandler(LOG_FILE)
                h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
                logger.addHandler(h)

            total_G_start = time.perf_counter()

            start    = time.perf_counter()
            g_motifs = compute_motifs(os.path.join(GRAPH_DIR, f"{graph_file_name}.json"))
            G_time   = time.perf_counter() - start
            times_logger.info(f"{graph_file_name} | G_compute_time={G_time:.4f}s")

            g_sum = g_motifs.get(MotifsNodeCalculator.MOTIF_SUM_KEY)

            false_pos_sum_only = 0

            for i in range(1, 1001):
                s_motifs = compute_motifs(os.path.join(INPUT_DIR, f"S_{i}.json"))
                s_motifs = s_motifs[MotifsNodeCalculator.MOTIF_SUM_KEY]

                feasible_sum = all(g_sum.get(m, 0) >= cnt for m, cnt in s_motifs.items())
                if i > 10 and feasible_sum:
                    false_pos_sum_only += 1
                logger.info(f"S_{i} {'PASS' if feasible_sum else 'FAIL'}")
                print(f"Done S_{i}")

            total_time = time.perf_counter() - total_G_start
            times_logger.info(f"{graph_file_name} | TOTAL_BATCH_TIME={total_time:.4f}s")
            summary_logger.info(f"{graph_file_name} | sum_only={false_pos_sum_only}")


if __name__ == "__main__":
    main()