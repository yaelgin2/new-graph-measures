import json
import os
import pickle
import logging

import networkx as nx

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger
from local_tests.induced.s_motif_cache import get_cached_S_motifs

# ---------------- CONFIG ---------------- #

BASE_DIR   = os.path.join(os.getcwd(), "local_tests")
GRAPH_DIR  = os.path.join(BASE_DIR, "graphs_by_density_3")
INDUCED_PICKLE_DIR = os.path.join(BASE_DIR, "induced", "cache")
PICKLE_DIR = os.path.join(BASE_DIR, "non_induced", "cache")
LOG_DIR    = os.path.join(BASE_DIR, "non_induced", "logs")

os.makedirs(PICKLE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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

# ---------------- MAIN ---------------- #

def main():
    SUMMARY_LOG = os.path.join(LOG_DIR, "summary_non_induced_different_distributions_3.log")

    summary_logger = logging.getLogger("summary_non_induced_different_distributions_3")
    summary_logger.setLevel(logging.INFO)
    if not summary_logger.handlers:
        h = logging.FileHandler(SUMMARY_LOG)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        summary_logger.addHandler(h)

    with open(f"local_tests/non_induced/create_inclusion_motifs_dag/{MOTIF_SIZE}_undirected_colored_dag", "rb") as f:
        motif_graph = pickle.load(f)

    for graph_avg_neighs in [5, 8, 10, 13, 15]:
        for color_distribution in ["uniform", "average", "rare"]:

            INPUT_DIR = os.path.join(BASE_DIR, f"input_color_{color_distribution}_deg_3")
            avg_false_positives = 0

            for j in range(10):
                graph_file_name = f"g_den_{graph_avg_neighs}_embedded_den_3_{color_distribution}_{j}"
                LOG_FILE = os.path.join(LOG_DIR, f"{graph_file_name}.log")

                logger = logging.getLogger(graph_file_name)
                logger.setLevel(logging.INFO)
                if not logger.handlers:
                    h = logging.FileHandler(LOG_FILE)
                    h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
                    logger.addHandler(h)

                NON_INDUCED_PICKLE = os.path.join(PICKLE_DIR, f"{graph_file_name}.pkl")
                G_PICKLE = os.path.join(INDUCED_PICKLE_DIR, f"{graph_file_name}.pkl")

                if os.path.exists(NON_INDUCED_PICKLE):
                    with open(NON_INDUCED_PICKLE, "rb") as f:
                        g_sum = pickle.load(f)
                    print(f"Loaded cached non induced G motifs for {graph_file_name}")
                else:
                    if os.path.exists(G_PICKLE):
                        with open(G_PICKLE, "rb") as f:
                            g_motif = pickle.load(f)
                            g_sum = g_motif.get(MotifsNodeCalculator.MOTIF_SUM_KEY)
                        print(f"Loaded cached induced G motifs for {graph_file_name}")
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
                        g_sum = g_calc.build()[MotifsNodeCalculator.MOTIF_SUM_KEY]

                        print(f"Computed induced G motifs for {graph_file_name}")

                    preprocess_motifs_for_non_induced(MOTIF_SIZE, g_sum, motif_graph)
                    with open(NON_INDUCED_PICKLE, "wb") as f:
                            pickle.dump(g_sum, f)
                    print(f"Computed non induced G motifs for {graph_file_name}")
                        
                false_pos_sum_only = 0

                for i in range(1, 101):
                    S_path  = os.path.join(INPUT_DIR, f"S_{i}.json")
                    # load induced S motifs from shared cache, then expand for non-induced
                    s_motifs = dict(get_cached_S_motifs(S_path))
                    preprocess_motifs_for_non_induced(MOTIF_SIZE, s_motifs, motif_graph)

                    feasible_sum = all(g_sum.get(m, 0) >= cnt for m, cnt in s_motifs.items())

                    if i > 10 and feasible_sum:
                        false_pos_sum_only += 1

                    logger.info(f"SUM {'PASS' if feasible_sum else 'FAIL'} S_{i}")
                    print(f"Done S_{i}")

                summary_logger.info(f"{graph_file_name} | sum_only={false_pos_sum_only}")
                avg_false_positives += false_pos_sum_only

            avg_false_positives /= 10
            summary_logger.info(
                f"g_den_{graph_avg_neighs}_embedded_den_3_{color_distribution}"
                f" | AVERAGE sum_only={avg_false_positives}"
            )


if __name__ == "__main__":
    main()