import json
import os
import pickle
import logging

import networkx as nx

from .PathMotifCalculator import PathMotifCalculator


# ---------------- CONFIG ---------------- #

BASE_DIR  = os.path.join(os.getcwd(), "local_tests")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs_by_density_3")

PATH_BASE_DIR   = os.path.join(BASE_DIR, "find_all_paths_length_5")
PICKLE_DIR      = os.path.join(PATH_BASE_DIR, "cache")
LOG_DIR         = os.path.join(PATH_BASE_DIR, "logs")

os.makedirs(PICKLE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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


# ---------------- MAIN ---------------- #

def main():

    SUMMARY_LOG = os.path.join(LOG_DIR, "summary_paths_different_distributions_3.log")

    summary_logger = logging.getLogger("summary_paths_different_distributions_3")
    summary_logger.setLevel(logging.INFO)
    if not summary_logger.handlers:
        handler = logging.FileHandler(SUMMARY_LOG)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        summary_logger.addHandler(handler)

    for graph_avg_neighs in [5, 8, 10, 13, 15]:
        for color_distribution in ['uniform', 'average', 'rare']:
            INPUT_DIR = os.path.join(BASE_DIR, f"input_color_{color_distribution}_deg_3")
            avg_false_positives = 0

            for j in range(10):
                graph_file_name = f"g_den_{graph_avg_neighs}_embedded_den_3_{color_distribution}_{j}"
                LOG_FILE        = os.path.join(LOG_DIR, f"{graph_file_name}.log")

                # Skip if already computed
                if os.path.exists(LOG_FILE):
                    print(f"Skipping {graph_file_name} (log exists)")
                    # Still need to read false positives for average calculation
                    fp = 0
                    with open(LOG_FILE) as f:
                        for line in f:
                            if "PATH PASS S_" in line:
                                s_num = int(line.strip().split("S_")[1])
                                if s_num > 10:
                                    fp += 1
                    avg_false_positives += fp
                    continue

                logger = logging.getLogger(graph_file_name)
                logger.setLevel(logging.INFO)
                if not logger.handlers:
                    handler = logging.FileHandler(LOG_FILE)
                    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
                    logger.addHandler(handler)

                G_PICKLE = os.path.join(PICKLE_DIR, f"{graph_file_name}.pkl")

                # ---------- LOAD OR COMPUTE G ----------
                if os.path.exists(G_PICKLE):
                    with open(G_PICKLE, "rb") as f:
                        g_paths = pickle.load(f)
                    print(f"Loaded cached G paths for {graph_file_name}")
                else:
                    G = read_graph_file(os.path.join(GRAPH_DIR, f"{graph_file_name}.json"))
                    g_paths = PathMotifCalculator(G, False).build()
                    with open(G_PICKLE, "wb") as f:
                        pickle.dump(g_paths, f)
                    print(f"Computed and cached G paths for {graph_file_name}")

                false_pos_sum_only = 0

                # ---------- S PROCESS ----------
                for i in range(1, 101):
                    S_path = os.path.join(INPUT_DIR, f"S_{i}.json")
                    S      = read_graph_file(S_path)
                    s_paths = PathMotifCalculator(S, False).build()

                    feasible_sum = all(g_paths.get(m, 0) >= cnt for m, cnt in s_paths.items())

                    if i > 10 and feasible_sum:
                        false_pos_sum_only += 1

                    logger.info(f"PATH {'PASS' if feasible_sum else 'FAIL'} S_{i}")
                    print(f"Done S_{i}")

                summary_logger.info(f"{graph_file_name} | path_only={false_pos_sum_only}")
                avg_false_positives += false_pos_sum_only

            avg_false_positives /= 10
            summary_logger.info(f"g_den_{graph_avg_neighs}_embedded_den_3_{color_distribution} | AVERAGE path_only={avg_false_positives}")


if __name__ == "__main__":
    main()