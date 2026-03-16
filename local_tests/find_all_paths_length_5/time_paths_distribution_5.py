import json
import os
import pickle
import logging
import time

import networkx as nx

from .PathMotifCalculator import PathMotifCalculator


# ---------------- CONFIG ---------------- #

BASE_DIR   = os.path.join(os.getcwd(), "local_tests")
GRAPH_DIR  = os.path.join(BASE_DIR, "graphs_by_density_5")

# 🔹 NEW base folder
PATH_BASE_DIR = os.path.join(BASE_DIR, "find_all_paths_length_5")

CACHE_DIR = os.path.join(PATH_BASE_DIR, "cache")
LOG_DIR   = os.path.join(PATH_BASE_DIR, "logs")
COMPARE_LOG_DIR = os.path.join(LOG_DIR, "compare_results")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(COMPARE_LOG_DIR, exist_ok=True)


# ---------------- HELPERS ---------------- #

def read_graph_file(filename):
    G = nx.Graph()
    with open(filename) as f:
        data = json.load(f)

    for node in data["nodes"]:
        G.add_node(node["id"], color=node["color"])

    for edge in data["links"]:
        G.add_edge(edge["source"], edge["target"])

    return G


def compute_paths(path):
    G = read_graph_file(path)
    return PathMotifCalculator(G).build()


# ---------------- MAIN ---------------- #

def main():

    TIMES_LOG_FILE   = os.path.join(COMPARE_LOG_DIR, "times_paths_5.log")
    SUMMARY_LOG_FILE = os.path.join(COMPARE_LOG_DIR, "summary_paths_5.log")

    summary_logger = logging.getLogger("paths_summary")
    summary_logger.setLevel(logging.INFO)
    if not summary_logger.handlers:
        h = logging.FileHandler(SUMMARY_LOG_FILE)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        summary_logger.addHandler(h)

    times_logger = logging.getLogger("paths_times")
    times_logger.setLevel(logging.INFO)
    if not times_logger.handlers:
        h = logging.FileHandler(TIMES_LOG_FILE)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        times_logger.addHandler(h)

    # Same iteration structure as your motif script
    for graph_avg_neighs in [8, 10, 13, 15]:
        for color_distribution in ["uniform", "average", "rare"]:

            INPUT_DIR = os.path.join(BASE_DIR, f"input_color_{color_distribution}_deg_5")

            graph_file_name = f"g_den_{graph_avg_neighs}_embedded_den_5_{color_distribution}_0"
            GRAPH_PATH = os.path.join(GRAPH_DIR, f"{graph_file_name}.json")

            LOG_FILE = os.path.join(COMPARE_LOG_DIR, f"{graph_file_name}.log")

            logger = logging.getLogger(graph_file_name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                h = logging.FileHandler(LOG_FILE)
                h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
                logger.addHandler(h)

            total_G_start = time.perf_counter()

            # ---------- COMPUTE OR LOAD G PATHS ----------

            G = read_graph_file(GRAPH_PATH)

            start = time.perf_counter()
            g_paths = PathMotifCalculator(G, False).build()
            G_time = time.perf_counter() - start


            print(f"Computed G paths for {graph_file_name}")

            times_logger.info(f"{graph_file_name} | G_compute_time={G_time:.4f}s")

            # ---------- PROCESS S ----------

            false_positive_paths = 0

            for i in range(1, 1001):

                S_path = os.path.join(INPUT_DIR, f"S_{i}.json")

                if not os.path.exists(S_path):
                    continue

                S = read_graph_file(S_path)
                s_paths = PathMotifCalculator(S, False).build()

                feasible = all(
                    g_paths.get(m, 0) >= cnt
                    for m, cnt in s_paths.items()
                )

                if i > 10 and feasible:
                    false_positive_paths += 1

                logger.info(f"S_{i} {'PASS' if feasible else 'FAIL'}")

            # ---------- TOTAL TIME ----------

            total_time = time.perf_counter() - total_G_start

            times_logger.info(
                f"{graph_file_name} | TOTAL_BATCH_TIME={total_time:.4f}s"
            )

            summary_logger.info(
                f"{graph_file_name} | path_only_false_pos={false_positive_paths}"
            )

            print(f"Finished {graph_file_name}")


if __name__ == "__main__":
    main()