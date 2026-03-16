import json
import os
import logging
import time

import networkx as nx

from .PathMotifCalculator import PathMotifCalculator


# ---------------- CONFIG ---------------- #

BASE_DIR        = os.path.join(os.getcwd(), "local_tests")
COMPARE_LOG_DIR = os.path.join(BASE_DIR, "find_all_paths_length_5", "logs", "compare_results")

os.makedirs(COMPARE_LOG_DIR, exist_ok=True)


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


def compute_paths(path):
    G = read_graph_file(path)
    calc = PathMotifCalculator(G, directed=False)
    return calc.build()


# ---------------- MAIN ---------------- #

def main():

    SUMMARY_LOG = os.path.join(COMPARE_LOG_DIR, "summary_paths_equal_degs.log")
    TIMES_LOG   = os.path.join(COMPARE_LOG_DIR, "equal_degs_times_paths.log")

    summary_logger = logging.getLogger("summary_paths")
    summary_logger.setLevel(logging.INFO)
    if not summary_logger.handlers:
        h = logging.FileHandler(SUMMARY_LOG)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        summary_logger.addHandler(h)

    times_logger = logging.getLogger("equal_degs_times_paths")
    times_logger.setLevel(logging.INFO)
    if not times_logger.handlers:
        h = logging.FileHandler(TIMES_LOG)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        times_logger.addHandler(h)

    for graph_avg_neighs in [3, 5, 8, 15]:
        for color_distribution in ["uniform", "average", "rare"]:

            run_name  = f"color_{color_distribution}_deg_{graph_avg_neighs}"
            LOG_FILE  = os.path.join(COMPARE_LOG_DIR, f"paths_{run_name}.log")

            # Skip if already computed
            if os.path.exists(LOG_FILE):
                print(f"Skipping {run_name} (log exists)")
                continue

            INPUT_DIR = os.path.join(BASE_DIR, f"input_{run_name}")
            run_log   = os.path.join(COMPARE_LOG_DIR, f"paths_{run_name}.log")

            logger = logging.getLogger(f"paths_{run_name}")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                h = logging.FileHandler(run_log)
                h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
                logger.addHandler(h)

            logger.info(f"Starting run {run_name}")
            total_start = time.perf_counter()

            # -------- COMPUTE G PATHS --------
            start   = time.perf_counter()
            g_paths = compute_paths(os.path.join(INPUT_DIR, "G_non_induced.json"))
            G_time  = time.perf_counter() - start

            times_logger.info(f"{run_name} | G_compute_time={G_time:.4f}s")

            false_pos_sum_only = 0

            # -------- PROCESS S --------
            for i in range(1, 1001):
                s_paths = compute_paths(os.path.join(INPUT_DIR, f"S_{i}.json"))

                feasible = all(g_paths.get(m, 0) >= cnt for m, cnt in s_paths.items())

                logger.info(f"PATH {'PASS' if feasible else 'FAIL'} S_{i}")

                if feasible:
                    false_pos_sum_only += 1

                print(f"Done S_{i}")

            # -------- TOTAL TIME --------
            total_time = time.perf_counter() - total_start
            times_logger.info(f"{run_name} | TOTAL_TIME={total_time:.4f}s")

            print("False positives (paths only):", false_pos_sum_only)
            summary_logger.info(f"{run_name} | paths_only={false_pos_sum_only}")


if __name__ == "__main__":
    main()