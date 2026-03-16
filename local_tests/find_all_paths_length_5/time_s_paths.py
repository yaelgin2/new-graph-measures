import json
import os
import logging
import time

import networkx as nx

from .PathMotifCalculator import PathMotifCalculator


# ---------------- CONFIG ---------------- #

BASE_DIR = os.path.join(os.getcwd(), "local_tests")

PATH_BASE_DIR   = os.path.join(BASE_DIR, "find_all_paths_length_5")
LOG_DIR         = os.path.join(PATH_BASE_DIR, "logs")
COMPARE_LOG_DIR = os.path.join(LOG_DIR, "compare_results")

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


def compute_s_paths_fresh(s_path):
    """Compute S paths without cache - for accurate timing."""
    S = read_graph_file(s_path)
    calc = PathMotifCalculator(S, directed=False)
    return calc.build()


# ---------------- MAIN ---------------- #

def run_for_degree(deg):

    TIMES_LOG = os.path.join(
        COMPARE_LOG_DIR,
        f"s_times_paths_deg_{deg}.log"
    )

    times_logger = logging.getLogger(f"s_times_paths_deg_{deg}")
    times_logger.setLevel(logging.INFO)
    times_logger.handlers.clear()

    handler = logging.FileHandler(TIMES_LOG, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    times_logger.addHandler(handler)

    for color_distribution in ["uniform", "average", "rare"]:

        INPUT_DIR = os.path.join(
            BASE_DIR,
            f"input_color_{color_distribution}_deg_{deg}"
        )

        for i in range(1, 101):

            S_path = os.path.join(INPUT_DIR, f"S_{i}.json")

            if not os.path.exists(S_path):
                continue

            start  = time.perf_counter()
            compute_s_paths_fresh(S_path)
            S_time = time.perf_counter() - start

            times_logger.info(
                f"{INPUT_DIR} | S_{i} time={S_time:.6f}s"
            )

            print(f"Done deg {deg} | {color_distribution} | S_{i}")


def main():

    # Run separately for deg 3 and deg 5
    run_for_degree(8)
    run_for_degree(15)


if __name__ == "__main__":
    main()