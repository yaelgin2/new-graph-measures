import json
import os
import pickle
import logging
import time

import networkx as nx

from .PathMotifCalculator import PathMotifCalculator


# ---------------- CONFIG ---------------- #

CACHE = True  # If True, load from cache when available. Always saves to cache.

BASE_DIR   = os.path.join(os.getcwd(), "local_tests")
GRAPH_DIR  = os.path.join(BASE_DIR, "graphs_by_density_3")

PATH_BASE_DIR   = os.path.join(BASE_DIR, "find_all_paths_length_5")
CACHE_DIR       = os.path.join(PATH_BASE_DIR, "cache")
LOG_DIR         = os.path.join(PATH_BASE_DIR, "logs")
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


def compute_and_cache_paths(graph_path, cache_key):
    """Compute paths for graph at graph_path, save to cache, return (paths, time)."""
    G = read_graph_file(graph_path)
    start  = time.perf_counter()
    paths  = PathMotifCalculator(G, False).build()
    elapsed = time.perf_counter() - start
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(paths, f)
    return paths, elapsed


def get_paths(graph_path, cache_key):
    """Load from cache if CACHE=True and cache exists, otherwise compute. Always saves."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if CACHE and os.path.exists(cache_file):
        start = time.perf_counter()
        with open(cache_file, "rb") as f:
            paths = pickle.load(f)
        elapsed = time.perf_counter() - start
        print(f"Loaded cached paths for {cache_key}")
        return paths, elapsed
    paths, elapsed = compute_and_cache_paths(graph_path, cache_key)
    print(f"Computed and cached paths for {cache_key}")
    return paths, elapsed


# ---------------- MAIN ---------------- #

def main():

    color_distribution = "average"
    graph_avg_neighs   = 5

    INPUT_DIR       = os.path.join(BASE_DIR, f"input_color_{color_distribution}_deg_3")
    graph_file_name = f"g_den_{graph_avg_neighs}_embedded_den_3_{color_distribution}_0"
    GRAPH_PATH      = os.path.join(GRAPH_DIR, f"{graph_file_name}.json")
    LOG_FILE        = os.path.join(COMPARE_LOG_DIR, f"{graph_file_name}.log")

    total_start = time.perf_counter()

    # ---------- COMPUTE OR LOAD G PATHS ----------
    g_paths, G_time = get_paths(GRAPH_PATH, graph_file_name)
    print(f"G paths done in {G_time:.4f}s")

    # ---------- PROCESS S ----------
    false_positive_paths = 0

    for i in range(1, 11):
        S_path = os.path.join(INPUT_DIR, f"S_{i}.json")
        if not os.path.exists(S_path):
            continue

        S = read_graph_file(S_path)
        s_paths = PathMotifCalculator(S, False).build()

        feasible = True
        for m, cnt in s_paths.items():
            if g_paths.get(m, 0) < cnt:
                print(f"Feasibility broken for S_{i}: m={m}, cnt={cnt}, g={g_paths.get(m,0)}")
                exit(1)
                feasible = False
                break

        if feasible:
            false_positive_paths += 1

    # ---------- TOTAL TIME ----------
    total_time = time.perf_counter() - total_start
    print(f"Finished {graph_file_name} | G_time={G_time:.4f}s | total={total_time:.4f}s")
    print(f"False positives (paths): {false_positive_paths}")


if __name__ == "__main__":
    main()