import json
import os
import pickle
import logging
import time

import networkx as nx

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger


# ---------------- CONFIG ---------------- #

BASE_DIR        = "/home/cohent59/new-graph-measures/local_tests"
REAL_GRAPHS_DIR = os.path.join(BASE_DIR, "real_graphs")
LOG_DIR         = os.path.join(BASE_DIR, "non_induced", "logs", "compare_results")

os.makedirs(LOG_DIR, exist_ok=True)

S_DIR = os.path.join(REAL_GRAPHS_DIR, "NCI109_subgraphs")

GRAPHS = [
    {"name": "Mutagenicity", "s_count": 1000, "s_index_start": 0},
    {"name": "DHFR-MD",      "s_count": 1000, "s_index_start": 0},
]

CONFIGURATION = {
    "colored_directed_variations_3":   "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_directed_colored.pkl",
    "colored_undirected_variations_3": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_undirected_colored.pkl",
    "colored_directed_variations_4":   "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_directed_colored.pkl",
    "colored_undirected_variations_4": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_undirected_colored.pkl",
}

MOTIF_SIZE    = 4
MOTIF_DAG     = "/home/cohent59/new-graph-measures/local_tests/non_induced/create_inclusion_motifs_dag/4_undirected_colored_dag"


# ---------------- LOADERS ---------------- #

def load_single_graph(folder, graph_name):
    G = nx.Graph()
    edges_path  = os.path.join(folder, f"{graph_name}.edges")
    labels_path = os.path.join(folder, f"{graph_name}.node_labels")
    with open(labels_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if "," in line:
                node_id, label = map(int, line.split(","))
            elif " " in line:
                node_id, label = map(int, line.split())
            else:
                node_id, label = i + 1, int(line)
            G.add_node(node_id, color=label)
    with open(edges_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split("," if "," in line else None))
            G.add_edge(u, v)
    return G


def read_graph_file(filename):
    G = nx.Graph()
    with open(filename) as f:
        data = json.load(f)
    for node in data["nodes"]:
        G.add_node(node["id"], color=node["color"])
    for edge in data["links"]:
        G.add_edge(edge["source"], edge["target"])
    return G


def compute_motifs(G):
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
    return calc.build()[MotifsNodeCalculator.MOTIF_SUM_KEY]


def preprocess_motifs_for_non_induced(motif_size, motifs, motif_graph):
    keys_to_add = {}
    for motif in list(motifs.keys()):
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
                    keys_to_add[perm_motif_num] = keys_to_add.get(perm_motif_num, 0) + motifs[motif]
                else:
                    motifs[perm_motif_num] += motifs[motif]
    motifs.update(keys_to_add)


def make_logger(name, filepath):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.FileHandler(filepath)
        h.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(h)
    return logger


# ---------------- MAIN ---------------- #

def main():
    with open(MOTIF_DAG, "rb") as f:
        motif_graph = pickle.load(f)

    summary_logger = make_logger("real_graphs_non_induced_summary",
                                 os.path.join(LOG_DIR, "real_graphs_summary.log"))
    times_logger   = make_logger("real_graphs_non_induced_times",
                                 os.path.join(LOG_DIR, "real_graphs_times.log"))

    for graph_cfg in GRAPHS:
        name    = graph_cfg["name"]
        s_count = graph_cfg["s_count"]
        s_start = graph_cfg["s_index_start"]

        graph_logger = make_logger(f"non_induced_{name}",
                                   os.path.join(LOG_DIR, f"{name}.log"))
        graph_logger.info(f"==== START {name} ====")

        # ---------- COMPUTE G ----------
        graph_logger.info("Computing G motifs")
        total_start = time.perf_counter()
        G = load_single_graph(REAL_GRAPHS_DIR, name)

        start    = time.perf_counter()
        g_motifs = compute_motifs(G)
        preprocess_motifs_for_non_induced(MOTIF_SIZE, g_motifs, motif_graph)
        G_time   = time.perf_counter() - start

        times_logger.info(f"{name} | G_compute_time={G_time:.4f}s")
        graph_logger.info(f"G motifs computed in {G_time:.4f}s")

        # ---------- PROCESS S ----------
        false_pos = 0
        processed = 0

        for i in range(s_start, s_start + s_count):
            S_path = os.path.join(S_DIR, f"S_{i}.json")
            if not os.path.exists(S_path):
                graph_logger.warning(f"Missing S_{i}.json")
                continue

            S        = read_graph_file(S_path)
            s_motifs = compute_motifs(S)
            preprocess_motifs_for_non_induced(MOTIF_SIZE, s_motifs, motif_graph)

            feasible = all(g_motifs.get(m, 0) >= cnt for m, cnt in s_motifs.items())
            graph_logger.info(f"SUM {'PASS' if feasible else 'FAIL'} S_{i}")

            if feasible:
                false_pos += 1

            processed += 1
            if processed % 50 == 0:
                graph_logger.info(f"Progress: {processed}/{s_count}")

        total_time = time.perf_counter() - total_start
        s_time     = total_time - G_time

        times_logger.info(f"{name} | G_compute_time={G_time:.4f}s")
        times_logger.info(f"{name} | S_total_time={s_time:.4f}s")
        times_logger.info(f"{name} | TOTAL_TIME={total_time:.4f}s")

        graph_logger.info(f"==== FINISHED {name} ====")
        graph_logger.info(f"Processed: {processed}, False positives: {false_pos}")
        summary_logger.info(
            f"{name} | processed={processed} | false_positives={false_pos} "
            f"| G_time={G_time:.4f}s | S_time={s_time:.4f}s | total_time={total_time:.4f}s"
        )

        print(f"{name}: processed={processed}, false_positives={false_pos}")


if __name__ == "__main__":
    main()