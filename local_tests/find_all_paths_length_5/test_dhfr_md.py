import os
import json
import pickle
import logging

import networkx as nx

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger

from .PathMotifCalculator import PathMotifCalculator


# ---------------- CONFIG ---------------- #

BASE_DIR   = "/home/cohent59/new-graph-measures/local_tests"
GRAPH_NAME = "DHFR-MD"

GRAPH_DIR  = os.path.join(BASE_DIR, "real_graphs")
NODES_FILE = os.path.join(GRAPH_DIR, f"{GRAPH_NAME}.node_labels")
EDGES_FILE = os.path.join(GRAPH_DIR, f"{GRAPH_NAME}.edges")

S_DIR = os.path.join(GRAPH_DIR, "NCI109_subgraphs")

G_MOTIF_PICKLE = "/home/cohent59/new-graph-measures/local_tests/non_induced/real_graphs/cache/DHFR-MD_motifs.pkl"
G_PATH_PICKLE  = "/home/cohent59/new-graph-measures/local_tests/find_all_paths_length_5/cache/DHFR-MD_paths.pkl"

LOG_DIR  = os.path.join(BASE_DIR, "find_all_paths_length_5", "logs")
LOG_FILE = os.path.join(LOG_DIR, f"{GRAPH_NAME}_paths.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    force=True
)

CONFIGURATION = {
    "colored_directed_variations_3":
        "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_directed_colored.pkl",
    "colored_undirected_variations_3":
        "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_undirected_colored.pkl",
    "colored_directed_variations_4":
        "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_directed_colored.pkl",
    "colored_undirected_variations_4":
        "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_undirected_colored.pkl",
}

MOTIF_SIZE = 4


# ---------------- HELPERS ---------------- #
def preprocess_motifs_for_non_induced(motif_size, motifs, motif_graph):
    keys_to_add = {}
    for motif in motifs:
        motif_number = motif >> (8 * motif_size)
        colors_bits = motif % (1 << (8 * motif_size))
        color_array = [((colors_bits >> (8 * (motif_size - 1 - i))) % (1 << 8)) for i in range(motif_size)]

        for _, v, data in motif_graph.out_edges(motif_number, data=True):
            for permutation in data["permutations"]:
                color_perm = 0
                    
                for i in range(len(permutation)):
                    color_perm += color_array[i] << ((motif_size - 1 - permutation[i]) * 8)
                        
                perm_motif_num = (v << (8 * motif_size)) + color_perm

                if perm_motif_num not in motifs:
                    if perm_motif_num not in keys_to_add:
                        keys_to_add[perm_motif_num] = 0
                    keys_to_add[perm_motif_num] += 1
                else:
                    motifs[perm_motif_num] += motifs[motif]
    motifs.update(keys_to_add)
    
def read_graph_file(nodes_file, edges_file):
    """Read from .node_labels and .edges files."""
    G = nx.Graph()
    with open(nodes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            node_id, color = line.split(",")
            G.add_node(int(node_id), color=int(color))
    with open(edges_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, dst = line.split(",")
            G.add_edge(int(src), int(dst))
    return G


def read_json_graph(path):
    """Read from node_link JSON format."""
    G = nx.Graph()
    with open(path) as f:
        data = json.load(f)
    for node in data["nodes"]:
        G.add_node(node["id"], color=node["color"])
    for edge in data["links"]:
        G.add_edge(edge["source"], edge["target"])
    return G


# ---------------- MAIN ---------------- #

def main():

    logging.info("===== START RUN =====")

    # ---------- LOAD OR COMPUTE G MOTIFS ----------

    if os.path.exists(G_MOTIF_PICKLE):
        with open(G_MOTIF_PICKLE, "rb") as f:
            g_sum = pickle.load(f)
        logging.info("Loaded cached motif calculator")
        print("Loaded cached motif calculator")
    else:
        logging.info("Computing motif calculator")
        print("Computing motif calculator")
        G = read_graph_file(NODES_FILE, EDGES_FILE)
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
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
        g_motifs = calc.build()
        g_sum = g_motifs[MotifsNodeCalculator.MOTIF_SUM_KEY]
        motif_graph = nx.read_gpickle("local_tests/non_induced/create_inclusion_motifs_dag/4_undirected_colored_dag")
        g_sum = preprocess_motifs_for_non_induced(MOTIF_SIZE, g_sum, motif_graph)

        os.makedirs(os.path.dirname(G_MOTIF_PICKLE), exist_ok=True)
        with open(G_MOTIF_PICKLE, "wb") as f:
            pickle.dump(g_motifs, f)

    # ---------- LOAD OR COMPUTE PATH MOTIFS ----------

    if os.path.exists(G_PATH_PICKLE):
        with open(G_PATH_PICKLE, "rb") as f:
            g_paths = pickle.load(f)
        logging.info("Loaded cached PATH motifs")
        print("Loaded cached PATH motifs")
    else:
        logging.info("Computing PATH motifs")
        print("Computing PATH motifs")
        G = read_graph_file(NODES_FILE, EDGES_FILE)
        g_paths = PathMotifCalculator(G, False).build()
        os.makedirs(os.path.dirname(G_PATH_PICKLE), exist_ok=True)
        with open(G_PATH_PICKLE, "wb") as f:
            pickle.dump(g_paths, f)

    # ---------- COUNTERS ----------

    failed_motif = 0
    passed_motif = 0
    failed_path = 0
    passed_path = 0
    passed_both = 0
    processed    = 0

    # ---------- PROCESS S ----------

    for i in range(0, 1000):

        S_PATH = os.path.join(S_DIR, f"S_{i}.json")

        if not os.path.exists(S_PATH):
            logging.info(f"S_{i} MISSING")
            continue

        S = read_json_graph(S_PATH)

        # ----- MOTIF SUM CHECK -----

        s_calc = MotifsNodeCalculator(
            graph=S,
            colores_loaded=True,
            configuration=CONFIGURATION,
            level=MOTIF_SIZE,
            calc_nodes=False,
            calc_edges=False,
            count_motifs=True,
        )

        s_sum = s_calc.build()[MotifsNodeCalculator.MOTIF_SUM_KEY]

        feasible_sum = True
        for m, cnt in s_sum.items():
            if g_sum.get(m, 0) < cnt:
                feasible_sum = False
                break

        if not feasible_sum:
            logging.info(f"S_{i} FAILED MOTIF CHECK")
            failed_motif += 1
            processed += 1
        else:
            logging.info(f"S_{i} PASSED MOTIF CHECK")
            passed_motif += 1

        # ----- PATH CHECK -----

        s_paths = PathMotifCalculator(S, False).build()

        feasible_path = True
        for m, cnt in s_paths.items():
            if g_paths.get(m, 0) < cnt:
                feasible_path = False
                break

        if not feasible_path:
            logging.info(f"S_{i} FAILED_PATH_CHECK")
            failed_path += 1
        else:
            logging.info(f"S_{i} PASSED PATH CHECK")
            passed_path += 1
            if feasible_sum:
                logging.info(f"S_{i} PASSED BOTH CHECKS")
                passed_both += 1

        processed += 1

        if processed % 50 == 0:
            logging.info(f"Progress {processed}/1000")

    # ---------- SUMMARY ----------

    logging.info("===== SUMMARY =====")
    logging.info(f"Processed          = {processed}")
    logging.info(f"FAILED MOTIF CHECK = {failed_motif}")
    logging.info(f"PASSED MOTIF CHECK AND FAILED PATH CHECK  = {failed_path}")
    logging.info(f"PASSED             = {passed}")

    print("Done.")
    print("Processed:    ", processed)
    print("FAILED MOTIF: ", failed_motif)
    print("FAILED PATH:  ", failed_path)
    print("PASSED PATH:  ", passed_path)
    print("PASSED BOTH:  ", passed_both)


# ---------------- ENTRY ---------------- #

if __name__ == "__main__":
    main()