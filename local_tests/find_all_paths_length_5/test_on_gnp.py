import json
import os
import pickle
import logging

import networkx as nx

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger

from .PathMotifCalculator import PathMotifCalculator


# ---------------- CONFIG ---------------- #

BASE_DIR = "/home/cohent59/new-graph-measures/local_tests"

GRAPH_NAME = "g_den_15_embedded_den_3_rare_0"

GRAPH_PATH = os.path.join(
    BASE_DIR,
    "graphs_by_density_3",
    f"{GRAPH_NAME}.json"
)

S_DIR = os.path.join(
    BASE_DIR,
    "input_color_rare_deg_3"
)

# ---- existing motif cache stays SAME ----
MOTIF_PICKLE_DIR = os.path.join(BASE_DIR, "induced", "cache")

# ---- NEW PATH CACHE ----
PATH_CACHE_DIR = os.path.join(
    BASE_DIR,
    "find_all_paths_length_5",
    "cache"
)

# ---- LOG ----
LOG_DIR = os.path.join(
    BASE_DIR,
    "find_all_paths_length_5",
    "logs"
)

os.makedirs(PATH_CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"{GRAPH_NAME}_paths.log"
)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    force=True
)

# motif cache (UNCHANGED LOCATION)
G_MOTIF_PICKLE = os.path.join(
    MOTIF_PICKLE_DIR,
    f"{GRAPH_NAME}.pkl"
)

# path cache (NEW LOCATION)
G_PATH_PICKLE = os.path.join(
    PATH_CACHE_DIR,
    f"{GRAPH_NAME}.pkl"
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

def read_graph_file(filename):

    G = nx.Graph()

    with open(filename) as f:
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
            g_motifs = pickle.load(f)

        logging.info("Loaded cached motif calculator")
        print("Loaded cached motif calculator")

    else:

        logging.info("Computing motif calculator")
        print("Computing motif calculator")

        G = read_graph_file(GRAPH_PATH)

        calc = MotifsNodeCalculator(
            graph=G,
            colores_loaded=True,
            configuration=CONFIGURATION,
            level=MOTIF_SIZE,
            calc_nodes=True,
            calc_edges=False,
            count_motifs=True,
            logger=PrintLogger(),
        )

        g_motifs = calc.build()

        with open(G_MOTIF_PICKLE, "wb") as f:
            pickle.dump(g_motifs, f)

    g_sum = g_motifs.get(
        MotifsNodeCalculator.MOTIF_SUM_KEY
    )

    # ---------- LOAD OR COMPUTE PATH MOTIFS ----------

    if os.path.exists(G_PATH_PICKLE):

        with open(G_PATH_PICKLE, "rb") as f:
            g_paths = pickle.load(f)

        logging.info("Loaded cached PATH motifs")
        print("Loaded cached PATH motifs")

    else:

        logging.info("Computing PATH motifs")
        print("Computing PATH motifs")

        G = read_graph_file(GRAPH_PATH)

        g_paths = PathMotifCalculator(G).build()

        with open(G_PATH_PICKLE, "wb") as f:
            pickle.dump(g_paths, f)

    # ---------- COUNTERS ----------

    failed_motif = 0
    failed_path = 0
    passed = 0
    processed = 0

    # ---------- PROCESS S ----------

    for i in range(1, 1001):

        S_PATH = os.path.join(S_DIR, f"S_{i}.json")

        if not os.path.exists(S_PATH):

            logging.info(f"S_{i} MISSING")
            continue

        S = read_graph_file(S_PATH)

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

        s_sum = s_calc.build()[
            MotifsNodeCalculator.MOTIF_SUM_KEY
        ]

        feasible_sum = True

        for m, cnt in s_sum.items():

            if g_sum.get(m, 0) < cnt:

                feasible_sum = False
                break

        if not feasible_sum:

            logging.info(f"S_{i} FAILED MOTIF CHECK")

            failed_motif += 1
            processed += 1
            continue

        # ----- PATH CHECK (ONLY IF PASSED) -----

        s_paths = PathMotifCalculator(S).build()

        feasible_path = True

        for m, cnt in s_paths.items():

            if g_paths.get(m, 0) < cnt:

                feasible_path = False
                break

        if not feasible_path:

            logging.info(f"S_{i} FAILED_PATH_CHECK")

            failed_path += 1

        else:

            logging.info(f"S_{i} PASSED")

            passed += 1

        processed += 1

        if processed % 50 == 0:

            logging.info(f"Progress {processed}/1000")

    # ---------- SUMMARY ----------

    logging.info("===== SUMMARY =====")

    logging.info(f"Processed = {processed}")
    logging.info(f"FAILED MOTIF CHECK = {failed_motif}")
    logging.info(f"FAILED PATH CHECK = {failed_path}")
    logging.info(f"PASSED = {passed}")

    print("Done.")
    print("Processed:", processed)
    print("FAILED MOTIF:", failed_motif)
    print("FAILED PATH:", failed_path)
    print("PASSED:", passed)


# ---------------- ENTRY ---------------- #

if __name__ == "__main__":
    main()
