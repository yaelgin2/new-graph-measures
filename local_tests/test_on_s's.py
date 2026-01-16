import json
import os
import pickle
import logging
import sys
from collections import defaultdict
from networkx.algorithms import isomorphism as iso

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import linprog

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger


# ================== CONFIG ==================

BASE_DIR = "/home/cohent59/new-graph-measures/local_tests"

LOG_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

SUMMARY_LOG = os.path.join(LOG_DIR, "summary.log")

CONFIGURATION = {
    "colored_directed_variations_3": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_directed_colored.pkl",
    "colored_undirected_variations_3": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/3_undirected_colored.pkl",
    "colored_directed_variations_4": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_directed_colored.pkl",
    "colored_undirected_variations_4": "graphMeasures/feature_calculators/node_features_calculators/calculators/motif_variations/4_undirected_colored.pkl",
}

MOTIF_SIZE = 4

DISTRIBUTIONS = ["uniform", "average", "rare"]
DEGREES = [3, 8, 15]


# ================== HELPERS ==================

def read_graph_file(filename):
    graph = nx.Graph()
    with open(filename) as f:
        graph_json = json.load(f)

    for node in graph_json["nodes"]:
        graph.add_node(node["id"], color=node["color"])

    for edge in graph_json["links"]:
        graph.add_edge(edge["source"], edge["target"])

    return graph


# ================== LP SOLVER ==================

class NodeSelectorLP:
    def __init__(self, node_motifs: dict[int, dict[int, int]], motif_size: int):
        self.node_motifs = node_motifs
        self.node_ids = list(node_motifs.keys())
        self.node_index = {n: i for i, n in enumerate(self.node_ids)}
        self.num_nodes = len(self.node_ids)
        self.motif_size = motif_size

        self.available_motifs = set()
        for d in node_motifs.values():
            self.available_motifs.update(d.keys())

    def solve(self, required_motifs: dict[int, int], max_nodes: int) -> bool:
        for m in required_motifs:
            if m not in self.available_motifs:
                return False

        motif_ids = list(required_motifs.keys())
        motif_index = {m: i for i, m in enumerate(motif_ids)}

        rows, cols, data = [], [], []

        for node, motifs in self.node_motifs.items():
            i = self.node_index[node]
            for m, cnt in motifs.items():
                if m in motif_index:
                    rows.append(motif_index[m])
                    cols.append(i)
                    data.append(-cnt)

        A_motifs = coo_matrix(
            (data, (rows, cols)),
            shape=(len(motif_ids), self.num_nodes)
        )

        b_motifs = -np.array([
            required_motifs[m] * self.motif_size for m in motif_ids
        ])

        A_nodes = np.ones((1, self.num_nodes))
        b_nodes = np.array([max_nodes])

        A = np.vstack([A_motifs.toarray(), A_nodes])
        b = np.concatenate([b_motifs, b_nodes])

        res = linprog(
            c=np.ones(self.num_nodes),
            A_ub=A,
            b_ub=b,
            bounds=[(0, 1)] * self.num_nodes,
            method="highs"
        )

        return res.success

def is_induced_subgraph(G, S):
    node_match = iso.categorical_node_match("color", None)

    matcher = iso.GraphMatcher(
        G,
        S,
        node_match=node_match
    )

    return matcher.subgraph_is_monomorphic()

# ================== MAIN ==================

def main():

    summary_logger = logging.getLogger("summary")
    summary_logger.setLevel(logging.INFO)
    summary_handler = logging.FileHandler(SUMMARY_LOG)
    summary_logger.addHandler(summary_handler)

    for dist in DISTRIBUTIONS:
        for deg in DEGREES:

            run_name = f"color_{dist}_deg_{deg}"
            INPUT_DIR = os.path.join(BASE_DIR, f"input_{run_name}")
            PICKLE_FILE = os.path.join(CACHE_DIR, f"G_motifs_{run_name}.pkl")
            LOG_FILE = os.path.join(LOG_DIR, f"lp_{run_name}.log")

            # -------- configure per-run logger --------
            logging.basicConfig(
                filename=LOG_FILE,
                level=logging.INFO,
                format="%(asctime)s - %(message)s",
                force=True
            )

            logging.info(f"Starting run {run_name}")

            G = read_graph_file(os.path.join(INPUT_DIR, "G.json"))
            # -------- Load or compute G motifs --------
            if os.path.exists(PICKLE_FILE):
                with open(PICKLE_FILE, "rb") as f:
                    g_motifs = pickle.load(f)
                logging.info("Loaded cached G motifs")
            else:
                
                g_calc = MotifsNodeCalculator(
                    graph=G,
                    colores_loaded=True,
                    configuration=CONFIGURATION,
                    level=MOTIF_SIZE,
                    calc_nodes=True,
                    calc_edges=False,
                    count_motifs=True,
                    logger=PrintLogger(),
                )
                g_motifs = g_calc.build()
                with open(PICKLE_FILE, "wb") as f:
                    pickle.dump(g_motifs, f)
                logging.info("Computed and cached G motifs")

            g_sum = g_motifs.pop(MotifsNodeCalculator.MOTIF_SUM_KEY)
            solver = NodeSelectorLP(g_motifs, MOTIF_SIZE)

            false_pos_sum_only = 0
            false_pos_sum_and_lp = 0

            # -------- Process S graphs --------
            for i in range(1, 1001):
                S = read_graph_file(os.path.join(INPUT_DIR, f"S_{i}.json"))

                s_calc = MotifsNodeCalculator(
                    graph=S,
                    colores_loaded=True,
                    configuration=CONFIGURATION,
                    level=MOTIF_SIZE,
                    calc_nodes=False,
                    calc_edges=False,
                    count_motifs=True,
                )
                s_motifs = s_calc.build()[MotifsNodeCalculator.MOTIF_SUM_KEY]

                print(is_induced_subgraph(G, S))
                feasible_sum = all(
                    g_sum.get(m, 0) >= cnt for m, cnt in s_motifs.items()
                )

                if i > 10 and feasible_sum:
                    false_pos_sum_only += 1
                    
                elif i < 10 and not feasible_sum:
                    logging.error(f"S{i} Failed sum check.")

                elif not feasible_sum:
                    logging.info(f"S{i} Failed sum check.")
                    continue

                if feasible_sum:
                    feasible_lp = solver.solve(s_motifs, max_nodes=len(S.nodes))

                    if i > 10 and feasible_lp:
                        false_pos_sum_and_lp += 1
                    elif i < 10 and not feasible_lp:
                        logging.error(f"S{i} Failed lp check.")
                    elif not feasible_lp:
                        logging.info(f"S{i} Failed lp check.")

            # -------- Log results --------
            logging.info(f"False positives (sum only): {false_pos_sum_only}")
            logging.info(f"False positives (sum + LP): {false_pos_sum_and_lp}")

            summary_logger.info(
                f"{run_name} | sum_only={false_pos_sum_only} | sum_plus_lp={false_pos_sum_and_lp}"
            )

            print(f"Finished {run_name}")

    print("All runs completed.")


if __name__ == "__main__":
    main()
