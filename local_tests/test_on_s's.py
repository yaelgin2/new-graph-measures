import json
import os
import pickle
import logging
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import linprog

from graphMeasures.feature_calculators import MotifsNodeCalculator
from graphMeasures.loggers import PrintLogger


# ---------------- CONFIG ---------------- #

BASE_DIR = r"C:\Users\ginzb\Documents\new-graph-measures"
INPUT_DIR = os.path.join(BASE_DIR, "local_tests", "input_1")
PICKLE_DIR = os.path.join(BASE_DIR, "local_tests", "cache_1")
LOG_FILE = os.path.join(BASE_DIR, "lp_false_positives.log")

os.makedirs(PICKLE_DIR, exist_ok=True)

G_PICKLE = os.path.join(PICKLE_DIR, "G_motifs.pkl")

CONFIGURATION = {
    "colored_directed_variations_3": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\3_directed_colored.pkl",
    "colored_undirected_variations_3": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\3_undirected_colored.pkl",
    "colored_directed_variations_4": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\4_directed_colored.pkl",
    "colored_undirected_variations_4": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\4_undirected_colored.pkl",
}

MOTIF_SIZE = 3


# ---------------- LOGGING ---------------- #

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


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


# ---------------- LP SOLVER ---------------- #

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


# ---------------- MAIN ---------------- #

def main():
    # ----- Load or compute G motifs -----
    if os.path.exists(G_PICKLE):
        with open(G_PICKLE, "rb") as f:
            g_motifs = pickle.load(f)
        print("Loaded cached G motifs")
    else:
        G = read_graph_file(os.path.join(INPUT_DIR, "G.json"))
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

        with open(G_PICKLE, "wb") as f:
            pickle.dump(g_motifs, f)


        print("Computed and cached G motifs")

    g_sum = g_motifs.get(MotifsNodeCalculator.MOTIF_SUM_KEY)
    g_motifs.pop(MotifsNodeCalculator.MOTIF_SUM_KEY)

    solver = NodeSelectorLP(g_motifs, MOTIF_SIZE)

    false_pos_sum_only = 0
    false_pos_sum_and_lp = 0

    # ----- Process S graphs -----
    for i in range(1, 16):
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

        # ---------- Stage 1: motif sum check ----------
        feasible_sum = True
        for m, cnt in s_motifs.items():
            if g_sum.get(m, 0) < cnt:
                feasible_sum = False
                break

        if i > 10 and feasible_sum:
            false_pos_sum_only += 1

        if not feasible_sum:
            logging.info(f"SUM FAIL S_{i}")
            print(f"S_{i}: failed sum check")
            continue

        # ---------- Stage 2: LP ----------
        feasible_lp = solver.solve(s_motifs, max_nodes=len(S.nodes))

        if i <= 10 and not feasible_lp:
            logging.info(f"FAILED TRUE POSITIVE S_{i}")
            print(f"FAILED ON {i}")

        elif i > 10 and feasible_lp:
            false_pos_sum_and_lp += 1
            logging.info(f"LP FALSE POSITIVE S_{i}")

        print(f"Done S_{i}")

    print("False positives (sum only):", false_pos_sum_only)
    print("False positives (sum + LP):", false_pos_sum_and_lp)


if __name__ == "__main__":
    main()
