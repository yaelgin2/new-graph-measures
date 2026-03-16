"""
Algorithm: dotmotif (GrandIso executor)
Article: Matelsky et al., "DotMotif: an open-source tool for connectome subgraph
         isomorphism search and graph queries", Scientific Reports 2021.
Link: https://github.com/aplbrain/dotmotif

============================================================
BUILD / INSTALL INSTRUCTIONS
============================================================
pip install dotmotif networkx

dotmotif works by querying one specific motif pattern at a time using a
simple DSL. Node attribute constraints are expressed inline, e.g.:
    A -- B
    A.color == 1
    B.color == 2

Like igraph, dotmotif is a per-pattern query tool, not a census tool.
We therefore enumerate all canonical (topology, color_tuple) pairs and
run one query per pair, exactly as in benchmark_igraph.py.

The GrandIsoExecutor uses the GrandIso library for fast subgraph matching.
It operates on NetworkX graphs and supports node/edge attribute constraints
natively via the DotMotif DSL.

NOTE: dotmotif finds all *labeled* isomorphisms (i.e. automorphisms are
not collapsed). We divide by the automorphism group size to get true
motif occurrence counts, consistent with the reference implementation.
============================================================
"""

import json
import os
import time
from itertools import permutations, product

import networkx as nx
from dotmotif import Motif, GrandIsoExecutor

# -------------------------------------------------------
# Config
# -------------------------------------------------------
GRAPH_PATTERN = "local_tests/graphs_by_density_3/g_den_{den}_embedded_den_3_{dist}_0.json"
DEN_LIST = [5, 8, 10, 13, 15]
DIST_LIST = ["uniform", "average", "rare"]

LOG_DIR = "local_tests/other_algorithms_tests/logs"
OUTPUT_DIR = "local_tests/other_algorithms_tests/dotmotif"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "dotmotif.log")

COLOR_ATTRIBUTE = "color"

# 5 canonical 4-node undirected topologies
# Node names A, B, C, D used in DSL
TOPOLOGIES_DSL = {
    "P4":      "A -- B\nB -- C\nC -- D\n",
    "STAR":    "A -- B\nA -- C\nA -- D\n",
    "C4":      "A -- B\nB -- C\nC -- D\nD -- A\n",
    "DIAMOND": "A -- B\nB -- C\nC -- D\nD -- A\nA -- C\n",
    "K4":      "A -- B\nA -- C\nA -- D\nB -- C\nB -- D\nC -- D\n",
}

TOPOLOGIES_EDGES = {
    "P4":      [(0,1),(1,2),(2,3)],
    "STAR":    [(0,1),(0,2),(0,3)],
    "C4":      [(0,1),(1,2),(2,3),(3,0)],
    "DIAMOND": [(0,1),(1,2),(2,3),(3,0),(0,2)],
    "K4":      [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],
}

NODE_NAMES = ["A", "B", "C", "D"]


# -------------------------------------------------------
# Read graph as NetworkX (dotmotif uses NetworkX)
# -------------------------------------------------------
def read_graph(filename):
    with open(filename) as f:
        gj = json.load(f)

    G = nx.Graph()
    colors_set = set()
    for node in gj["nodes"]:
        color = node.get(COLOR_ATTRIBUTE, 0)
        G.add_node(node["id"], color=color)
        colors_set.add(color)
    for edge in gj["links"]:
        G.add_edge(edge["source"], edge["target"])

    return G, colors_set


# -------------------------------------------------------
# Automorphism group helpers (reused from igraph script)
# -------------------------------------------------------
def get_automorphisms(edge_list):
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edges_from(edge_list)
    autos = []
    for perm in permutations(range(4)):
        mapping = {i: perm[i] for i in range(4)}
        H = nx.relabel_nodes(G, mapping)
        if nx.is_isomorphic(G, H):
            autos.append(perm)
    return autos


AUTOMORPHISMS = {name: get_automorphisms(edges) for name, edges in TOPOLOGIES_EDGES.items()}


def canonical_color_tuples(topology_name, color_set):
    autos = AUTOMORPHISMS[topology_name]
    colors = sorted(color_set)
    seen = set()
    canonical = []
    for combo in product(colors, repeat=4):
        min_perm = min(tuple(combo[p] for p in auto) for auto in autos)
        if min_perm not in seen:
            seen.add(min_perm)
            canonical.append(combo)
    return canonical


# -------------------------------------------------------
# Build dotmotif DSL string with color constraints
# -------------------------------------------------------
def build_dotmotif_dsl(topology_name, color_tuple):
    dsl = TOPOLOGIES_DSL[topology_name]
    for i, name in enumerate(NODE_NAMES):
        dsl += f"{name}.color == {color_tuple[i]}\n"
    return dsl


# -------------------------------------------------------
# Count all colored motifs in G
# -------------------------------------------------------
def count_colored_motifs(G, colors_set):
    executor = GrandIsoExecutor(graph=G)
    motif_counts = {}

    for topo_name in TOPOLOGIES_DSL:
        canonical_combos = canonical_color_tuples(topo_name, colors_set)
        autos = AUTOMORPHISMS[topo_name]
        auto_size = len(autos)

        for color_combo in canonical_combos:
            dsl = build_dotmotif_dsl(topo_name, color_combo)
            try:
                motif = Motif(dsl)
                results = executor.find(motif)
            except Exception:
                # skip invalid or degenerate queries
                continue

            count = len(results) // auto_size
            if count > 0:
                canonical_key = min(
                    tuple(color_combo[p] for p in auto) for auto in autos
                )
                key = (topo_name, canonical_key)
                motif_counts[key] = count

    return motif_counts


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    with open(LOG_FILE, "w") as logf:
        for den in DEN_LIST:
            for dist in DIST_LIST:
                graph_file = GRAPH_PATTERN.format(den=den, dist=dist)
                if not os.path.exists(graph_file):
                    print(f"Missing: {graph_file}")
                    logf.write(f"MISSING: {graph_file}\n")
                    continue

                G, colors_set = read_graph(graph_file)

                start = time.time()
                motif_counts = count_colored_motifs(G, colors_set)
                elapsed = time.time() - start

                logf.write(f"{graph_file}: {elapsed:.4f} sec\n")
                logf.flush()

                out_base = os.path.basename(graph_file).replace(".json", "_motifs.txt")
                out_path = os.path.join(OUTPUT_DIR, out_base)
                with open(out_path, "w") as f:
                    for (topo, color_key), count in sorted(motif_counts.items()):
                        f.write(f"{topo}\t{color_key}\t{count}\n")

                print(f"Processed {graph_file} in {elapsed:.2f}s "
                      f"| {len(motif_counts)} distinct motifs | saved {out_path}")


if __name__ == "__main__":
    main()
