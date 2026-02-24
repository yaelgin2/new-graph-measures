"""
Algorithm: igraph (VF2 subgraph isomorphism with vertex colors)
Article: Csardi & Nepusz, "The igraph software package for complex network research",
         InterJournal Complex Systems, 2006.
Link: https://igraph.org/python/

============================================================
BUILD / INSTALL INSTRUCTIONS
============================================================
pip install igraph networkx

NOTE ON APPROACH:
igraph does not have a native colored motif census function.
Instead we use get_subisomorphisms_vf2() which finds all occurrences
of a specific query graph (with color constraints) inside the host graph.
We must therefore enumerate all canonical (topology, color_tuple) pairs
explicitly and query each one.

For k=4 undirected there are 5 topologies (P4, STAR, C4, DIAMOND, K4).
For each topology we query all color-assignments that are non-isomorphic
under the topology's automorphism group (using the same canonical
min-color-permutation logic as the reference implementation).

This makes igraph a fair comparison but it will be slower than FANMOD+
on large color spaces since it runs one VF2 query per canonical motif.
============================================================
"""

import json
import os
import time
from collections import defaultdict
from itertools import combinations, product, permutations

import igraph as ig
import networkx as nx

# -------------------------------------------------------
# Config
# -------------------------------------------------------
GRAPH_PATTERN = "local_tests/graphs_by_density_3/g_den_{den}_embedded_den_3_{dist}_0.json"
DEN_LIST = [5, 8, 10, 13, 15]
DIST_LIST = ["uniform", "average", "rare"]

LOG_DIR = "local_tests/other_algorithms_tests/logs"
OUTPUT_DIR = "local_tests/other_algorithms_tests/igraph"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "igraph.log")

COLOR_ATTRIBUTE = "color"

# 5 canonical 4-node undirected topologies as edge lists (0-indexed nodes)
TOPOLOGIES = {
    "P4":      [(0,1),(1,2),(2,3)],
    "STAR":    [(0,1),(0,2),(0,3)],
    "C4":      [(0,1),(1,2),(2,3),(3,0)],
    "DIAMOND": [(0,1),(1,2),(2,3),(3,0),(0,2)],
    "K4":      [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],
}


# -------------------------------------------------------
# Read graph
# -------------------------------------------------------
def read_graph(filename):
    with open(filename) as f:
        gj = json.load(f)

    # Build igraph
    node_list = [n["id"] for n in gj["nodes"]]
    node_colors = {n["id"]: n.get(COLOR_ATTRIBUTE, 0) for n in gj["nodes"]}
    id_map = {nid: i for i, nid in enumerate(node_list)}

    g = ig.Graph(n=len(node_list), directed=False)
    g.vs["color"] = [node_colors[nid] for nid in node_list]
    g.vs["name"] = node_list

    for edge in gj["links"]:
        g.add_edge(id_map[edge["source"]], id_map[edge["target"]])

    colors_set = set(node_colors.values())
    return g, colors_set


# -------------------------------------------------------
# Get automorphism group of a topology (as permutations of 4 nodes)
# We use networkx for this since igraph automorphism is harder to enumerate
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


# Precompute automorphisms for each topology
AUTOMORPHISMS = {name: get_automorphisms(edges) for name, edges in TOPOLOGIES.items()}


# -------------------------------------------------------
# For a topology, enumerate canonical color tuples
# (one representative per automorphism equivalence class)
# -------------------------------------------------------
def canonical_color_tuples(topology_name, color_set):
    autos = AUTOMORPHISMS[topology_name]
    colors = sorted(color_set)
    seen = set()
    canonical = []
    for combo in product(colors, repeat=4):
        # find min permutation under automorphism group
        min_perm = min(tuple(combo[p] for p in auto) for auto in autos)
        if min_perm not in seen:
            seen.add(min_perm)
            canonical.append(combo)  # original combo, min_perm is the key
    return canonical


# -------------------------------------------------------
# Build a query igraph with specific topology and color assignment
# -------------------------------------------------------
def build_query_graph(topology_name, color_tuple):
    edge_list = TOPOLOGIES[topology_name]
    q = ig.Graph(n=4, directed=False)
    q.add_edges(edge_list)
    q.vs["color"] = list(color_tuple)
    return q


# -------------------------------------------------------
# Count all colored motif occurrences in host graph g
# Returns dict: {(topology, color_tuple_canonical) -> count}
# -------------------------------------------------------
def count_colored_motifs(g, colors_set):
    host_colors = g.vs["color"]
    motif_counts = {}

    for topo_name in TOPOLOGIES:
        canonical_combos = canonical_color_tuples(topo_name, colors_set)
        autos = AUTOMORPHISMS[topo_name]

        for color_combo in canonical_combos:
            query = build_query_graph(topo_name, color_combo)

            # get_subisomorphisms_vf2 returns list of mappings (host_vertex for each query_vertex)
            isos = g.get_subisomorphisms_vf2(
                query,
                color1=host_colors,      # colors of host vertices
                color2=list(color_combo) # colors of query vertices
            )

            if not isos:
                continue

            # Each undirected motif occurrence is counted once per automorphism,
            # so divide by the automorphism group size
            auto_size = len(autos)
            count = len(isos) // auto_size

            if count > 0:
                # canonical key = min color tuple under automorphisms
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

                g, colors_set = read_graph(graph_file)

                start = time.time()
                motif_counts = count_colored_motifs(g, colors_set)
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
