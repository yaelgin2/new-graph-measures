"""
Extract subgraphs for specific node groups and save as individual JSON files.
"""

import json
import os
import networkx as nx

GRAPH_FILE = "local_tests/graphs_by_density_3/g_den_8_embedded_den_3_average_0.json"
OUT_DIR = "local_tests/test/extracted_subgraphs"
os.makedirs(OUT_DIR, exist_ok=True)

NODE_GROUPS = [
[1757, 385, 0, 4, 15]
]

# Load graph
with open(GRAPH_FILE) as f:
    data = json.load(f)

G = nx.Graph()
for node in data["nodes"]:
    G.add_node(node["id"], color=node["color"])
for edge in data["links"]:
    G.add_edge(edge["source"], edge["target"])

for idx, group in enumerate(NODE_GROUPS):
    node_set = set(group)
    subgraph_nodes = [
        {"id": i, "original_id": n, "color": G.nodes[n]["color"]}
        for i, n in enumerate(group)
    ]
    orig_to_idx = {n: i for i, n in enumerate(group)}
    subgraph_links = [
        {"source": orig_to_idx[u], "target": orig_to_idx[v]}
        for u, v in G.edges()
        if u in node_set and v in node_set
    ]

    out = {"nodes": subgraph_nodes, "links": subgraph_links}
    fname = os.path.join(OUT_DIR, f"group_{idx:02d}_{'_'.join(map(str, group))}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {fname} — edges: {subgraph_links}")

print(f"\nDone. {len(NODE_GROUPS)} subgraphs saved to {OUT_DIR}")