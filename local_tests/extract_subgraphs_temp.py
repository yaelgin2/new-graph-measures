"""
Extract subgraphs for specific node groups and save as individual JSON files.
"""

import json
import os
import networkx as nx

GRAPH_FILE = "local_tests/graphs_by_density_3/g_den_5_embedded_den_3_uniform_0.json"
OUT_DIR = "local_tests/test/extracted_subgraphs"
os.makedirs(OUT_DIR, exist_ok=True)

NODE_GROUPS = [
[20479, 37559, 34646, 21335],
[3472, 47907, 41451, 24543],
[3472, 47907, 15372, 24543],
[3472, 47907, 10520, 24543],
[3472, 41451, 15372, 24543],
[3472, 41451, 10520, 24543],
[3472, 15372, 10520, 24543],
[10973, 24337, 37282, 19337],
[9775, 13984, 21784, 35764],
[9980, 24917, 38665, 22838],
[12624, 31909, 29603, 44708],
[1570, 47305, 43571, 33466],
[4051, 20890, 17827, 8393],
[4082, 4691, 17785, 27194],
[7346, 39259, 22613, 34361],
[11255, 49701, 42293, 16797],
[1283, 5384, 18469, 14168],
[20796, 47811, 49637, 24569],
[20796, 47811, 49637, 25050],
[20796, 49637, 24569, 25050],
[25358, 30688, 47299, 26488],
[8797, 10714, 22734, 47127],
[16902, 26758, 45519, 22352],
[17101, 19025, 37282, 30424],
[17101, 19025, 17838, 30424],
[24780, 38026, 36879, 46203],
[26551, 31142, 35965, 48859],
[4779, 23686, 12424, 41645],
[7406, 25002, 19113, 22541],
[7406, 25002, 19113, 38103],
[7406, 25002, 22541, 45980],
[7406, 25002, 38103, 45980],
[8664, 16805, 36999, 38391],
[19113, 25002, 22541, 45980],
[19113, 25002, 38103, 45980],
[22613, 39259, 31019, 34361],
[23690, 47995, 41060, 31335],
[5944, 22155, 20960, 30162],
[7469, 30263, 16581, 11478],
[9803, 13193, 16208, 45393],
[12488, 48448, 14354, 30099],
[635, 7626, 23058, 7455],
[5095, 14354, 48448, 9626],
[14834, 47547, 17734, 32205],
[14834, 47547, 32205, 48463],
[21080, 41262, 26129, 38647],
[2900, 27096, 7182, 17511],
[5880, 20476, 28514, 40282],
[6179, 23290, 28307, 8313]
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