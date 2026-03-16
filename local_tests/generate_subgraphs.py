import networkx as nx
import os
import json
import random
from collections import deque


def load_single_graph(folder, graph_name):

    G = nx.Graph()

    edges_path = os.path.join(folder, f"{graph_name}.edges")
    labels_path = os.path.join(folder, f"{graph_name}.node_labels")

    with open(labels_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            if "," in line or " " in line:
                if "," in line:
                    node_id, label = map(int, line.split(","))
                else:
                    node_id, label = map(int, line.split())
            else:
                node_id = i + 1
                label = int(line)

            G.add_node(node_id, color=label)

    with open(edges_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "," in line:
                u, v = map(int, line.split(","))
            else:
                u, v = map(int, line.split())

            G.add_edge(u, v)

    return G


# BFS sampler
def bfs_sample(G, start_node, target_size):

    visited = set([start_node])
    queue = deque([start_node])

    while queue and len(visited) < target_size:

        node = queue.popleft()

        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)

        for neighbor in neighbors:

            if neighbor not in visited:

                visited.add(neighbor)
                queue.append(neighbor)

                if len(visited) >= target_size:
                    break

    return G.subgraph(visited).copy()


# Correct JSON format
def save_graph_json(G, filepath):

    data = {
        "nodes": [
            {
                "id": int(n),
                "color": int(G.nodes[n].get("color", -1))
            }
            for n in G.nodes()
        ],
        "links": [
            {
                "source": int(u),
                "target": int(v)
            }
            for u, v in G.edges()
        ]
    }

    with open(filepath, "w") as f:
        json.dump(data, f)


def extract_subgraphs(
    G,
    output_folder,
    num_subgraphs=1000,
    min_size=2000,
    max_size=20000
):

    os.makedirs(output_folder, exist_ok=True)

    nodes = list(G.nodes())

    for i in range(num_subgraphs):

        start = random.choice(nodes)

        target_size = random.randint(min_size, max_size)

        subgraph = bfs_sample(G, start, target_size)

        filename = f"S_{i}.json"

        filepath = os.path.join(output_folder, filename)

        save_graph_json(subgraph, filepath)

        print(f"Saved {filename}")


# RUN
folder = "/home/cohent59/new-graph-measures/local_tests/real_graphs"

output = "/home/cohent59/new-graph-measures/local_tests/real_graphs/NCI109_subgraphs"

G = load_single_graph(folder, "NCI109")

extract_subgraphs(
    G,
    output_folder=output,
    num_subgraphs=1000,
    min_size=2000,
    max_size=20000
)
