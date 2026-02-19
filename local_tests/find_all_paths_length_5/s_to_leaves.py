import argparse
import json
import os
from collections import deque
import networkx as nx

periphery_color = 10000


def find_leaves_upstream(graph):
    # color_nodes = {node for node, data in graph.nodes(data=True) if data.get('color') == color}
    exits = [node for node in graph.nodes if graph.out_degree(node) == 0]
    leaf_subgraphs = []

    for start in exits:
        visited = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            predecessors = list(graph.predecessors(node))

            # Otherwise, add all unvisited predecessors to the queue
            for nbr in predecessors:
                if nbr not in visited:
                    queue.append(nbr)
        subgraph = graph.subgraph(visited).copy()
        leaf_subgraphs.append(subgraph)
    return leaf_subgraphs


def from_sub_to_leaves(subgraph_file):
    with open(subgraph_file, 'r') as f:
        data = json.load(f)
    # Check if the graph is directed
    if data['directed']:
        G = nx.DiGraph()  # Create a directed graph
    else:
        G = nx.Graph()  # Create an undirected graph
    # Add nodes with attributes
    for node in data['nodes']:
        G.add_node(node['id'], **node)
    # Add edges
    for link in data['links']:
        G.add_edge(link['source'], link['target'])
    leaves = find_leaves_upstream(G)
    return leaves


def save_graph_info(graph, folder_path, text):
    # Create subfolder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Save the nodes with their colors
    nodes_file = os.path.join(folder_path, f"{text}.node_labels")
    with open(nodes_file, "w") as f:
        for node, attr in graph.nodes(data=True):
            color = attr.get("color", "unknown")  # Default to 'unknown' if no color attribute
            f.write(f"{node} {color}\n")

    # Save the edges (source, target)
    edges_file = os.path.join(folder_path, f"{text}.edges")
    with open(edges_file, "w") as f:
        for source, target in graph.edges():
            f.write(f"{source}  {target}\n")


def main():
    parser = argparse.ArgumentParser(description="Process graph and save core and leaf nodes/edges.")
    parser.add_argument('--sub_file', required=True, help='Path to the graph file (GraphML format)')
    # parser.add_argument('--out_color', required=True, help='Color to identify leaf nodes')
    parser.add_argument('--folder', required=True, help='Output folder path to save the leaf information')

    args = parser.parse_args()

    os.makedirs(args.folder, exist_ok=True)

    # Process the graph based on input arguments
    leafs = from_sub_to_leaves(args.sub_file)


    # Save each leaf's graph information
    for index, leaf in enumerate(leafs):
        leaf_folder = os.path.join(args.folder, f"leaf_{index}")
        save_graph_info(leaf, leaf_folder, f"leaf_{index}")
    print(len(leafs))


if __name__ == "__main__":
    main()
