import itertools

import networkx as nx
import matplotlib.pyplot as plt
import random
import json
from collections import deque, defaultdict
import inspect
import os, sys
OUTPUT_DIR = r"C:\Users\ginzb\Documents\new-graph-measures\local_tests\input_color_uniform_deg_3"

# Create Colored Graph G(n,p)
def create_colored_graph(n,p):
    
    C=20
    pi=None
    seed=None

    if seed is not None:
        random.seed(seed)

    # Default color distribution
    if pi is None:
        pi = [
            0.05, 0.05, 0.05, 0.05, 0.05,  # 5 dominant colors (75%)
            0.05, 0.05, 0.05, 0.05, 0.05,  # medium colors
            0.05, 0.05, 0.05, 0.05, 0.05,  # rare
            0.05, 0.05, 0.05, 0.05, 0.05  # very rare
            ]


    G = nx.erdos_renyi_graph(n=n, p=p)

    # Assign colors
    for v in G.nodes():
        G.nodes[v]['color'] = random.choices(range(C), weights=pi, k=1)[0]

    components = list(nx.connected_components(G))

    for i in range(len(components) - 1):
        u = next(iter(components[i]))
        v = next(iter(components[i + 1]))
        G.add_edge(u, v)

    return G


# Convert NetworkX graph → required JSON format
def graph_to_json_struct(S):
    data = {"nodes": [], "links": []}

    for u in S.nodes():
        data["nodes"].append({
            "id": int(u),
            "color": int(S.nodes[u]['color'])
        })

    for u, v in S.edges():
        data["links"].append({
            "source": int(u),
            "target": int(v)
        })

    return data

# Save JSON to file
def save_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)
    #print(f"Saved: {filename}")

#create sub-graphs:
def generate_and_save_subgraphs(count, avg_degree, sizeL, sizeH, folder, keep_first):
    os.makedirs(folder, exist_ok=True)

    s_list = []  # only the first keep_first
    
    for i in range(1, count+1):
        print(f"Generating S_{i} ...")

        # random size
        k = random.randint(int(sizeL), int(sizeH))

        # generate connected S_i
        S = create_colored_graph(k, avg_degree.__float__()/k)
        while not nx.is_connected(S):
            print(f"S_{i} not connected — regenerating")
            S = create_colored_graph(k, avg_degree.__float__()/k)

        # save immediately to JSON
        data_S = graph_to_json_struct(S)
        filename = os.path.join(folder, f"S_{i}.json")
        save_json(data_S, filename)

        print(f"S_{i} saved.")

        # keep only the first 100
        if i <= keep_first:
            s_list.append(S)

        # free memory (important!)
        del S

    return s_list

def embed_subgraph(G,S,available_nodes):
    """
    Embed subgraph S into G as a non-induced subgraph.
    Only ensures edges of S exist in G.
    Does NOT remove any extra edges that G already has.
    """     
    nodes_S = list(S.nodes())
    k = len(nodes_S)
    
    if len(available_nodes) < k:
        raise ValueError("NOT ENOUGH UNUSED NODES LEFT TO EMBED S!")

    # random k nodes in G
    nodes_G = random.sample(list(available_nodes), k)

    # map S nodes → G nodes
    mapping = dict(zip(nodes_S, nodes_G))

    # copy colors
    for u_S, u_G in mapping.items():
        G.nodes[u_G]["color"] = S.nodes[u_S]["color"]
        available_nodes.remove(u_G)

    print(mapping)

    for v_G, u_G, in itertools.combinations(mapping.values(), 2):
        if (v_G, u_G) in G.edges():
            G.remove_edge(v_G, u_G)
        if (u_G, v_G) in G.edges():
            G.remove_edge(u_G, v_G)

    # add edges that appear in S
    for u_S, v_S in S.edges():
        u_G = mapping[u_S]
        v_G = mapping[v_S]
        G.add_edge(u_G, v_G)

    print (f"G edges in s{i}");
    for edge in G.edges():
        if edge[0] in mapping.values() and edge[1] in mapping.values():
            print(edge)

    print("________________")


    return available_nodes


    
# Run
if __name__ == "__main__":
    n = 50000
    avg_neighbors = 3
    G = create_colored_graph(n, float(avg_neighbors) / n)
    print("done generating G")

    degrees = defaultdict(int)
    for node in G.nodes():
        degrees[G.degree(node)] += 1
    print(degrees)

    sizeL=1500
    sizeH=2000
    count_S=1000
     #embed sub-graphs
    how_many_to_embed = 10

    s_list=generate_and_save_subgraphs(count_S, avg_neighbors,sizeL,sizeH,OUTPUT_DIR,how_many_to_embed)
    print("done generating s_list")


    available_nodes = set(G.nodes())
    for i in range(how_many_to_embed):
        print(f"embed s number {i+1}")
        available = embed_subgraph(G, s_list[i],available_nodes)
        available_nodes = available

    data_G = graph_to_json_struct(G)
    save_json(data_G, f"{OUTPUT_DIR}\\G.json")
    print("DONE: G and all S_i saved.")