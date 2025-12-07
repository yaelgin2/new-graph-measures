import numpy as np
import networkx as nx

iterable_types = (list, tuple)
num_types = (int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)

def filter_graph_into_subraphs(graph, is_max_connected=False):
    if not is_max_connected:
        return graph
    if graph.is_directed():
        subgraphs = [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]
    else:
        subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    return max(subgraphs, key=len)
