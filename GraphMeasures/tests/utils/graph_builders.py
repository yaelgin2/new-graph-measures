import networkx as nx
from matplotlib import pyplot as plt


def get_directed_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(12, 1), (1, 12), (2, 3), (3, 4), (5, 2), (2, 6), (4, 7),
                        (4, 8), (9, 6), (7, 10), (11, 7), (10, 11), (10, 13), (10, 14),
                        (14, 10), (15, 12), (12, 16), (16, 12), (16, 15)])
    return graph


def get_undirected_graph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 6), (3, 6), (3, 7), (4, 5), (6, 7),
                        (4, 9), (6, 9), (6, 10), (10, 11), (9, 12), (8, 13),
                        (10, 14), (14, 15)])
    return graph
