import networkx as nx
from matplotlib import pyplot as plt


def get_empty_directed_graph():
    graph = nx.DiGraph()
    return graph

def get_directed_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(12, 1), (1, 12), (2, 3), (3, 4), (5, 2), (2, 6), (4, 7),
                        (4, 8), (9, 6), (7, 10), (11, 7), (10, 11), (10, 13), (10, 14),
                        (14, 10), (15, 12), (12, 16), (16, 12), (16, 15)])
    return graph

def get_empty_undirected_graph():
    graph = nx.Graph()
    return graph

def get_undirected_graph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 6), (3, 6), (3, 7), (4, 5), (6, 7),
                        (4, 9), (6, 9), (6, 10), (10, 11), (9, 12), (8, 13),
                        (10, 14), (14, 15)])
    return graph

def get_undirected_motif0_graph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    return graph

def get_undirected_motif1_graph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return graph

def get_directed_motif0_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2)])
    return graph

def get_directed_motif1_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (1, 0)])
    return graph

def get_directed_motif2_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 0)])
    return graph

def get_directed_motif3_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (1, 2)])
    return graph

def get_directed_motif4_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 2)])
    return graph

def get_directed_motif5_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 0), (1, 2)])
    return graph

def get_directed_motif6_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 0), (2, 0)])
    return graph

def get_directed_motif7_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 0), (2, 0)])
    return graph

def get_directed_motif8_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return graph

def get_directed_motif9_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 0)])
    return graph

def get_directed_motif10_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (1, 0), (1, 2), (2, 0)])
    return graph

def get_directed_motif11_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0)])
    return graph

def get_directed_motif12_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
    return graph
