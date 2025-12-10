import os.path
import pickle
from itertools import combinations, permutations

import networkx as nx

OUT_FOLDER_PATH = r"graphMeasures\feature_calculators\node_features_calculators\calculators\motif_variations"

class IsomorphismGenerator:
    def __init__(self, group_size, is_directed):
        self._group_size = group_size
        self._is_directed = is_directed
        graphs = self._generate_all_graphs()
        self._isomorphisms = self._group_to_isomorphisms(graphs)
        self._remove_irrelevant()
        self._reorganize()

    # Generate all possible graphs of size 'group_size'
    def _generate_all_graphs(self):
        # handshake lemma
        num_edges = int(self._group_size * (self._group_size - 1) / 2.)
        num_bits = num_edges * 2 if self._is_directed else num_edges
        edge_iter = permutations if self._is_directed else combinations
        graph_type = nx.DiGraph if self._is_directed else nx.Graph

        graphs = {}
        for num in range(2 ** num_bits):
            g = graph_type()
            g.add_nodes_from(range(self._group_size))
            g.add_edges_from((x, y) for i, (x, y) in enumerate(edge_iter(range(self._group_size), 2)) if (2 ** i) & num)
            graphs[num] = g
        return graphs

    @staticmethod
    def _group_to_isomorphisms(graphs):
        isomorphisms = []
        keys = sorted(list(graphs.keys()))
        while keys:
            g1 = graphs[keys[0]]
            isomorphisms.append({num: graphs[num] for num in keys if nx.is_isomorphic(g1, graphs[num])})
            keys = [x for x in keys if x not in isomorphisms[-1]]
        return isomorphisms

    def _remove_irrelevant(self):
        isomorphisms = self._isomorphisms
        # Remove disconnected graphs
        irrelevant = [group_index for group_index, isomorphs in enumerate(isomorphisms) if not nx.is_connected(list(isomorphs.values())[0].to_undirected())]
        self._isomorphisms = [isomorphs for index, isomorphs in enumerate(isomorphisms) if index not in irrelevant]

    def _reorganize(self):
        self._isomorphisms = {min(isomorphs.keys()): isomorphs for isomorphs in self._isomorphisms}

    def num_2_motif(self):
        return {num: motif_num for motif_num, group in self._isomorphisms.items() for num in group}


def main(level, is_directed):
    fname = "%d_%sdirected" % (level, "" if is_directed else "un")
    print("Calculating ", fname)
    gs = IsomorphismGenerator(level, is_directed)
    with open(os.path.join(OUT_FOLDER_PATH, fname + ".pkl"), "wb") as pickle_file:
        pickle.dump(gs.num_2_motif(), pickle_file)
    print("Finished calculating ", fname)
    # for y in gs.values():
    #     print(list(map(lambda i: len(i.edges()), y.values())))


if __name__ == "__main__":
    main(3, False)
    main(3, True)
    main(4, False)
    main(4, True)
