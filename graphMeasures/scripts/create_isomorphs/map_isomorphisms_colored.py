import json
import os
import pickle
from itertools import combinations, permutations

import networkx as nx
from bitstring import BitArray

OUT_FOLDER_PATH = r"graphMeasures\feature_calculators\node_features_calculators\calculators\motif_variations"

class IsomorphismGenerator:
    def __init__(self, group_size, is_directed):
        self._group_size = group_size
        self._is_directed = is_directed
        self._graphs = self._generate_all_graphs()
        self._group_to_isomorphisms()
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
            all_pairs = list(edge_iter(range(self._group_size), 2))
            g.add_edges_from((x, y) for i, (x, y) in enumerate(all_pairs) if ((2 ** (len(all_pairs)-1)) >> i) & num)
            graphs[num] = g
        return graphs

    def _group_to_isomorphisms(self):
        isomorphisms = []

        keys = sorted(list(self._graphs.keys()))
        while keys:
            g1 = self._graphs[keys[0]]
            isomorphisms.append({})
            for num in keys:
                if nx.is_isomorphic(g1, self._graphs[num]):
                    isomorphisms[-1][num] = (self._graphs[num])

            keys = [x for x in keys if x not in isomorphisms[-1]]
        self._isomorphisms = isomorphisms

    def _remove_irrelevant(self):
        isomorphisms = self._isomorphisms
        # Remove disconnected graphs
        irrelevant = [group_index for group_index, isomorphs in enumerate(isomorphisms) if not nx.is_connected(list(isomorphs.values())[0].to_undirected())]
        self._isomorphisms = [isomorphs for index, isomorphs in enumerate(isomorphisms) if index not in irrelevant]

    def _reorganize(self):
        self._isomorphisms = {min(isomorphs.keys()): isomorphs for isomorphs in self._isomorphisms}

    def motif_to_minimal_motif_and_permutations(self):
        motif_to_minimal_motif = {}
        for minimal_motif, motifs in self._isomorphisms.items():
            for motif_number, motif in motifs.items():
                mappings_to_minimal = []
                for permutation in permutations(motif.nodes()):
                    func = permutations if self._is_directed else combinations
                    # Reversing is a technical issue. We saved our node variations files
                    bit_form = BitArray(motif.has_edge(n1, n2) for n1, n2 in func(permutation, 2))
                    if bit_form.uint == minimal_motif:
                        mappings_to_minimal.append(permutation)
                motif_to_minimal_motif[motif_number] = (minimal_motif, mappings_to_minimal)
        return motif_to_minimal_motif

def main(level, is_directed):
    fname = os.path.join(OUT_FOLDER_PATH, "%d_%sdirected_colored" % (level, "" if is_directed else "un"))
    print("Calculating ", fname)
    gs = IsomorphismGenerator(level, is_directed)
    print(gs.motif_to_minimal_motif_and_permutations())
    pickle.dump(gs.motif_to_minimal_motif_and_permutations(), open(fname + ".pkl", "wb"))
    print("Finished calculating ", fname)


if __name__ == "__main__":
    main(3, False)
    main(3, True)
    main(4, False)
    main(4, True)
