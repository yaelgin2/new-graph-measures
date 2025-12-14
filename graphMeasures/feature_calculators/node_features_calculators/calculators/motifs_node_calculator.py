import os
import pickle
from itertools import permutations, combinations

import networkx as nx
import numpy as np
from bitstring import BitArray

from graphMeasures.configuration.configuration_keys import KEY_DIRECTED_VARIATIONS_3, KEY_UNDIRECTED_VARIATIONS_3, \
    KEY_UNDIRECTED_VARIATIONS_4, KEY_DIRECTED_VARIATIONS_4
from graphMeasures.exceptions.exception_codes import VARIATION_FILE_NOT_FOUND_EXCEPTION, CONFIGURATION_MISSING_KEY
from graphMeasures.exceptions.graph_measures_exception import GraphMeasuresException
from graphMeasures.feature_calculators.node_features_calculators.node_feature_calculator import NodeFeatureCalculator

CUR_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(os.path.dirname(CUR_PATH))
VERBOSE = False
DEBUG = False


class MotifsNodeCalculator(NodeFeatureCalculator):
    def __init__(self, *args, level=3, calc_edges=False, **kwargs):
        super(MotifsNodeCalculator, self).__init__(*args, **kwargs)
        assert level in [3, 4], f"Unsupported motif level {level}"
        self._level = level
        self._node_variations = {}
        self._all_motifs = None
        self._get_name += f"_{self._level}"
        self._graph = self._graph.copy()
        self._load_variations()
        self.calc_edges = calc_edges

    def is_relevant(self):
        return True

    @classmethod
    def get_name(cls, level=None, calc_edges=None):
        print_name = super(MotifsNodeCalculator, cls).get_name()
        if level is None:
            return print_name
        return f"{print_name}_{level}"

    def _load_variations_file(self):
        try:
            if self._level == 3:
                if self._colores_loaded:
                    fname = self._configuration[KEY_DIRECTED_COLORED_VARIATIONS_3] \
                        if self._graph.is_directed() else self._configuration[KEY_UNDIRECTED_COLORED_VARIATIONS_3]
                else:
                    fname = self._configuration[KEY_DIRECTED_VARIATIONS_3] \
                        if self._graph.is_directed() else self._configuration[KEY_UNDIRECTED_VARIATIONS_3]
            if self._level == 4:
                if self._colores_loaded:
                    fname = self._configuration[KEY_DIRECTED_COLORED_VARIATIONS_4] \
                        if self._graph.is_directed() else self._configuration[KEY_UNDIRECTED_COLORED_VARIATIONS_4]
                else:
                    fname = self._configuration[KEY_DIRECTED_VARIATIONS_4] \
                        if self._graph.is_directed() else self._configuration[KEY_UNDIRECTED_VARIATIONS_4]
        except KeyError as e:
            raise GraphMeasuresException(f"Configuration missing key {e.args[0]}", CONFIGURATION_MISSING_KEY)
        if not os.path.isfile(fname):
            raise GraphMeasuresException(f"File {fname} not found.", VARIATION_FILE_NOT_FOUND_EXCEPTION)
        with open(fname, "rb") as variation_file:
            variations = pickle.load(variation_file)
        return variations

    def _load_variations(self):
        self._node_variations, self._node_permutations = self._load_variations_file()
        self._all_motifs = set(self._node_variations.values())

    # passing on all:
    #  * undirected graph: combinations [(n*(n-1)/2) combs - handshake lemma]
    #  * directed graph: permutations [(n*(n-1) perms - handshake lemma with respect to order]
    # checking whether the edge exist in the graph - and construct a bitmask of the existing edges
    def _get_group_number(self, nbunch):
        func = permutations if self._graph.is_directed() else combinations
        # Reversing is a technical issue. We saved our node variations files
        bit_form = BitArray(self._graph.has_edge(n1, n2) for n1, n2 in func(nbunch, 2))
        bit_form.reverse()
        return bit_form.uint

    # implementing the "Kavosh" algorithm for subgroups of length 3
    def _get_motif3_sub_tree(self, root):
        visited_vertices = {root: 0}
        visited_index = 1

        # variation - two neighbors of the root
        first_neighbors = set(nx.all_neighbors(self._graph, root))
        for n1 in first_neighbors:
            visited_vertices[n1] = visited_index
            visited_index += 1

        for n1, n2 in combinations(first_neighbors, 2):
            if (visited_vertices[n1] < visited_vertices[n2]) and \
                    not (self._graph.has_edge(n1, n2) or self._graph.has_edge(n2, n1)):
                yield [root, n1, n2]

        # variation - one vertex of depth 1, one of depth 2

        for n1 in first_neighbors:
            last_neighbors = set(nx.all_neighbors(self._graph, n1))
            for n2 in last_neighbors:
                if n2 in visited_vertices:
                    if visited_vertices[n1] < visited_vertices[n2]:
                        yield [root, n1, n2]
                else:
                    visited_vertices[n2] = visited_index
                    visited_index += 1
                    yield [root, n1, n2]

    # implementing the "Kavosh" algorithm for subgroups of length 4
    def _get_motif4_sub_tree(self, root):
        visited_vertices = {root: 0}

        # variation - three neighbors of the root
        neighbors_first_deg = set(nx.all_neighbors(self._graph, root))
        neighbors_first_deg = list(neighbors_first_deg)

        for n1 in neighbors_first_deg:
            visited_vertices[n1] = 1
        for n1, n2, n3 in combinations(neighbors_first_deg, 3):
            group = [root, n1, n2, n3]
            yield group

        # variations - depths 1, 1, 2 and 1, 2, 2
        for n1 in neighbors_first_deg:
            # all neighbors adjacent to vertices of depth 1, that are not of depth 1 themselves, are of depth 2.
            neighbors_sec_deg = set(nx.all_neighbors(self._graph, n1))
            neighbors_sec_deg = list(neighbors_sec_deg)
            for n in neighbors_sec_deg:
                if n not in visited_vertices:
                    visited_vertices[n] = 2

            # variation - depths 1, 1, 2
            for n2 in neighbors_sec_deg:
                for n11 in neighbors_first_deg:
                    if visited_vertices[n2] == 2 and n1 != n11:
                        edge_exists = (self._graph.has_edge(n2, n11) or self._graph.has_edge(n11, n2))
                        # avoid double-counting due to two paths from root to n2 - from n1 and from n11.
                        if (not edge_exists) or (edge_exists and n1 < n11):
                            group = [root, n1, n11, n2]
                            yield group

            # variation - depths 1, 2, 2
            for comb in combinations(neighbors_sec_deg, 2):
                if visited_vertices[comb[0]] == 2 and visited_vertices[comb[1]] == 2:
                    group = [root, n1, comb[0], comb[1]]
                    yield group

        # variation - one vertex of each depth (root, 1, 2, 3)
        for n1 in neighbors_first_deg:
            neighbors_sec_deg = set(nx.all_neighbors(self._graph, n1))
            neighbors_sec_deg = list(neighbors_sec_deg)
            for n2 in neighbors_sec_deg:
                if visited_vertices[n2] == 1:
                    continue

                for n3 in set(nx.all_neighbors(self._graph, n2)):
                    if n3 not in visited_vertices:
                        visited_vertices[n3] = 3
                        if visited_vertices[n2] == 2:
                            group = [root, n1, n2, n3]
                            yield group
                    else:
                        if visited_vertices[n3] == 1:
                            continue

                        if visited_vertices[n3] == 2 and not (self._graph.has_edge(n1, n3) or self._graph.has_edge(n3, n1)):
                            group = [root, n1, n2, n3]
                            yield group

                        elif visited_vertices[n3] == 3 and visited_vertices[n2] == 2:
                            group = [root, n1, n2, n3]
                            yield group

    def _order_by_degree(self, gnx=None):
        if gnx is None:
            gnx = self._graph
        return sorted(gnx, key=lambda n: len(list(nx.all_neighbors(gnx, n))), reverse=True)

    def _calculate_motif(self):
        # consider first calculating the nth neighborhood of a node
        # and then iterate only over the corresponding graph
        motif_func = self._get_motif3_sub_tree if self._level == 3 else self._get_motif4_sub_tree
        sorted_nodes = self._order_by_degree()
        for node in sorted_nodes:
            for group in motif_func(node):
                group_num = self._get_group_number(group)
                motif_num = self._node_variations[group_num]
                yield group, group_num, motif_num
            if VERBOSE:
                self._logger.debug("Finished node: %s" % node)
            self._graph.remove_node(node)

    def _update_edges(self, group, motif_num):
        # we should save it in an array
        for v1 in group:
            for v2 in group:
                if v1 != v2 and self._graph.has_edge(v1, v2):
                    print("edge:", v1, v2, "motif num:", motif_num)

    def _update_nodes_group(self, group, motif_num):
        for node in group:
            self._features[node][motif_num] += 1

    def _calculate(self, include=None):
        m_graph = self._graph.copy()
        motif_counter = {motif_number: 0 for motif_number in self._all_motifs}

        if self.calc_edges:
            self._features = {edge: motif_counter.copy() for edge in self._graph.edges()}
        else:
            self._features = {node: motif_counter.copy() for node in self._graph}

        for i, (group, group_num, motif_num) in enumerate(self._calculate_motif()):
            if self.calc_edges:
                self._update_edges(group, motif_num)
            else:
                self._update_nodes_group(group, motif_num)

            if (i + 1) % 1000 == 0 and VERBOSE:
                self._logger.debug("Groups: %d" % i)

        # print('Max num of duplicates:', max(self._double_counter.values()))
        # print('Number of motifs counted twice:', len(self._double_counter))

        self._graph = m_graph

    def _get_feature(self, element):
        all_motifs = self._all_motifs.difference({None})
        cur_feature = self._features[element]
        return np.array([cur_feature[motif_num] for motif_num in sorted(all_motifs)])
