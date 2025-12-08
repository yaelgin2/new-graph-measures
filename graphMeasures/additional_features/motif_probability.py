"""
Motif Probability Computation Module.

This module provides the `MotifProbability` class, which computes:
    • Probabilities of observing specific graph motifs (size-3 and size-4),
    • Expected motif counts in Erdős–Rényi–type random graphs,
    • Classification of motifs based on the number of clique vertices involved.

Motif variations (isomorphism classes) are loaded from pre-computed pickle files.
The class supports both directed and undirected graphs and accounts for:
    • Number of edges per motif,
    • Motif symmetries (isomorphic variations),
    • Clique vs. non-clique motif cases,
    • Directed vs. undirected edge configurations.

This is primarily used to estimate motif frequencies and compare observed motif
counts to expected values under a random-graph baseline.
"""
import pickle

import numpy as np
from scipy.special import comb

from graphMeasures.configuration.configuration_keys import (KEY_DIRECTED_VARIATIONS_3, KEY_UNDIRECTED_VARIATIONS_3,
                                                            KEY_DIRECTED_VARIATIONS_4, KEY_UNDIRECTED_VARIATIONS_4)


class MotifProbability: #pylint: disable=too-many-instance-attributes
    """
    Motif probability calculator.
    """
    def __init__(self, size, edge_probability: float, clique_size, directed, configuration):
        self._is_directed = directed
        self._size = size
        self._probability = edge_probability
        self._cl_size = clique_size
        self._build_variations(configuration)
        self._motif_index_to_edge_num = {"motif3": self._motif_num_to_number_of_edges(3),
                                         "motif4": self._motif_num_to_number_of_edges(4)}
        self._gnx = None
        self._labels = {}

    def _build_variations(self, configuration):
        path3 = configuration[KEY_DIRECTED_VARIATIONS_3] \
            if self._is_directed else configuration[KEY_UNDIRECTED_VARIATIONS_3]
        self._motif3_variations = pickle.load(open(path3, "rb"))
        path4 = configuration[KEY_DIRECTED_VARIATIONS_4] \
            if self._is_directed else configuration[KEY_UNDIRECTED_VARIATIONS_4]
        self._motif4_variations = pickle.load(open(path4, "rb"))

    def _motif_num_to_number_of_edges(self, level):
        motif_edge_num_dict = {}
        if level == 3:
            variations = self._motif3_variations
        elif level == 4:
            variations = self._motif4_variations
        else:
            return None
        for bit_sec, motif_num in variations.items():
            motif_edge_num_dict[motif_num] = bin(bit_sec).count('1')
        return motif_edge_num_dict

    def get_2_clique_motifs(self, level):
        """
        Return motif IDs corresponding to motifs containing exactly 2 clique vertices.

        Args:
            level (int): Motif size (3 or 4).

        Returns:
            List[int]: IDs of motifs with 2-clique vertices.
        """
        if level == 3:
            variations = self._motif3_variations
            motif_3_with_2_clique = []
            for number in variations.keys():
                if variations[number] is None:
                    continue
                bitnum = np.binary_repr(number, 6) \
                    if self._is_directed else np.binary_repr(number, 3)
                if self._is_directed:
                    if all([int(x) for x in [bitnum[0], bitnum[2]]]
                           + [(variations[number]) not in motif_3_with_2_clique]):
                        motif_3_with_2_clique.append(variations[number])
                else:
                    if variations[number] not in motif_3_with_2_clique:
                        motif_3_with_2_clique.append(variations[number])
            return motif_3_with_2_clique
        if level == 4:
            variations = self._motif4_variations
            motif_4_with_2_clique = []
            for number in variations.keys():
                if variations[number] is None:
                    continue
                bitnum = np.binary_repr(number, 12)\
                    if self._is_directed else np.binary_repr(number, 6)
                if self._is_directed:
                    if all([int(x) for x in [bitnum[0], bitnum[3]]] +
                           [(variations[number] + 13) not in motif_4_with_2_clique]):
                        motif_4_with_2_clique.append(variations[number] + 13)
                else:
                    if (variations[number] + 2) not in motif_4_with_2_clique:
                        motif_4_with_2_clique.append(variations[number] + 2)
            return motif_4_with_2_clique
        return []

    def get_3_clique_motifs(self, level):
        """
        Return motif IDs corresponding to 3-clique motifs (triangles) for the given level.

        Args:
            level (int): Motif size (3 or 4).

        Returns:
        List[int]: IDs of motifs containing a 3-clique.
        """
        if level == 3:
            if self._is_directed:
                return [12]
            return [1]
        if level == 4:
            variations = self._motif4_variations
            motif_4_with_3_clique = []
            for number in variations.keys():
                if variations[number] is None:
                    continue
                bitnum = np.binary_repr(number, 12) \
                    if self._is_directed else np.binary_repr(number, 6)
                if self._is_directed:
                    if all([int(x) for x in [bitnum[0], bitnum[1], bitnum[3],
                                             bitnum[4], bitnum[6], bitnum[7]]] +
                           [(variations[number] + 13) not in motif_4_with_3_clique]):
                        motif_4_with_3_clique.append(variations[number] + 13)
                else:
                    if all([int(x) for x in [bitnum[5], bitnum[4], bitnum[2]]] +
                           [(variations[number] + 2) not in motif_4_with_3_clique]):
                        motif_4_with_3_clique.append(variations[number] + 2)
            return motif_4_with_3_clique
        return []

    def _for_probability_calculation(self, motif_index):
        if self._is_directed:
            if motif_index > 12:
                motif_index -= 13
                variations = self._motif4_variations
                num_edges = self._motif_index_to_edge_num['motif4'][motif_index]
                num_max = 12
                flag = 4
            else:
                variations = self._motif3_variations
                num_edges = self._motif_index_to_edge_num['motif3'][motif_index]
                num_max = 6
                flag = 3
        else:
            if motif_index > 1:
                motif_index -= 2
                variations = self._motif4_variations
                num_edges = self._motif_index_to_edge_num['motif4'][motif_index]
                num_max = 6
                flag = 4
            else:
                variations = self._motif3_variations
                num_edges = self._motif_index_to_edge_num['motif3'][motif_index]
                num_max = 3
                flag = 3
        return motif_index, variations, num_edges, num_max, flag

    def motif_probability_non_clique_vertex(self, motif_index):
        """
        Compute the probability of observing a motif at a non-clique vertex.

        Parameters
        ----------
        motif_index : int
            Identifier of the motif type.

        Returns
        -------
        float
            Probability of the motif occurring, accounting for all
            isomorphic variations and edge configurations.
        """
        motif_index, variations, num_edges, num_max, _ = (
            self._for_probability_calculation(motif_index))
        motifs = []
        for original_number in variations.keys():
            if variations[original_number] == motif_index:
                motifs.append(np.binary_repr(original_number, num_max))
        num_isomorphic = len(motifs)
        prob = (num_isomorphic * (self._probability ** num_edges) *
                ((1 - self._probability) ** (num_max - num_edges)))
        return prob

    def motif_expected_non_clique_vertex(self, motif_index):
        """
        Return the expected count of a motif at a non-clique vertex.

        Parameters
        ----------
        motif_index : int
            Motif identifier.

        Returns
        -------
        float
            Expected number of occurrences based on random-graph probability.
        """
        if self._is_directed:
            if motif_index > 12:
                to_choose = 4
            else:
                to_choose = 3
        else:
            if motif_index > 1:
                to_choose = 4
            else:
                to_choose = 3
        prob = self.motif_probability_non_clique_vertex(motif_index)
        return comb(self._size - 1, to_choose - 1) * prob

    @staticmethod
    def _second_condition(binary_motif, clique_edges):
        return all(int(binary_motif[i]) for i in clique_edges)

#pylint: disable=too-many-arguments
    def _clique_edges(self, flag: int, i: int) -> list[int]:
        """
        Given i clique motifs (plus one we focus on) in fixed indices, return
        the list of edges that must appear in the motif.
        """

        # Edge lookup tables for readability
        directed_flag3 = {
            0: [],
            1: [0, 2],
            "default": list(range(6)),
        }

        directed_other = {
            0: [],
            1: [0, 3],
            2: [0, 1, 3, 4, 6, 7],
            "default": list(range(12)),
        }

        undirected_flag3 = {
            0: [],
            1: [0],
            "default": list(range(3)),
        }

        undirected_other = {
            0: [],
            1: [0],
            2: [0, 1, 3],
            "default": list(range(6)),
        }

        # Choose the correct lookup table
        if self._is_directed:
            table = directed_flag3 if flag == 3 else directed_other
        else:
            table = undirected_flag3 if flag == 3 else undirected_other

        # Return specific rule or default rule
        return table.get(i, table["default"])


    # pylint: disable=too-many-positional-arguments
    def _specific_combination_motif_probability(self, motif_index, num_edges,
                                                num_max, flag, variations, i):
        # P(motif|i clique vertices except for the vertex on which we focus)
        clique_edges = self._clique_edges(flag, i)
        motifs = []
        for original_number in variations.keys():
            if variations[original_number] == motif_index:
                b = np.binary_repr(original_number, num_max)
                if self._second_condition(b, clique_edges):
                    motifs.append(b)
        num_iso = len(motifs)
        num_already_there = (i + 1) * i if self._is_directed else (i + 1) * i / 2
        return num_iso * self._probability ** (num_edges - num_already_there) * (
                    1 - self._probability) ** (num_max - num_edges)

    def motif_probability_clique_vertex(self, motif_index):
        """
        Compute P(motif) when the center vertex *is* part of the clique.

        Parameters
        ----------
        motif_index : int

        Returns
        -------
        float
            Probability of observing the motif for a clique vertex.
        """
        motif_ind, variations, num_edges, num_max, flag = (
            self._for_probability_calculation(motif_index))
        clique_non_clique = []
        for i in range(flag if self._cl_size > 1 else 1):
            # Probability that a specific set of vertices contains exactly i + 1 clique vertices.
            if i == 1:
                indicator = 1 if motif_index in self.get_2_clique_motifs(flag) else 0
            elif i == 2:
                indicator = 1 if motif_index in self.get_3_clique_motifs(flag) else 0
            elif i == 3:
                indicator = 1 if motif_index == 211 else 0
            else:
                indicator = 1
            if not indicator:
                clique_non_clique.append(0)
                continue
            cl_ncl_comb_prob = (comb(max(self._cl_size - 1, 0), i) *
                                comb(self._size - max(self._cl_size, 1),
                                     flag - 1 - i) / float(
                                   comb(self._size - 1, flag - 1)))
            spec_comb_motif_prob = self._specific_combination_motif_probability(
                motif_ind, num_edges, num_max, flag, variations, i)

            clique_non_clique.append(cl_ncl_comb_prob * spec_comb_motif_prob)
        prob = sum(clique_non_clique)
        return prob

    def motif_expected_clique_vertex(self, motif_index):
        """
        Expected number of motifs centered at a clique vertex.

        Parameters
        ----------
        motif_index : int

        Returns
        -------
        float
            Expected motif count.
        """
        if self._is_directed:
            if motif_index > 12:
                to_choose = 4
            else:
                to_choose = 3
        else:
            if motif_index > 1:
                to_choose = 4
            else:
                to_choose = 3
        prob = self.motif_probability_clique_vertex(motif_index)
        return comb(self._size - 1, to_choose - 1) * prob

    def clique_non_clique_angle(self, motifs):
        """
        Compute the angle between expected clique-motif vector
        and expected non-clique-motif vector.

        Parameters
        ----------
        motifs : list[int]
            List of motif indices.

        Returns
        -------
        float
            Angle (in radians) between the two vectors.
        """
        clique_vec = [self.motif_expected_clique_vertex(m) for m in motifs]
        non_clique_vec = [self.motif_expected_non_clique_vertex(m) for m in motifs]
        return self._angle(clique_vec, non_clique_vec)

    def clique_non_clique_zscored_angle(self, mean_vector, std_vector, motifs):
        """
        Compute the angle between clique and non-clique expected motif vectors
        after Z-scoring them.

        Parameters
        ----------
        mean_vector : array-like
            Expected motif means used to normalize.
        std_vector : array-like
            Standard deviations for normalization.
        motifs : list[int]

        Returns
        -------
        float
                Angle (in radians) between the Z-normalized vectors.
        """
        clique_vec = np.array([self.motif_expected_clique_vertex(m) for m in motifs])
        non_clique_vec = np.array([self.motif_expected_non_clique_vertex(m) for m in motifs])
        normed_clique_vec = np.divide(clique_vec - mean_vector, std_vector)
        normed_non_clique_vec = np.divide(non_clique_vec - mean_vector, std_vector)
        return self._angle(normed_clique_vec, normed_non_clique_vec)

    @staticmethod
    def _angle(v1, v2):
        """
        Compute the angle between two vectors.

        Parameters
        ----------
        v1 : array-like
        v2 : array-like

        Returns
        -------
        float
            Angle in radians.
        """
        cos = np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(cos)
