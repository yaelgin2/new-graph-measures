"""
Node feature calculator module.

Provides the NodeFeatureCalculator class, which extends FeatureCalculator
to compute features at the node level. Includes methods to retrieve
features for specific nodes, determine processing order, and compute
edge-based node features derived from node-level data.
"""
from itertools import chain
from typing import List

import numpy as np

from ..feature_calculator import FeatureCalculator


class NodeFeatureCalculator(FeatureCalculator):
    """
    Feature calculator for graph nodes.

    Extends the FeatureCalculator base class to provide node-level
    feature computation. Supports retrieving node features, ordering
    nodes for processing, and generating edge-based features derived
    from node features.
    """

    def _params_order(self, input_order: List = None) -> List:
        """
        Determine the order of nodes for feature computation.

        Args:
            input_order (list, optional): Explicit order of nodes to use.
                If None, nodes are sorted by default.

        Returns:
            list: List of nodes in the order to process for feature computation.
        """
        if input_order is None:
            return sorted(self._graph)
        return input_order

    def _get_feature(self, element) -> np.ndarray:
        """
        Retrieve the computed feature for a specific node.

        Args:
            element: The node for which to retrieve the feature.

        Returns:
            np.ndarray: The computed feature values for the given node.
        """
        return np.array(self.features[element])

    def edge_based_node_feature(self):
        """
        Compute edge-level features based on node features.

        For each edge, combines the difference between the node features
        and the mean of the node features into a single list.

        Returns:
            dict: Dictionary mapping edges (tuples of nodes) to their
                corresponding edge-based feature lists.
         """
        nodes_dict = self.features
        edge_dict = {}
        for edge in self._graph.edges():
            n1_val = np.array(nodes_dict[edge[0]])
            n2_val = np.array(nodes_dict[edge[1]])

            edge_dict[edge] = list(chain(*zip(n1_val - n2_val, np.mean([n1_val, n2_val], axis=0))))
        return edge_dict

    def feature(self, element):
        """
        Convenience method to retrieve the feature of a node.

        Args:
            element: The node for which to retrieve the feature.

        Returns:
            np.ndarray: The computed feature values for the node.
        """
        return self._get_feature(element)
