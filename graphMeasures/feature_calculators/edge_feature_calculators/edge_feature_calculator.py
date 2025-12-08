"""
Edge feature calculator module.

Provides the EdgeFeatureCalculator class, which extends FeatureCalculator
to compute features specifically for graph edges.
"""
from typing import List

import numpy as np

from ..feature_calculator import FeatureCalculator


class EdgeFeatureCalculator(FeatureCalculator):
    """
    Feature calculator for graph edges.

    Extends the FeatureCalculator base class to provide edge-level
    feature computation. Features are stored in a dictionary keyed
    by edges, and can be accessed in a specified order or in the
    default sorted order of edges.
    """

    def _params_order(self, input_order: List = None) -> List:
        """
        Determine the order of edges for feature computation.

        Args:
            input_order (list, optional): Explicit order of edges to use.
                If None, the edges are sorted by default.

        Returns:
            list: List of edges in the order to process for feature computation.
        """
        if input_order is None:
            return sorted(self._graph.edges())
        return input_order


    def _get_feature(self, element) -> np.ndarray:
        """
        Retrieve the computed feature for a specific edge.

        Args:
            element: The edge (tuple of nodes) for which to retrieve the feature.

        Returns:
            np.ndarray: The computed feature values for the given edge.
        """
        return np.array(self.features[element])
