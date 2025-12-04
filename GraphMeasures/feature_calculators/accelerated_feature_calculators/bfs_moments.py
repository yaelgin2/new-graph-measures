"""
Class definition for BfsMomentsCalculator.
"""
from ..node_features_calculators import NodeFeatureCalculator
from ...feature_calculators.accelerated_feature_calculators.src import bfs_moments


class BfsMomentsCalculator(NodeFeatureCalculator):
    """
    Bfs moments calculator implementation.
    """
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        self._features = bfs_moments(self._graph)
