"""
class definition for AttractorBasinCalculator.
"""
from ...feature_calculators.accelerated_feature_calculators.src import attraction_basin
from ...feature_calculators.node_features_calculators import NodeFeatureCalculator


class AttractorBasinCalculator(NodeFeatureCalculator):
    """
    Calculator for attracted basin.
    """
    def __init__(self, *args, alpha=2, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._default_val = float('nan')

    def is_relevant(self):
        return self._graph.is_directed()

    def _calculate(self, include: set):
        self._features = attraction_basin(self._graph, alpha=self._alpha)
