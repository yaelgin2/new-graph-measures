"""
Flow calculator class definition.
"""
from .src import flow
from ...feature_calculators.node_features_calculators import NodeFeatureCalculator


class FlowCalculator(NodeFeatureCalculator):
    """
    Flow calculator implementation.
    """
    def __init__(self, *args, threshold=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._threshold = threshold

    def is_relevant(self):
        return self._graph.is_directed()

    def _calculate(self, include):
        self._features = flow(self._graph, threshold=self._threshold)
