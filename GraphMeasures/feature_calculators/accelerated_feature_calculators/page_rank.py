"""
Page rank calculator definition.
"""
from .src import node_page_rank
from ...feature_calculators.node_features_calculators import NodeFeatureCalculator


class PageRankCalculator(NodeFeatureCalculator):
    """
    Page rank calculator implementation.
    """
    def __init__(self, *args, alpha=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = alpha

    def is_relevant(self):
        # Undirected graphs will be converted to a directed
        #       graph with two directed edges for each undirected edge.
        return True

    def _calculate(self, include: set):
        self._features = node_page_rank(self._graph, dumping=self._alpha)
