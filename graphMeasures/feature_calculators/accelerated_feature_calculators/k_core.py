"""
Class definitions for finding k-cores.
"""
import networkx as nx
from .src import k_core
from ...feature_calculators.node_features_calculators import NodeFeatureCalculator


class KCoreCalculator(NodeFeatureCalculator):
    """
    Calculator for finding k cores implementation.
    """
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        # K_core gets only undirected graphs
        if nx.is_directed(self._graph):
            dgraph = self._graph.to_undirected()
            self._features = k_core(dgraph)
        else:
            self._features = k_core(self._graph)
