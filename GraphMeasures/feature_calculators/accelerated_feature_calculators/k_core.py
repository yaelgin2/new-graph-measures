import networkx as nx

from ...features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta
from ...features_algorithms.accelerated_graph_features.src import k_core


class KCoreCalculator(NodeFeatureCalculator):
    def is_relevant(self):
        return True

    def _calculate(self, include: set):
        # K_core gets only undirected graphs
        if nx.is_directed(self._gnx):
            dgraph = self._gnx.to_undirected()
            self._features = k_core(dgraph)
        else:
            self._features = k_core(self._gnx)



feature_entry = {
    "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
}
