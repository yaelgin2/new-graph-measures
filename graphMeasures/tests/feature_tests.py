import unittest

from .test_cases.specific_feature_test import SpecificFeatureTest
from .utils.graph_builders import get_directed_graph, get_undirected_graph
from ..feature_calculators.feature_calculator_adapters.motif_node_calculator_adapter import MotifNodeCalculatorAdapter

CONFIGURATION = {
  "directed_variations_3": "feature_calculators\\node_features_calculators\\calculators\\motif_variations\\3_directed.pkl",
  "undirected_variations_3": "feature_calculators\\node_features_calculators\\calculators\\motif_variations\\3_undirected.pkl",
  "directed_variations_4": "feature_calculators\\node_features_calculators\\calculators\\motif_variations\\4_directed.pkl",
  "undirected_variations_4": "feature_calculators\\node_features_calculators\\calculators\\motif_variations\\4_undirected.pkl"
}

class FeatureTests(SpecificFeatureTest):

    def test_edge_motifs3_directed(self):
        self._test_feature(MotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_graph(), {})

    def test_edge_motifs3_undirected(self):
        expected_solution = {0 : {0 : 1, 1 : 0},
                             1 : {2 : 3, 1 : 0},
                             2 : {0 : 7, 1 : 1},
                             3 : {0 : 4, 1 : 2},
                             4 : {0 : 3, 1 : 0},
                             5 : {0 : 1, 1 : 0},
                             6 : {0 : 13, 1 : 2},
                             7 : {0 : 3, 1 : 1},
                             8 : {0 : 4, 1 : 0},
                             9 : {0 : 8, 1 : 0},
                             10 : {0 : 15, 1 : 0},
                             11 : {0 : 4, 1 : 0},
                             12 : {0 : 2, 1 : 0},
                             13 : {0 : 4, 1 : 0},
                             14 : {0 : 0, 1 : 0},
                             15 : {0 : 0, 1 : 0}}
        self._test_feature(MotifNodeCalculatorAdapter(3), CONFIGURATION, get_undirected_graph(), {})

