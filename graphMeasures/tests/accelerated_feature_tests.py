from .test_cases.specific_feature_test import SpecificFeatureTest
from .utils.graph_builders import get_directed_graph, get_undirected_graph, get_undirected_motif0_graph, \
    get_undirected_motif1_graph, get_directed_motif0_graph, get_directed_motif1_graph, get_directed_motif12_graph, \
    get_directed_motif11_graph, get_directed_motif10_graph, get_directed_motif9_graph, get_directed_motif8_graph, \
    get_directed_motif7_graph, get_directed_motif6_graph, get_directed_motif5_graph, get_directed_motif4_graph, \
    get_directed_motif3_graph, get_directed_motif2_graph, get_empty_undirected_graph, get_empty_directed_graph
from ..feature_calculators.feature_calculator_adapters.accelerated_motif_node_calculator_adapter import \
    AcceleratedMotifNodeCalculatorAdapter
from ..feature_calculators.feature_calculator_adapters.motif_node_calculator_adapter import MotifNodeCalculatorAdapter

CONFIGURATION = {
  "directed_variations_3": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\3_directed.pkl",
  "undirected_variations_3": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\3_undirected.pkl",
  "directed_variations_4": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\4_directed.pkl",
  "undirected_variations_4": "graphMeasures\\feature_calculators\\node_features_calculators\\calculators\\motif_variations\\4_undirected.pkl"
}

class AcceleratedFeatureTests(SpecificFeatureTest):

    def test_edge_motifs3_directed_find_motif0_expected_success(self):
        expected_result = {0 : {0 : 1, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           1 : {0 : 1, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           2 : {0 : 1, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif0_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif1_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 1, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           1 : {0 : 0, 1 : 1, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           2 : {0 : 0, 1 : 1, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif1_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif2_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 1, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           1 : {0 : 0, 1 : 0, 2 : 1, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           2 : {0 : 0, 1 : 0, 2 : 1, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif2_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif3_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 1, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           1 : {0 : 0, 1 : 0, 2 : 0, 3 : 1, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           2 : {0 : 0, 1 : 0, 2 : 0, 3 : 1, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif3_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif4_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 1, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           1 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 1, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           2 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 1, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif4_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif5_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 1, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           1 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 1, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           2 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 1, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif5_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif6_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 1, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           1 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 1, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           2 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 1, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif6_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif7_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 1, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           1 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 1, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           2 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 1, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif7_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif8_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 1, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           1 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 1, 9 : 0, 10 : 0, 11 : 0, 12 : 0},
                           2 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 1, 9 : 0, 10 : 0, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif8_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif9_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 1, 10 : 0, 11 : 0, 12 : 0},
                           1 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 1, 10 : 0, 11 : 0, 12 : 0},
                           2 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 1, 10 : 0, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif9_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif10_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 1, 11 : 0, 12 : 0},
                           1 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 1, 11 : 0, 12 : 0},
                           2 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 1, 11 : 0, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif10_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif11_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 1, 12 : 0},
                           1 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 1, 12 : 0},
                           2 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 1, 12 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif11_graph(), expected_result)

    def test_edge_motifs3_directed_find_motif12_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 1},
                           1 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 1},
                           2 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0, 11 : 0, 12 : 1}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_motif12_graph(), expected_result)

    def test_edge_motifs3_directed(self):
        expected_results = {1: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
                            2: {0: 1, 1: 3, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
                            3: {0: 1, 1: 4, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
                            4: {0: 1, 1: 4, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
                            5: {0: 0, 1: 2, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
                            6: {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
                            7: {0: 1, 1: 3, 2: 0, 3: 1, 4: 0, 5: 0, 6: 1, 7: 0, 8: 1, 9: 0, 10: 0, 11: 0, 12: 0},
                            8: {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
                            9: {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
                            10: {0: 1, 1: 2, 2: 2, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0, 8: 1, 9: 0, 10: 0, 11: 0, 12: 0},
                            11: {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 0, 10: 0, 11: 0, 12: 0},
                            12: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 0, 9: 1, 10: 0, 11: 0, 12: 0},
                            13: {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
                            14: {0: 0, 1: 0, 2: 2, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
                            15: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0, 12: 0},
                            16: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 0, 9: 1, 10: 0, 11: 0, 12: 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_directed_graph(), expected_results)

    def test_edge_motifs3_empty_directed_expected_success(self):
        expected_result = {}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_empty_directed_graph(), expected_result)

    def test_edge_motifs3_empty_undirected_expected_success(self):
        expected_result = {}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_empty_undirected_graph(), expected_result)

    def test_edge_motifs3_undirected_expected_success(self):
        expected_result = {0 : {0 : 1, 1 : 0},
                           1 : {0 : 3, 1 : 0},
                           2 : {0 : 7, 1 : 1},
                           3 : {0 : 4, 1 : 2},
                           4 : {0 : 3, 1 : 0},
                           5 : {0 : 1, 1 : 0},
                           6 : {0 : 13, 1 : 2},
                           7 : {0 : 4, 1 : 1},
                           8 : {0 : 0, 1 : 0},
                           9 : {0 : 8, 1 : 0},
                           10 : {0 : 8, 1 : 0},
                           11 : {0 : 2, 1 : 0},
                           12 : {0 : 2, 1 : 0},
                           13 : {0 : 0, 1 : 0},
                           14 : {0 : 3, 1 : 0},
                           15 : {0 : 1, 1 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_undirected_graph(), expected_result)


    def test_edge_motifs3_undirected_find_motif0_expected_success(self):
        expected_result = {0 : {0 : 1, 1 : 0},
                           1 : {0 : 1, 1 : 0},
                           2 : {0 : 1, 1 : 0}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_undirected_motif0_graph(), expected_result)

    def test_edge_motifs3_undirected_find_motif1_expected_success(self):
        expected_result = {0 : {0 : 0, 1 : 1},
                           1 : {0 : 0, 1 : 1},
                           2 : {0 : 0, 1 : 1}}
        self._test_feature(AcceleratedMotifNodeCalculatorAdapter(3), CONFIGURATION, get_undirected_motif1_graph(), expected_result)

