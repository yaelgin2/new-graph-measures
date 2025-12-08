import unittest

from ..utils.utils import filter_graph_into_subraphs
from ...feature_calculators.feature_calculator_adapters.feature_calculator_adapter import FeatureMetaAdapterBase
from ...loggers import PrintLogger


class SpecificFeatureTest(unittest.TestCase):
    logger = PrintLogger()

    def _test_feature(self, feature_cls: FeatureMetaAdapterBase, configuration, graph, expected_result, is_max_connected=False):
        graph = filter_graph_into_subraphs(graph, is_max_connected)
        feature = feature_cls(graph, configuration, logger=self.logger)
        result = feature.build()

        self.assertEqual(result.keys(), expected_result.keys())
        for key in result.keys():
            self.assertEqual(result[key].keys(), expected_result[key].keys())
            for inner_key in result[key].keys():
                self.assertEqual(result[key][inner_key], expected_result[key][inner_key])