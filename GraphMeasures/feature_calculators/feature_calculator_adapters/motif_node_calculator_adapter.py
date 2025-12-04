from GraphMeasures.feature_calculators.feature_calculator_adapters.feature_calculator_adapter import FeatureMetaAdapterBase
from ..node_features_calculators.calculators.motifs_node_calculator import MotifsNodeCalculator

class MotifNodeCalculatorAdapter(FeatureMetaAdapterBase):
    def __init__(self, level: int = 3, calc_edges: bool = False):
        self.motif_size = level
        self.calc_edges = calc_edges

    def __call__(self, graph):
        calc = MotifsNodeCalculator(graph, motif_size=self.motif_size, calc_edges=self.calc_edges)
        return calc.run()