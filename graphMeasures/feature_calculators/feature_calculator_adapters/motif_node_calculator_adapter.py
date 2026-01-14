from graphMeasures.feature_calculators.feature_calculator_adapters.feature_calculator_adapter import FeatureMetaAdapterBase
from ..node_features_calculators.calculators.motifs_node_calculator import MotifsNodeCalculator

class MotifNodeCalculatorAdapter(FeatureMetaAdapterBase):
    def __init__(self, level: int = 3, calc_edges: bool = False, count_motifs: bool = False,):
        self.motif_size = level
        self.calc_edges = calc_edges
        self.count_motifs = count_motifs

    def __call__(self, graph, colores_loaded, configuration, logger):
        calc = MotifsNodeCalculator(level= self.motif_size, graph=graph, colores_loaded=colores_loaded,
                                    configuration=configuration, calc_edges=self.calc_edges,
                                    count_motifs=self.count_motifs, logger=logger)
        return calc