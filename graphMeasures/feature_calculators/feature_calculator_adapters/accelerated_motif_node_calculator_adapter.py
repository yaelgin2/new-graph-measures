from ..accelerated_feature_calculators.accelerated_motifs_calculator import \
    AcceleratedMotifsCalculator
from graphMeasures.feature_calculators.feature_calculator_adapters.feature_calculator_adapter import FeatureMetaAdapterBase

class AcceleratedMotifNodeCalculatorAdapter(FeatureMetaAdapterBase):
    def __init__(self, level: int = 3, calc_edges: bool = False,
                 gpu: bool = False, device = 0,  edges: bool =True):
        self.motif_size = level
        self.calc_edges = calc_edges
        self.gpu = gpu
        self.device = device
        self.edges = edges

    def __call__(self, graph, colores_loaded, configuration, logger):
        calc = AcceleratedMotifsCalculator(graph=graph, colores_loaded=colores_loaded, configuration=configuration,
                                           motif_size=self.motif_size, calc_edges=self.calc_edges, gpu=self.gpu,
                                           device=self.device, edges=self.edges, logger=logger)
        return calc