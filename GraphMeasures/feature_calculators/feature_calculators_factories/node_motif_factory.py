from GraphMeasures.feature_calculators.node_features_calculators.calculators.motif_node_calculator.motifs_node_calculator import MotifsNodeCalculator


class NodeMotifFactory:
    @staticmethod
    def create(level: int) -> MotifsNodeCalculator:
        return MotifsNodeCalculator(level=level, calc_edges=False)