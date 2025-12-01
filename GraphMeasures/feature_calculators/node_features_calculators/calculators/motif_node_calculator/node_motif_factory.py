from .motifs_node_calculator import MotifsNodeCalculator


class NodeMotifFactory:
    @staticmethod
    def create(level: int) -> MotifsNodeCalculator:
        return MotifsNodeCalculator(level=level, calc_edges=False)