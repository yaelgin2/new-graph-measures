from .accelerated_motifs_calculator import AcceleratedMotifsCalculator


class AcceleratedNodeMotifFactory:
    @staticmethod
    def create(level: int) -> AcceleratedMotifsCalculator:
        return AcceleratedMotifsCalculator(level=level, calc_edges=False)