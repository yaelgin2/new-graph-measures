from .accelerated_motifs_calculator import AcceleratedMotifsCalculator


class EdgeMotifFactory:
    @staticmethod
    def create(level: int) -> AcceleratedMotifsCalculator:
        return AcceleratedMotifsCalculator(level=level, calc_edges=True)