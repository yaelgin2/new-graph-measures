from .accelerated_motifs_calculator import AcceleratedMotifsCalculator


class EdgeMotifFactory:
    @staticmethod
    def create(level: int, gpu: bool, device: int, edges: bool=True) -> AcceleratedMotifsCalculator:
        return AcceleratedMotifsCalculator(level=level, calc_edges=True,  gpu=gpu, device=device, edges=edges)