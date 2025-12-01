from dataclasses import dataclass
from typing import Set

@dataclass
class FeatureMeta:
    calculator: FeatureCalculator
    abbriviation_set: Set[str]

    def compute(self, graph):
        return self.calculator.compute(graph)