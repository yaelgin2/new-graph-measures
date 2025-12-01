"""
Feature metadata module.

Defines the FeatureMeta dataclass, which stores a feature calculator
and its associated abbreviations. Provides a convenient interface
to compute features on a given graph.
"""
from dataclasses import dataclass
from typing import Set

from ..types import GraphType


@dataclass
class FeatureMeta:
    """
    Metadata container for a graph feature.

    Attributes:
        calculator (FeatureCalculator): The object that performs the feature computation.
        abbriviation_set (Set[str]): A set of short names (abbreviations) for the feature.

    Methods:
        compute(graph: GraphType) -> Any:
            Computes the feature values for the provided graph using the calculator.
    """
    calculator: FeatureCalculator
    abbriviation_set: Set[str]

    def compute(self, graph: GraphType):
        """
        Run the computing function  of the feature calculator.
        :param graph:
        :return:
        """
        return self.calculator.compute(graph)
