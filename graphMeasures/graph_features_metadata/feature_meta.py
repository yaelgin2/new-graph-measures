"""
Feature metadata module.

Defines the FeatureMeta dataclass, which stores a feature calculator
and its associated abbreviations. Provides a convenient interface
to compute features on a given graph.
"""
from dataclasses import dataclass
from typing import Set

from ..feature_calculators.feature_calculator import FeatureCalculator


@dataclass
class FeatureMeta:
    """
    Metadata container for a graph feature.

    Attributes:
        calculator (FeatureCalculator): The object that performs the feature computation.
        abbreviation_set (Set[str]): A set of short names (abbreviations) for the feature.

    Methods:
        compute(graph: GraphType) -> Any:
            Computes the feature values for the provided graph using the calculator.
    """
    calculator: FeatureCalculator
    abbreviation_set: Set[str]
