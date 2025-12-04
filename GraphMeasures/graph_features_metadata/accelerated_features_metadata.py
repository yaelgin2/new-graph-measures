# pylint: disable=line-too-long
"""
Accelerated features metadata module.

This module defines the AcceleratedFeaturesMetadata class, which contains
the accelerated versions of graph feature calculators. Each feature is
wrapped in a FeatureMeta object, providing both the calculator and a set
of abbreviations for identification. Features are organized by duration
(short, medium, long) for convenience.

The accelerated versions typically use GPU or other optimizations where
available, while the regular features are defined in the features_metadata.py module.
"""
from .base_features_metadata import BaseFeaturesMeta
from .feature_meta import FeatureMeta
from ..feature_calculators.feature_calculator_adapters.accelerated_motif_node_calculator_adapter import \
    AcceleratedMotifNodeCalculatorAdapter


class AcceleratedFeaturesMetadata(BaseFeaturesMeta): # pylint: disable=too-few-public-methods

    """
    The following are the implemented features.
    This file includes the accelerated versions for the features which have
    the option, whereas the similar features_metadata.py file includes the regular versions.
    For each feature, the comment to the right describes to which graph they are intended.
    We split the features into 3 classes by duration, below.
    """

    def __init__(self, gpu=False, device=0):
        self.node_level = {
            "motif3_gpu": FeatureMeta(AcceleratedMotifNodeCalculatorAdapter(level=4, calc_edges=False, gpu=gpu, device=device), {"m3"}),  # Any
            "motif3_edges_gpu": FeatureMeta(AcceleratedMotifNodeCalculatorAdapter(level=3, calc_edges=True,  gpu=gpu, device=device, edges=True), {"m4"}),  # Any
            "motif4_gpu": FeatureMeta(AcceleratedMotifNodeCalculatorAdapter(level=4, calc_edges=False, gpu=gpu, device=device), {"m4"}),  # Any
            "motif4_edges_gpu": FeatureMeta(AcceleratedMotifNodeCalculatorAdapter(level=3, calc_edges=True,  gpu=gpu, device=device, edges=True), {"m4"}),  # Any
        }
        super().__init__(self.node_level)

        """
        Features by duration:
        Short:
            - Average neighbor degree
            - General
            - Louvain
            - Hierarchy energy
            - Motif 3
            - K core
            - Attraction basin
            - Page Rank 
            
        Medium:
            - Fiedler vector
            - Closeness centrality 
            - Eccentricity
            - Load centrality
            - BFS moments
            - Flow
            - Motif 4
        Long:
            - Betweenness centrality
            - Communicability betweenness centrality
        """
