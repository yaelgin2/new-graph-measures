# pylint: disable=line-too-long
"""
Features metadata module.

This module defines the FeaturesMetadata class, which contains the
regular (non-accelerated) versions of graph feature calculators.
Each feature is wrapped in a FeatureMeta object, providing both the
calculator and a set of abbreviations for identification.

Features are organized by type and computation duration:

- Short-duration features: Average neighbor degree, General, Louvain,
  Hierarchy energy
- Medium-duration features: Fiedler vector, K core, Motif 3, Closeness
  centrality, Attraction basin, Eccentricity, Load centrality, Page rank
- Long-duration features: Betweenness centrality, BFS moments,
  Communicability betweenness centrality, Flow, Motif 4

For accelerated versions of these features, see
accelerated_features_metadata.py.
"""

from .base_features_metadata import BaseFeaturesMeta
from .feature_meta import FeatureMeta
from graphMeasures.feature_calculators.feature_calculator_adapters.motif_node_calculator_adapter import \
    MotifNodeCalculatorAdapter


class FeaturesMetadata(BaseFeaturesMeta): # pylint: disable=too-few-public-methods

    """
    The following are the implemented features.
    This file includes the regular versions for feature calculators, whereas
    the similar accelerated_features_metadata.py file includes the the accelerated features
    (for the features that have the accelerated version).
    For each feature, the comment to the right describes to which graph they are intended.
    We split the features into 3 classes by duration, below.
    """

    def __init__(self):
        self.node_level = {
            "motif3": FeatureMeta(MotifNodeCalculatorAdapter(level=3, calc_edges=False, count_motifs=False), {"m3"}),  # Any
            "edges_motif3": FeatureMeta(MotifNodeCalculatorAdapter(level=3, calc_edges=True, count_motifs=False), {"m3"}),  # Any
            "count_motif3": FeatureMeta(MotifNodeCalculatorAdapter(level=3, calc_edges=False, count_motifs=True), {"m3"}),
            "motif4": FeatureMeta(MotifNodeCalculatorAdapter(level=4, calc_edges=False, count_motifs=False), {"m4"}),  # Any
            "edges_motif4": FeatureMeta(MotifNodeCalculatorAdapter(level=4, calc_edges=True, count_motifs=False), {"m4"}),  # Any
            "count_motif4": FeatureMeta(MotifNodeCalculatorAdapter(level=4, calc_edges=False, count_motifs=True), {"m4"}),
        }
        super().__init__(self.node_level)

    # Features by duration:
    # Short:
    #     - Average neighbor degree
    #     - General
    #     - Louvain
    #     - Hierarchy energy
    # Medium:
    #     - Fiedler vector
    #     - K core
    #     - Motif 3
    #     - Closeness centrality
    #     - Attraction basin
    #     - Eccentricity
    #     - Load centrality
    #     - Page rank
    # Long:
    #     - Betweenness centrality
    #     - BFS moments
    #     - Communicability betweenness centrality
    #     - Flow
    #     - Motif 4
