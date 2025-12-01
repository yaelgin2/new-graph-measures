from .base_features_metadata import BaseFeaturesMeta
from .feature_meta import FeatureMeta


class FeaturesMetadata(BaseFeaturesMeta):
    """
    The following are the implemented features. This file includes the regular versions for feature calculators, whereas
    the similar accelerated_features_metadata.py file includes the the accelerated features (for the features that have the
    accelerated version).
    For each feature, the comment to the right describes to which graph they are intended.
    We split the features into 3 classes by duration, below.
    """

    def __init__(self):
        self.NODE_LEVEL = {
            "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),  # Any
            "edges_motif3": FeatureMeta(nth_edges_motif(3), {"m3"}),  # Any
            "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),  # Any
            "edges_motif4": FeatureMeta(nth_edges_motif(4), {"m4"}),  # Any
        }

        self.MOTIFS = {
            "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),
            "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"})
        }

    """
    Features by duration:
    Short:
        - Average neighbor degree
        - General
        - Louvain
        - Hierarchy energy
    Medium:
        - Fiedler vector
        - K core
        - Motif 3
        - Closeness centrality 
        - Attraction basin
        - Eccentricity
        - Load centrality
        - Page rank
    Long:
        - Betweenness centrality
        - BFS moments
        - Communicability betweenness centrality
        - Flow
        - Motif 4
    """
