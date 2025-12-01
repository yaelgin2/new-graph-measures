from .base_features_metadata import BaseFeaturesMeta
from .feature_meta import FeatureMeta


class AcceleratedFeaturesMetadata(BaseFeaturesMeta):
    """
    The following are the implemented features. This file includes the accelerated versions for the features which have
    the option, whereas the similar features_metadata.py file includes the regular versions.
    For each feature, the comment to the right describes to which graph they are intended.
    We split the features into 3 classes by duration, below.
    """

    def __init__(self, gpu=False, device=0):
        self.NODE_LEVEL = {
            "motif3": FeatureMeta(nth_nodes_motif(3, gpu, device), {"m3"}),  # Any
            "edges_motif3": FeatureMeta(nth_edges_motif(3), {"m3"}),  # Any
            "motif4": FeatureMeta(nth_nodes_motif(4, gpu, device), {"m4"}),  # Any
            "edges_motif4": FeatureMeta(nth_edges_motif(4), {"m3"}),  # Any
        }

        self.MOTIFS = {
            "motif3": FeatureMeta(nth_nodes_motif(3, gpu, device), {"m3"}),
            "motif4": FeatureMeta(nth_nodes_motif(4, gpu, device), {"m4"})
        }

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
