# --------------------------
# Base interface
# --------------------------
from abc import ABC
from typing import Dict

from GraphMeasures.graph_features_metadata.feature_meta import FeatureMeta


class BaseFeaturesMeta(ABC):
    NODE_LEVEL: Dict[str, FeatureMeta]
    MOTIFS: Dict[str, FeatureMeta]