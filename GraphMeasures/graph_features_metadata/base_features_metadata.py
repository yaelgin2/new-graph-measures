"""
Base features metadata module.

Defines the abstract base class for graph feature metadata containers.
Provides the common interface for storing node-level and motif-based
features using FeatureMeta objects. This base class is intended to
be extended by concrete implementations, such as regular or
accelerated feature sets.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Dict

from .feature_meta import FeatureMeta

@dataclass
class BaseFeaturesMeta(ABC):
    """
    Abstract base class for graph feature metadata.

    Attributes:
        NODE_LEVEL (Dict[str, FeatureMeta]): Dictionary mapping names of
            node-level features to their corresponding FeatureMeta objects.
        MOTIFS (Dict[str, FeatureMeta]): Dictionary mapping names of motif-
            based features to their corresponding FeatureMeta objects.

    Notes:
        This class does not implement any computation itself. Concrete
        subclasses should provide initialized FeatureMeta objects for
        the attributes.
    """
    node_level: Dict[str, FeatureMeta]
    motifs: Dict[str, FeatureMeta]
