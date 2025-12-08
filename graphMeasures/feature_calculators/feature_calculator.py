"""
Feature calculator module.

Provides the abstract base class `FeatureCalculator` for computing
graph features. Supports node-level, edge-level, or motif-based
features with optional logging and metadata management.

Subclasses must implement the computation logic and element ordering
by overriding abstract methods. Provides utility functions to convert
features into matrices and handle missing or infinite values.
"""
import re
from abc import ABC, abstractmethod
from typing import Union, Dict

import numpy as np
from scipy.stats import zscore

from .decorators import time_log
from .utils import z_scoring
from ..loggers import EmptyLogger
from graphMeasures.graph_measure_types import GraphType


class FeatureCalculator(ABC):
    """
    Abstract base class for graph feature calculators.

    Attributes:
        META_VALUES (list[str]): Names of attributes considered metadata.
        _is_loaded (bool): Whether the features have been computed.
        _features (dict): Stores computed features keyed by element.
        _logger (Logger): Logger instance for debug/info output.
        _graph (Union[nx.Graph, nx.DiGraph]): The graph on which features are computed.
        _get_name (str): Human-readable name of the feature calculator.
        _default_val (float): Default value to replace infinities in feature matrices.

    Properties:
        is_loaded (bool): Indicates whether features have been calculated.
        features (dict): Accessor for the computed features.
        name (str): Returns the human-readable calculator name.

    Methods:
        is_relevant(): Abstract. Determine whether the feature is relevant for this graph.
        clean_meta(): Temporarily removes metadata attributes and returns their values.
        load_meta(meta): Restores metadata attributes from a dictionary.
        build(include: set = None): Compute features for the graph.
        _calculate(include): Abstract. Compute features for the given elements.
        _get_feature(element): Abstract. Return the feature value for a single element.
        _params_order(input_order: list = None): Abstract. Return ordered list of elements.
        to_matrix(params_order: list = None, mtype=np.matrix, should_zscore: bool = True):
            Convert features to a NumPy matrix with optional z-scoring.
    """
    META_VALUES = ["_graph", "_logger"]

    #pylint: disable=unused-argument
    def __init__(self, graph: GraphType, configuration: Dict[str, str],
                 *args, logger=None, **kwargs):
        """
        Initialize the feature calculator.

        Args:
            graph (Union[nx.Graph, nx.DiGraph]): The graph to compute features on.
            *args: Additional positional arguments (reserved for subclasses).
            logger: Optional logger instance. Defaults to EmptyLogger.
            **kwargs: Additional keyword arguments (reserved for subclasses).

        Initializes metadata, feature storage, logger, and human-readable name.
        """
        self._is_loaded = False
        self._features = {}
        self._logger = EmptyLogger() if logger is None else logger
        self._graph = graph
        self._get_name = self.get_name()
        self._default_val = 0
        self._configuration = configuration

    @property
    def is_loaded(self) -> bool:
        """Whether the features were calculated."""
        return self._is_loaded

    @abstractmethod
    def is_relevant(self) -> bool:
        """
        Determine whether the feature is relevant for the given graph.

        Returns:
            bool: True if the feature should be calculated, False otherwise.
        """


    def clean_meta(self) -> dict:
        """
        Temporarily remove metadata attributes and return their values.

        Returns:
            dict: Dictionary mapping metadata attribute names to their values.
        """
        meta = {}
        for name in type(self).META_VALUES:
            meta[name] = getattr(self, name)
            setattr(self, name, None)
        return meta

    def load_meta(self, meta: dict) -> None:
        """
        Restore metadata attributes from a dictionary.

        Args:
            meta (dict): Dictionary mapping metadata attribute names to values.
        """
        for name, val in meta.items():
            setattr(self, name, val)

    def _is_meta_loaded(self) -> bool:
        """
        Check if any metadata attributes are currently loaded.

        Returns:
            bool: True if at least one metadata attribute is not None.
        """
        return any(getattr(self, name) is not None for name in self.META_VALUES)

    @classmethod
    def get_name(cls) -> str:
        """
        Generate a human-readable name for the calculator from the class name.

        Splits CamelCase class names and removes the word 'Calculator' if present.

        Returns:
            str: Lowercase, underscore-separated name of the calculator.
        """
        split_name = re.findall("[A-Z][^A-Z]*", cls.__name__)
        if "calculator" == split_name[-1].lower():
            split_name = split_name[:-1]
        return "_".join(map(lambda x: x.lower(), split_name))

    @time_log
    def build(self, include: set = None) -> dict:
        """
        Compute features for the graph.

        Args:
            include (set, optional): Subset of elements to compute. If None, compute all.

        Returns:
            dict: Computed features keyed by element.
        """
        if not self.is_relevant():
            self._is_loaded = True
            return {}

        if include is None:
            include = set()
        self._calculate(include)
        self._is_loaded = True
        return self._features

    @abstractmethod
    def _calculate(self, include: set) -> None:
        """
        Compute features for the given elements.

        Args:
            include (set): Subset of elements (nodes, edges, motifs, etc.) to compute features for.
        """


    @abstractmethod
    def _get_feature(self, element) -> Union[np.ndarray, float, int]:
        """
        Return the computed feature value for a single element.

        Args:
            element: The element (node, edge, or motif) to retrieve features for.

        Returns:
            np.ndarray or numeric: Feature values for the element.
        """

    @abstractmethod
    def _params_order(self, input_order: list = None) -> list:
        """
        Determine the order of elements for computation.

        Args:
            input_order (list, optional): Explicit order of elements. If None, use default order.

        Returns:
            list: Ordered list of elements to process.
        """

    @property
    def features(self) -> dict:
        """Access the computed features dictionary."""
        return self._features

    @property
    def name(self) -> str:
        """Return the human-readable calculator name."""
        return self._get_name

    def to_matrix(
            self,
            params_order: list = None,
            mtype=np.matrix,
            should_zscore: bool = True
    ) -> np.matrix:
        """
        Convert computed features to a matrix.

        Args:
            params_order (list, optional): Order of elements for the matrix rows.
            mtype (callable, optional): Matrix type to use (default: np.matrix).
            should_zscore (bool, optional): Whether to z-score features along columns.

        Returns:
            np.matrix: Matrix of feature values, optionally z-scored and cast to float32.
        """
        x = [self._get_feature(element) for element in self._params_order(params_order)]
        mx = np.matrix(x).astype(np.float32)
        mx[np.isinf(mx)] = self._default_val
        if mx.shape[0] == 1:
            mx = mx.transpose()
        if should_zscore:
            mx = z_scoring(mx)
        return mtype(mx)

    def __repr__(self) -> str:
        """
        Return a string representation of the feature calculator.

        Shows the name of the calculator and its current status:
        'loaded', 'not loaded', 'no_meta', or 'irrelevant'.

        Returns:
            str: String representation of the calculator.
        """
        status = "loaded" if self.is_loaded else "not loaded"
        if not self._is_meta_loaded():
            status = "no_meta"
        elif not self.is_relevant():
            status = "irrelevant"
        return f"<Feature {self._get_name}: {status}>"
