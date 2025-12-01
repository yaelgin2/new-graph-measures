import re
from abc import ABC, abstractmethod
from typing import Union

import networkx as nx
import numpy as np
from scipy.stats import zscore

from .decorators import time_log
from ..loggers import EmptyLogger

class FeatureCalculator(ABC):
    META_VALUES = ["_gnx", "_logger"]

    def __init__(self, graph: Union[nx.Graph, nx.DiGraph[]], *args, logger=None, **kwargs):
        # super(FeatureCalculator, self).__init__()
        self._is_loaded = False
        self._features = {}
        self._logger = EmptyLogger() if logger is None else logger
        self._graph = graph
        self._print_name = self.print_name()
        self._default_val = 0

    is_loaded = property(lambda self: self._is_loaded, None, None, "Whether the features were calculated")

    @abstractmethod
    def is_relevant(self):
        """
        is method relevant
        """
        pass

    def clean_meta(self):
        meta = {}
        for name in type(self).META_VALUES:
            meta[name] = getattr(self, name)
            setattr(self, name, None)
        return meta

    def load_meta(self, meta):
        for name, val in meta.items():
            setattr(self, name, val)

    def _is_meta_loaded(self):
        return any(getattr(self, name) is not None for name in self.META_VALUES)

    @classmethod
    def print_name(cls):
        split_name = re.findall("[A-Z][^A-Z]*", cls.__name__)
        if "calculator" == split_name[-1].lower():
            split_name = split_name[:-1]
        return "_".join(map(lambda x: x.lower(), split_name))

    @time_log
    def build(self, include: set = None):
        # Don't calculate it!
        if not self.is_relevant():
            self._is_loaded = True
            return

        if include is None:
            include = set()
        self._calculate(include)
        self._is_loaded = True
        return self._features

    @abstractmethod
    def _calculate(self, include):
        """Compute features for the given set."""
        pass

    @abstractmethod
    def _get_feature(self, element):
        """Return the feature value for a single element."""
        pass

    @abstractmethod
    def _params_order(self, input_order: list = None):
        """Return the ordered list of elements to iterate over."""
        pass

    @property
    def features(self):
        return self._features

    @property
    def name(self):
        return self._print_name

    def to_matrix(self, params_order: list = None, mtype=np.matrix, should_zscore: bool = True):
        x = []
        for element in self._params_order(params_order):
            x.append(self._get_feature(element))
        mx = np.matrix(x).astype(np.float32)

        # infinity is possible due to the change of the matrix type (i.e. overflow from 64 bit to 32 bit)
        mx[np.isinf(mx)] = self._default_val
        if 1 == mx.shape[0]:
            mx = mx.transpose()
        if should_zscore:
            mx = zscore(mx, axis=0)
        return mtype(mx)

    def __repr__(self):
        status = "loaded" if self.is_loaded else "not loaded"
        if not self._is_meta_loaded():
            status = "no_meta"
        elif not self.is_relevant():
            status = "irrelevant"
        return "<Feature %s: %s>" % (self._print_name, status,)

