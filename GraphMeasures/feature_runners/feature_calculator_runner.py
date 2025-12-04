#pylint: disable=too-many-arguments,too-many-positional-arguments
"""
Feature calculation runner module.

Provides Worker and FeatureCalculatorRunner classes for parallel or serial feature computation
and storage of graph features.
"""

import os
import pickle

from multiprocessing import Process, Queue

import networkx as nx
import numpy as np
import pandas as pd

from ..feature_calculators.feature_calculator import FeatureCalculator
from ..loggers import EmptyLogger


class Worker(Process):
    """Process worker for computing features from a queue."""

    def __init__(self, queue, calculators, include, logger=None):
        """Initialize worker with queue, calculators, include set, and optional logger."""
        super().__init__()
        self._logger = logger or EmptyLogger()
        self._queue = queue
        self._calculators = calculators
        self._include = include

    def run(self):
        """Run the worker, consuming feature names from queue and building them."""
        self._logger.info('Worker started')
        self._logger.info('Computing things!')
        for feature_name in iter(self._queue.get, None):
            self._calculators[feature_name].build(include=self._include)


class FeatureCalculatorRunner(dict):
    """Runner and container for graph feature calculators."""

    def __init__(self, graph, features, dir_path, logger=None, is_max_connected=False):
        """Initialize runner with a graph, feature definitions, storage path, and logger."""
        self._base_dir = dir_path
        self._logger = logger or EmptyLogger()
        self._matrix = None
        self._is_build = False

        if is_max_connected:
            subgraphs = [graph.subgraph(c) for c in
                         (nx.weakly_connected_components(graph)
                          if graph.is_directed() else nx.connected_components(graph))]
            self._graph = max(subgraphs, key=len)
        else:
            self._graph = graph

        self._abbreviations = {abbr: name for name, meta
                               in features.items() for abbr in meta.abbr_set}
        super().__init__({name: meta.calculator(self._graph, logger=logger)
                          for name, meta in features.items()})

    @property
    def graph(self):
        """Return the graph associated with this runner."""
        return self._graph

    @property
    def is_build(self):
        """Return whether features have been built."""
        return self._is_build

    def _build_serially(self, include, force_build: bool = False,
                        dump_path: str = None, dumping_specs: dict = None):
        """Build features one by one in the current process."""
        if dump_path and self._graph is not None:
            pickle.dump(self._graph, open(self._feature_path("graph", dump_path), "wb"))
        for name, feature in self.items():
            if force_build or not os.path.exists(self._feature_path(name)):
                try:
                    feature.build(include=include)
                except ImportError:
                    print('\033[91mAccelerate option not properly installed.\033[0m')
                if dump_path:
                    self._dump_feature(name, feature, dump_path, dumping_specs)
            else:
                self._load_feature(name)

    def build(self, num_processes: int = 1, include: set = None, should_dump: bool = False,
              dumping_specs: dict = None, force_build=False):
        """Build features using either a single process or multiple worker processes."""
        include = include or set()
        if num_processes == 1:
            dump_path = self._base_dir if should_dump else None
            if dump_path and not os.path.exists(dump_path):
                os.makedirs(dump_path)
            self._build_serially(include, dump_path=dump_path,
                                 force_build=force_build, dumping_specs=dumping_specs)
            self._is_build = True
            return

        request_queue = Queue()
        workers = [Worker(request_queue, self, include, logger=self._logger)
                   for _ in range(num_processes)]
        for worker in workers:
            worker.start()
        for feature_name in self:
            request_queue.put(feature_name)
        for _ in range(num_processes):
            request_queue.put(None)
        for worker in workers:
            worker.join()

    def _load_feature(self, name):
        """Load a feature from disk or build it if not present."""
        if self._graph is None:
            assert os.path.exists(self._feature_path("graph")), "Graph is not present"
            self._graph = pickle.load(open(self._feature_path("graph"), "rb"))
        feature = pickle.load(open(self._feature_path(name), "rb"))
        feature.load_meta({name: getattr(self, name) for name in FeatureCalculator.META_VALUES})
        self[name] = feature
        return feature

    def __getattr__(self, name):
        """Access a feature by name or abbreviation, building or loading if needed."""
        if name not in self:
            name = self._abbreviations.get(name, name)
        obj = self[name]
        if obj.is_loaded:
            return obj
        if not os.path.exists(self._feature_path(name)):
            obj.build()
            return obj
        return self._load_feature(name)

    @property
    def features(self):
        """Return the set of feature names."""
        return set(self)

    def _feature_path(self, name, dir_path=None):
        """Return the file path for a feature pickle."""
        dir_path = dir_path or self._base_dir
        return os.path.join(dir_path, name + ".pkl")

    def _dump_feature(self, name, feature, dir_path, dumping_specs=None):
        """Save a feature to disk according to dumping specifications."""
        if feature.is_loaded:
            cl = (dumping_specs.get('object', 'both')
                  in ['class', 'both']) if dumping_specs else True
            ftr = (dumping_specs.get('object', 'both')
                   in ['feature', 'both']) if dumping_specs else False
            if cl:
                prev_meta = feature.clean_meta()
                pickle.dump(feature, open(self._feature_path(name, dir_path), "wb"))
                feature.load_meta(prev_meta)
            if ftr:
                if dumping_specs['file_type'] == 'pkl':
                    pickle.dump(feature.features,
                                open(self._feature_path(name + "_ftr", dir_path), "wb"))
                else:
                    ftr_df = pd.DataFrame(self._feature_to_dict(feature.features)).transpose()
                    try:
                        ftr_df = ftr_df.rename(index=dumping_specs['vertex_names'])
                    except KeyError:
                        pass
                    ftr_df.to_csv(self._feature_path(name + "_ftr", dir_path),
                                  header=False, index=False)

    def dump(self, dir_path=None):
        """Dump all features to disk."""
        dir_path = dir_path or self._base_dir
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for name, feature in self.items():
            self._dump_feature(name, feature, dir_path)

    def edges_features_to_matrix(self, entries_order: list = None,
                                 add_ones=False, dtype=None, mtype=np.matrix,
                                 should_zscore: bool = True, get_features_order: bool = False):
        """Return a matrix of edge features, optionally z-scored or with a ones column."""
        entries_order = entries_order or sorted(self._graph)
        edge_features = [f for f in self.values()
                         if f.is_relevant() and f.is_loaded and str(f.name).startswith("edge")]
        if edge_features:
            x = [f.to_matrix(entries_order, mtype=mtype,
                             should_zscore=should_zscore) for f in edge_features]
            emx = np.hstack(x)
            if add_ones:
                emx = np.hstack([emx, np.ones((emx.shape[0], 1))])
            emx.astype(dtype)
        else:
            emx = np.empty((len(entries_order), 0))
        if get_features_order:
            return mtype(emx), [f.name for f in edge_features]
        return mtype(emx)

    def to_matrix(self, entries_order: list = None, add_ones=False, dtype=None, mtype=np.matrix,
                  should_zscore: bool = True, get_features_order: bool = False):
        """Return a matrix of node features, optionally z-scored or with a ones column."""
        entries_order = entries_order or sorted(self._graph)
        sorted_features = [f for f in self.values()
                           if f.is_relevant() and f.is_loaded
                           and not str(f.name).startswith("edge")]
        if sorted_features:
            x = [f.to_matrix(entries_order, mtype=mtype,
                             should_zscore=should_zscore) for f in sorted_features]
            mx = np.hstack(x)
            if add_ones:
                mx = np.hstack([mx, np.ones((mx.shape[0], 1))])
            mx.astype(dtype)
        else:
            mx = np.empty((len(entries_order), 0))
        if get_features_order:
            return mtype(mx), [f.name for f in sorted_features]
        return mtype(mx)

    def to_dict(self, dtype=None, should_zscore: bool = True):
        """Return node features as a dictionary keyed by node ID."""
        mx = self.to_matrix(dtype=dtype, mtype=np.matrix, should_zscore=should_zscore)
        return {node: mx[i, :] for i, node in enumerate(sorted(self._graph))}

    @staticmethod
    def _feature_to_dict(feat):
        """Convert a feature array or dict to a dictionary suitable for pandas DataFrame."""
        if isinstance(feat, dict):
            if type(next(iter(feat.values()))) in [list, np.ndarray]:
                return feat
            return {key: [value] for key, value in feat.items()}
        return {i: feat[i] if isinstance(feat[i], (list, np.ndarray))
        else [feat[i]] for i in range(len(feat))}
