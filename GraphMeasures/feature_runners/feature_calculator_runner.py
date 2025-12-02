import os
import pickle

from multiprocessing import Process, Queue

from ..feature_calculators.feature_calculator import FeatureCalculator
from ..loggers import PrintLogger, EmptyLogger

import networkx as nx
import numpy as np
import pandas as pd


class Worker(Process):
    def __init__(self, queue, calculators, include, logger=None):
        super(Worker, self).__init__()
        if logger is None:
            logger = EmptyLogger()

        self._queue = queue
        self._calculators = calculators
        self._logger = logger
        self._include = include

    def run(self):
        self._logger.info('Worker started')
        # do some initialization here

        self._logger.info('Computing things!')
        for feature_name in iter(self._queue.get, None):
            self._calculators[feature_name].build(include=self._include)


# object that calculates & holds a list of features of a graph.
class FeatureCalculatorRunner(dict):
    def __init__(self, graph, features, dir_path, logger=None, is_max_connected=False):
        # features is a dict that mapping feature name to feature-build object
        self._base_dir = dir_path
        self._logger = EmptyLogger() if logger is None else logger
        self._matrix = None  # is used???
        self._is_build = False

        if is_max_connected:
            if graph.is_directed():
                subgraphs = [graph.subgraph(c) for c in nx.weakly_connected_components(graph)]
            else:
                subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            self._graph = max(subgraphs, key=len)
        else:
            self._graph = graph

        self._abbreviations = {abbr: name for name, meta in features.items() for abbr in meta.abbr_set}
        # building the feature calculators data structure
        super(FeatureCalculatorRunner, self).__init__({name: meta.calculator(self._graph, logger=logger)
                                                       for name, meta in features.items()})

    @property
    def graph(self):
        return self._graph

    @property
    def is_build(self):
        return self._is_build

    def _build_serially(self, include, force_build: bool = False, dump_path: str = None, dumping_specs: dict = None):
        if dump_path is not None and self._graph is not None:
            pickle.dump(self._graph, open(self._feature_path("graph", dump_path), "wb"))
        for name, feature in self.items():
            if force_build or not os.path.exists(self._feature_path(name)):
                try:
                    feature.build(include=include)
                except ImportError:
                    # Probably user chooses acc=True without using the correct version.
                    print(
                        '\033[91m' + "You chose the accelerate option, but the accelerate compiled files were not"
                                     " found. Check that the accelerate version was installed properly. " + '\033[0m'
                    )
                    exit(1)

                if dump_path is not None:
                    self._dump_feature(name, feature, dump_path, dumping_specs)
            else:
                self._load_feature(name)

    # a single process means it is calculated serially
    def build(self, num_processes: int = 1, include: set = None, should_dump: bool = False, dumping_specs: dict = None,
              force_build=False):  # , exclude: set=None):
        # if exclude is None:
        #     exclude = set()
        if include is None:
            include = set()

        if 1 == num_processes:
            dump_path = None
            if should_dump:
                dump_path = self._base_dir
                if not os.path.exists(dump_path):
                    os.makedirs(dump_path)
            self._build_serially(include, dump_path=dump_path, force_build=force_build, dumping_specs=dumping_specs)
            self._is_build = True
            return

        request_queue = Queue()
        workers = [Worker(request_queue, self, include, logger=self._logger) for _ in range(num_processes)]
        # Starting all workers
        for worker in workers:
            worker.start()

        # Feeding the queue with all the features
        for feature_name in self:
            request_queue.put(feature_name)

        # Sentinel objects to allow clean shutdown: 1 per worker.
        for _ in range(num_processes):
            request_queue.put(None)

        # Joining all workers
        for worker in workers:
            worker.join()

    def _load_feature(self, name):
        if self._graph is None:
            assert os.path.exists(self._feature_path("graph")), "Graph is not present in the given directory"
            self._graph = pickle.load(open(self._feature_path("graph"), "rb"))
        feature = pickle.load(open(self._feature_path(name), "rb"))
        feature.load_meta({name: getattr(self, name) for name in FeatureCalculator.META_VALUES})
        self[name] = feature
        return self[name]

    def __getattr__(self, name):
        if name not in self:
            if name in self._abbreviations:
                name = self._abbreviations[name]
            else:
                return super(FeatureCalculatorRunner, self).__getattribute__(name)

        # if obj is already calculated - return it
        obj = self[name]
        if obj.is_loaded:
            return obj

        # if obj is not calculated, check if it exist on the file system
        # if it doesn't - calculate it, if it does - load it and return it
        if not os.path.exists(self._feature_path(name)):
            obj.build()
            return obj

        return self._load_feature(name)

    @property
    def features(self):
        return set(self)

    def _feature_path(self, name, dir_path=None):
        if dir_path is None:
            dir_path = self._base_dir
        return os.path.join(dir_path, name + ".pkl")

    def _dump_feature(self, name, feature, dir_path, dumping_specs=None):
        if feature.is_loaded:
            if dumping_specs is not None:
                cl = True if dumping_specs['object'] in ['class', 'both'] else False  # Whether to save the class
                ftr = True if dumping_specs['object'] in ['feature', 'both'] else False  # Whether to save the ftr value
            else:
                cl = True
                ftr = False
            if cl:
                prev_meta = feature.clean_meta()  # in order not to save unnecessary data
                pickle.dump(feature, open(self._feature_path(name, dir_path), "wb"))
                feature.load_meta(prev_meta)
            if ftr:
                if dumping_specs['file_type'] == 'pkl':
                    pickle.dump(feature.features, open(self._feature_path(name + "_ftr", dir_path), "wb"))
                else:  # dumping_specs['file_type'] is 'csv'
                    ftr_as_dict = self._feature_to_dict(feature.features)
                    ftr_df = pd.DataFrame(ftr_as_dict).transpose()
                    try:
                        ftr_df = ftr_df.rename(index=dumping_specs['vertex_names'])
                        if dir_path is None:
                            dir_path = self._base_dir
                        saving_path = os.path.join(dir_path, name + "_ftr.csv")
                        ftr_df.to_csv(saving_path, header=False)
                    except KeyError:
                        if dir_path is None:
                            dir_path = self._base_dir
                        saving_path = os.path.join(dir_path, name + "_ftr.csv")
                        ftr_df.to_csv(saving_path, header=False, index=False)

    def dump(self, dir_path=None):
        if dir_path is None:
            dir_path = self._base_dir

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for name, feature in self.items():
            self._dump_feature(name, feature, dir_path)

    def edges_features_to_matrix(self, entries_order: list = None, add_ones=False, dtype=None, mtype=np.matrix,
                                 should_zscore: bool = True, get_features_order: bool = False):
        if entries_order is None:
            entries_order = sorted(self._graph)

        sorted_features = [feature[1] for feature in self.items() if feature[1].is_relevant() and feature[1].is_loaded]
        edge_features = [feat for feat in sorted_features if str(feat.name).startswith("edge")]

        # edges features
        if edge_features:
            x = []
            for feature in sorted_features:
                x.append(feature.to_matrix(entries_order, mtype=mtype, should_zscore=should_zscore))
            emx = np.hstack(x)
            if add_ones:
                emx = np.hstack([emx, np.ones((emx.shape[0], 1))])
            emx.astype(dtype)
        else:
            emx = np.empty((len(entries_order), 0))

        if get_features_order:
            return mtype(emx), [feature.name for feature in edge_features]

        return mtype(emx)

    # sparse.csr_matrix(matrix, dtype=np.float32)
    def to_matrix(self, entries_order: list = None, add_ones=False, dtype=None, mtype=np.matrix,
                  should_zscore: bool = True, get_features_order: bool = False):
        if entries_order is None:
            entries_order = sorted(self._graph)

        sorted_features = [feature[1] for feature in self.items() if feature[1].is_relevant() and feature[1].is_loaded]

        # edge_features = [feat for feat in sorted_features if str(feat).startswith("edge")]
        sorted_features = [feat for feat in sorted_features if not str(feat.name).startswith("edge")]

        # nodes features:
        if sorted_features:
            x = []
            for feature in sorted_features:
                x.append(feature.to_matrix(entries_order, mtype=mtype, should_zscore=should_zscore))
            mx = np.hstack(x)
            if add_ones:
                mx = np.hstack([mx, np.ones((mx.shape[0], 1))])
            mx.astype(dtype)
        else:
            mx = np.empty((len(entries_order), 0))

        if get_features_order:
            return mtype(mx), [feature.name for feature in sorted_features]

        return mtype(mx)

    def to_dict(self, dtype=None, should_zscore: bool = True):
        mx = self.to_matrix(dtype=dtype, mtype=np.matrix, should_zscore=should_zscore)
        return {node: mx[i, :] for i, node in enumerate(sorted(self._graph))}

    @staticmethod
    def _feature_to_dict(feat):
        # Creating a dictionary of features that can enter the pandas DataFrame.
        if type(feat) == dict:
            if type(next(iter(feat.values()))) in [list, np.ndarray]:
                return feat
            else:
                return {key: [value] for key, value in feat.items()}
        else:
            return {i: (feat[i] if type(feat[i]) in [list, np.ndarray] else [feat[i]]) for i in range(len(feat))}
