import datetime
import logging
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd

from .features_infra.graph_features import GraphFeatures
from .loggers import PrintLogger, FileLogger, multi_logger

not_exist_feature_for_directed_graph = ["louvain", "fiedler_vector", "communicability_betweenness_centrality",
                                        "generalized_degree"]
not_exist_feature_for_undirected_graph = ["attractor_basin", "flow"]


class FeatureCalculator:
    def __init__(self, graph, features, dir_path="", acc=True, directed=False, gpu=False, device=2, verbose=True,
                 params=None, should_zscore: bool = True):
        """
        A class used to calculate features for a given graph, input as a text-like file.

        :param graph: str|nx.Graph|nx.DiGraph
        Path to graph edges file (text-like file, e.g. txt or csv), from which the graph is built using networkx.
        The graph must be unweighted. If its vertices are not [0, 1, ..., n-1], they are mapped to become
        [0, 1, ..., n-1] and the mapping is saved.
        Every row in the edges file should include "source_id,distance_id", without a header row.
        :param features: list of strings
        List of the names of each feature. Could be any name from features_meta.py or "additional_features".
        :param dir_path: str
        Path to the directory in which the feature calculations will be (or already are) located.
        :param acc: bool
        Whether to run the accelerated features, assuming it is possible to do so.
        :param directed: bool
        Whether the built graph is directed.
        :param gpu: bool
        Whether to use GPUs, assuming it is possible to do so (i.e. the GPU exists and the CUDA matches).
        :param device: int
        If gpu is True, indicates on which GPU device to calculate. Will return error if the index doesn't match the
        available GPUs.
        :param verbose: bool
        Whether to print things indicating the phases of calculations.
        :param params: dict, or None
        For clique detection uses, this is a dictionary of the graph settings:
        size, directed, clique size, edge probability. The corresponding keys must have be:
        subgraph_size, directed, vertices, probability
        Ignored for any other use.
        """

        self._dir_path = dir_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self._features = features  # By their name as appears in accelerated_features_meta
        self._gpu = gpu
        self._device = device
        self._verbose = verbose
        self.should_zscore = should_zscore

        self._logger = multi_logger([PrintLogger("Logger", level=logging.DEBUG),
                                     FileLogger("FLogger", path=dir_path, level=logging.INFO)], name=None) \
            if verbose else None

        self._params = params
        self.unknown_features = []
        self.nodes_order = None
        self._adj_matrix = None
        self._raw_features = None
        self._other_features = None

        self._load_graph(graph, directed)
        self._get_feature_meta(features, acc)  # acc determines whether to use the accelerated features

    def _load_graph(self, graph, directed=False):
        if type(graph) is str:
            self._graph = nx.read_edgelist(graph, delimiter=',', create_using=nx.DiGraph() if directed else nx.Graph())
        elif type(graph) is nx.Graph or type(graph) is nx.DiGraph:
            if type(graph) is nx.Graph and directed:
                raise ValueError("Expect directed graph, but got undirected graph.")
                # self._graph = nx.to_directed(graph.copy())
            elif type(graph) is nx.DiGraph and not directed:
                raise ValueError("Expect undirected graph, but got directed graph.")
                # self._graph = nx.to_undirected(graph.copy())
            else:
                self._graph = graph.copy()
        else:
            raise ValueError("Graph must be path to edges list, nx.Graph or nx.DiGraph")

        # save the nodes before the convert to integers which mix their order
        self.nodes_order = list(self._graph.nodes)

        vertices = np.array(self._graph.nodes)
        should_be_vertices = np.arange(len(vertices))
        self._mapping = {i: v for i, v in enumerate(self._graph)}
        if not np.array_equal(vertices, should_be_vertices):
            if self._verbose:
                self._logger.debug("Relabeling vertices to [0, 1, ..., n-1]")
            pickle.dump(self._mapping, open(os.path.join(self._dir_path, "vertices_mapping.pkl"), "wb"))
            self._graph = nx.convert_node_labels_to_integers(self._graph)
        if self._verbose:
            self._logger.info(str(datetime.datetime.now()) + " , Loaded graph")
            self._logger.debug("Graph Size: %d Nodes, %d Edges" % (len(self._graph), len(self._graph.edges)))

    def is_valid_feature(self, feature, all_node_features):
        if self._graph.is_directed():
            return feature in all_node_features and feature not in not_exist_feature_for_directed_graph

        return feature in all_node_features and feature not in not_exist_feature_for_undirected_graph

    def _get_feature_meta(self, features, acc):
        if acc:
            from .features_meta.accelerated_features_meta import FeaturesMeta
            features_meta_kwargs = dict(gpu=self._gpu, device=self._device)
        else:
            from .features_meta.features_meta import FeaturesMeta
            features_meta_kwargs = dict()

        all_node_features = FeaturesMeta(**features_meta_kwargs).NODE_LEVEL
        valid_features = [feature for feature in features if self.is_valid_feature(feature, all_node_features)]
        self.unknown_features = [feature for feature in features
                                 if not self.is_valid_feature(feature, all_node_features)]

        self._features = {}
        self._special_features = []
        if self._verbose:
            for key in self.unknown_features:
                self._logger.debug("Feature %s is not available for this graph, ignoring this feature." % key)

        for key in valid_features:
            if key in ['degree', 'in_degree', 'out_degree', 'additional_features']:
                self._special_features.append(key)
            else:
                self._features[key] = all_node_features[key]

    def get_features(self):
        feature_matrix, names = self.feature_matrix_with_names()
        features_df = pd.DataFrame(feature_matrix)
        features_df.columns = names

        features_df.index = self.nodes_order
        features_df.sort_index(inplace=True)

        # Fill non exists features with nan
        for key in self.unknown_features:
            features_df[key] = np.nan

        return features_df

    def calculate_features(self, should_dump=True, dumping_specs=None, force_build=False):
        """
        :param should_dump: A boolean flag. If True the feature will be dumped and saved.
                            Otherwise, the features will not be saved.
        :param dumping_specs: A dictionary of specifications how to dump the non-special features.
                              The default is saving the class only (as a pickle file).
                              'object': What to save - either 'class' (save the calculator with the features inside),
                                        'feature' (the feature itself only, saved as name + '_ftr') or 'both'.
                                        Note that if only the feature is saved, when one calls the calculator again,
                                        the class will not load the feature and instead calculate it again.
                              'file_type': If the feature itself is saved, one can choose between two formats:
                                           either 'pkl' (save the feature as a pickle file, as is) or 'csv' (save a
                                           csv file of the feature values).
                              'vertex_names': If the features are saved as a csv file, there is an option of saving
                                              the name of each vertex in each row, before the feature values.
                                              The value here is a boolean indicating whether to put the original names
                                              the vertices in the beginning of each row.
        :force_build: If True the features will build even if there is a pickle file with former values of the feature.
        """
        if not len(self._features) + len(self._special_features) and self._verbose:
            print("No features were chosen!")
        else:
            self._adj_matrix = nx.adjacency_matrix(self._graph)
            self._raw_features = GraphFeatures(gnx=self._graph, features=self._features, dir_path=self._dir_path,
                                               logger=self._logger)
            if dumping_specs is not None:
                if 'vertex_names' in dumping_specs:
                    if dumping_specs['vertex_names']:
                        dumping_specs['vertex_names'] = self._mapping
                    else:
                        del dumping_specs['vertex_names']
            self._raw_features.build(should_dump=should_dump, dumping_specs=dumping_specs, force_build=force_build)
            self._other_features = OtherFeatures(self._graph, self._special_features, self._dir_path, self._params,
                                                 self._logger)
            self._other_features.build(should_dump=should_dump)
            if self._verbose:
                self._logger.info(str(datetime.datetime.now()) + " , Calculated features")

    @property
    def feature_matrix(self):
        return np.hstack((self._raw_features.to_matrix(mtype=np.array, should_zscore=self.should_zscore),
                          self._other_features.feature_matrix))

    def feature_matrix_with_names(self):
        # return the feature matrix and the order of the features in the matrix
        raw_features, raw_order = self._raw_features.to_matrix(mtype=np.array, should_zscore=self.should_zscore,
                                                               get_features_order=True)
        other_features = self._other_features.feature_matrix
        other_order = self._other_features.features
        return np.hstack((raw_features, other_features)), self.build_names_list(raw_order + other_order)

    @property
    def adjacency_matrix(self):
        return self._adj_matrix

    def build_names_list(self, names):
        """
        This function get a features list, and return a new list
        with each feature time it's output size.
        For example - feature which it's output is two columns will appear twice.
        """
        new_list = []
        features_output_size = {
            "average_neighbor_degree": 1,
            "betweenness_centrality": 1,
            "bfs_moments": 2,
            "closeness_centrality": 1,
            "eccentricity": 1,
            "fiedler_vector": 0 if self._graph.is_directed() else 1,
            "k_core": 1,
            "load_centrality": 1,
            "motif3": 13 if self._graph.is_directed() else 2,
            "edges_motif3": 13 if self._graph.is_directed() else 2,
            "motifs_node_3": 13 if self._graph.is_directed() else 2,
            "motif3_edges_gpu": 13 if self._graph.is_directed() else 2,
            "motif4": 199 if self._graph.is_directed() else 6,
            "edges_motif4": 199 if self._graph.is_directed() else 6,
            "motifs_node_4": 199 if self._graph.is_directed() else 6,
            "motifs4_edges_gpu": 199 if self._graph.is_directed() else 6,
            "degree": 1,
            "eigenvector_centrality": 1,
            "clustering_coefficient": 1,
            "square_clustering_coefficient": 1,
            "generalized_degree": 0 if self._graph.is_directed() else 16,
            "all_pairs_shortest_path_length": len(self._graph.nodes),
            "attractor_basin": 1 if self._graph.is_directed() else 0,
            "flow": 1 if self._graph.is_directed() else 0,
            "general": 2 if self._graph.is_directed() else 1,
            "page_rank": 1,
            "in_degree": 1,
            "out_degree": 1
        }

        for name in names:
            if name not in features_output_size:
                new_list.append(name)
            else:
                num = features_output_size[name]
                if num == 1:
                    new_list.append(name)
                if num > 1:
                    for i in range(num):
                        new_list.append(f'{name}_{i + 1}')

        return new_list


class OtherFeatures:
    def __init__(self, graph, features, dir_path, params=None, logger=None):
        self._graph = graph
        self._features = features
        self._dir_path = dir_path
        self._logger = logger
        self._params = params
        self._feat_string_to_function = {
            'degree': self._calc_degree,
            'in_degree': self._calc_in_degree,
            'out_degree': self._calc_out_degree,
            'additional_features': self._calc_additional_features
        }

        self._feature_matrix = None

    def _calc_degree(self):
        degrees = list(self._graph.degree)
        # return {n: d for n, d in degrees}
        return np.array([d[1] for d in degrees]).reshape(-1, 1)

    def _calc_in_degree(self):
        if not self._graph.is_directed():
            arr = np.empty(len(self._graph.nodes))
            arr[:] = np.nan
            return arr.reshape(-1, 1)

        degrees = list(self._graph.in_degree)
        return np.array([d[1] for d in degrees]).reshape(-1, 1)

    def _calc_out_degree(self):
        if not self._graph.is_directed():
            arr = np.empty(len(self._graph.nodes))
            arr[:] = np.nan
            return arr.reshape(-1, 1)

        degrees = list(self._graph.out_degree)
        return np.array([d[1] for d in degrees]).reshape(-1, 1)

    def _calc_additional_features(self):
        from .additional_features import AdditionalFeatures
        if self._params is None:
            raise ValueError("params is None")
        if not os.path.exists(os.path.join(self._dir_path, "motif3.pkl")):
            raise FileNotFoundError("Motif 3 must be calculated")
        if not os.path.exists(os.path.join(self._dir_path, "motif4.pkl")):
            raise FileNotFoundError("Motif 4 must be calculated")

        motif_matrix = np.hstack((pickle.load(open(os.path.join(self._dir_path, "motif3.pkl"), "rb")),
                                  pickle.load(open(os.path.join(self._dir_path, "motif4.pkl"), "rb"))))
        add_ftrs = AdditionalFeatures(self._params, self._graph, self._dir_path, motif_matrix)
        return add_ftrs.calculate_extra_ftrs()

    def build(self, should_dump=False):
        self._feature_matrix = np.empty((len(self._graph), 0))
        for feat_str in self._features:
            if self._logger:
                start_time = datetime.datetime.now()
                self._logger.info("Start %s" % feat_str)
            if os.path.exists(
                    os.path.join(self._dir_path, feat_str + '.pkl')) and feat_str != "additional_features":
                feat = pickle.load(open(os.path.join(self._dir_path, feat_str + ".pkl"), "rb"))
                self._feature_matrix = np.hstack((self._feature_matrix, feat))
                if self._logger:
                    cur_time = datetime.datetime.now()
                    self._logger.info("Finish %s at %s" % (feat_str, cur_time - start_time))
            else:
                feat = self._feat_string_to_function[feat_str]()
                if should_dump:
                    pickle.dump(feat, open(os.path.join(self._dir_path, feat_str + ".pkl"), "wb"))
                self._feature_matrix = np.hstack((self._feature_matrix, feat))
                if self._logger:
                    cur_time = datetime.datetime.now()
                    self._logger.info("Finish %s at %s" % (feat_str, cur_time - start_time))

    @property
    def feature_matrix(self):
        return self._feature_matrix

    @property
    def features(self):
        return self._features
