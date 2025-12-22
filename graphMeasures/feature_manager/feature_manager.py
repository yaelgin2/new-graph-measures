# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes, line-too-long
"""
Package entry point. Managing building graph and running features on it.
"""
import datetime
import json
import logging
import os
import pickle
from typing import Union, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .constants import NOT_EXIST_FEATURE_FOR_DIRECTED_GRAPH, \
    NOT_EXIST_FEATURE_FOR_UNDIRECTED_GRAPH
from ..exceptions.exception_codes import FAILED_TO_CREATE_OUTPUT_FOLDER_EXCEPTION, \
    OUTPUT_FOLDER_IS_EMPTY_EXCEPTION, GRAPH_FILE_DOES_NOT_EXIST_EXCEPTION, CONFIGURATION_FORMAT_EXCEPTION, \
    CONFIGURATION_FILE_DOES_NOT_EXIST_EXCEPTION, GRAPH_NOT_LOADED_EXCEPTION, COLOR_FILE_DOES_NOT_EXIST_EXCEPTION, \
    COLORS_FORMAT_IS_INVALID_EXCEPTION
from ..feature_calculators.feature_calculator import FeatureCalculator
from ..feature_runners.additional_features_runner import AdditionalFeatureRunner
from ..graph_features_metadata import AcceleratedFeaturesMetadata, FeaturesMetadata
from ..feature_runners.feature_calculator_runner import FeatureCalculatorRunner
from ..exceptions.graph_measures_exception import GraphMeasuresException
from ..loggers import PrintLogger, FileLogger, MultiLogger

class FeatureManager:
    """
        A class used to calculate features for a given graph.
    """

    def __init__(self, graph, features,
                 configuration: Union[str, Dict[str, str]],
                 colors: Optional[Union[Dict[int, int], str]] = None,
                 dir_path="", acc=True, directed=False,
                 gpu=False, device=2, verbose=True,
                 params=None, should_zscore: bool = True):
        """
        A class used to calculate features for a given graph, input as a text-like file.

        :param graph: str|nx.Graph|nx.DiGraph
        Path to graph edges file (text-like file, e.g. txt or csv),
         from which the graph is built using networkx.
        The graph must be unweighted. If its vertices are not [0, 1, ..., n-1],
         they are mapped to become
        [0, 1, ..., n-1] and the mapping is saved.
        Every row in the edges file should include "source_id,distance_id",
         without a header row.
        :param features: list of strings
        List of the names of each feature.
        Could be any name from features_meta.py or "additional_features".
        :param dir_path: str
        Path to the directory in which the feature calculations will be (or already are) located.
        :param acc: bool
        Whether to run the accelerated features, assuming it is possible to do so.
        :param directed: bool
        Whether the built graph is directed.
        :param gpu: bool
        Whether to use GPUs, assuming it is possible to do so
        i.e. the GPU exists and the CUDA matches).
        :param device: int
        If gpu is True, indicates on which GPU device to calculate.
         Will return error if the index doesn't match the
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

        self._features = features  # By their name as appears in accelerated_features_meta
        self._gpu = gpu
        self._device = device
        self._verbose = verbose
        self.should_zscore = should_zscore

        self._set_loggers()

        self._params = params
        self.unknown_features = []
        self.nodes_order = None
        self._adj_matrix = None
        self._raw_features = None
        self._other_features = None

        self._create_output_folder()
        self._colors_loaded = False
        self._load_graph(graph, colors, directed)
        self._load_configuration(configuration)
        self._get_feature_meta(features, acc)  # acc determines whether to use the accelerated features

    def _load_configuration(self, configuration: Union[str, Dict[str, str]]):
        """
        Load configuration file.
        :param configuration: configuration dictionary or file path.
        :return: None
        """
        if isinstance(configuration, str):
            if not os.path.exists(configuration):
                raise GraphMeasuresException("Configuration file does not exist.",
                                             CONFIGURATION_FILE_DOES_NOT_EXIST_EXCEPTION)
            with open(configuration, "rb") as configuration_file:
                configuration = json.load(configuration_file)
        if not isinstance(configuration, dict):
            raise GraphMeasuresException("Configuration format is invalid.", CONFIGURATION_FORMAT_EXCEPTION)
        self._configuration = configuration

    def _set_loggers(self):
        failed_logger_messages = []
        logger_list = []
        try:
            logger_list.append(PrintLogger("Logger", level=logging.DEBUG))
        except ValueError:
            failed_logger_messages.append("Filed to create a print logger.")

        try:
            logger_list.append(FileLogger("FLogger", path=self._dir_path, level=logging.INFO))
        except ValueError:
            failed_logger_messages.append("Failed to create a file logger.")

        self._logger = MultiLogger("MLogger", logger_list) \
            if self._verbose and len(logger_list) > 0 else None
        for error_msg in failed_logger_messages:
            self._logger.warning(error_msg)

    def _create_output_folder(self):
        if self._dir_path == "":
            self._logger.error("Output folder cannot be empty.")
            raise GraphMeasuresException("Output folder cannot be empty.",
                                         OUTPUT_FOLDER_IS_EMPTY_EXCEPTION)
        if not os.path.exists(self._dir_path):
            try:
                os.makedirs(self._dir_path)
            except OSError as exception:
                self._logger.error(exception)
                raise GraphMeasuresException("Failed to create output folder.",
                                             FAILED_TO_CREATE_OUTPUT_FOLDER_EXCEPTION) from exception

    def _load_colors(self, colors) -> None:
        """
        Load colors, either from dictionary or from file path.
        :param colors:
        :return:
        """
        if colors is None:
            return
        if self._graph is None:
            raise GraphMeasuresException("Graph is not loaded when loading colors.", GRAPH_NOT_LOADED_EXCEPTION)
        graph_colors = colors
        if isinstance(colors, str):
            if not os.path.exists(colors):
                raise GraphMeasuresException("Colors file does not exist.", COLOR_FILE_DOES_NOT_EXIST_EXCEPTION)
            with open(colors, "rb") as color_file:
                try:
                    graph_colors = json.load(color_file)
                except json.JSONDecodeError:
                    raise GraphMeasuresException("Colors format is invalid.", COLORS_FORMAT_IS_INVALID_EXCEPTION)
        if isinstance(graph_colors, dict):
            for key, value in graph_colors.items():
                if not isinstance(key, str) or not isinstance(value, int):
                    raise GraphMeasuresException("Colors keys must be integers and keys must be strings.", COLORS_FORMAT_IS_INVALID_EXCEPTION)
        else:
            raise GraphMeasuresException("Colors must be a dictionary.", COLORS_FORMAT_IS_INVALID_EXCEPTION)
        # update to mapping
        new_graph_colors = {}
        if self._mapping:
            for new_index, old_index in self._mapping.items():
                new_graph_colors[new_index] = graph_colors[old_index]
        else:
            new_graph_colors = graph_colors
        nx.set_node_attributes(self._graph, new_graph_colors, FeatureCalculator.COLOR_ATTRIBUTE_KEY)
        self._colors_loaded = True


    def _load_graph(self, graph: Union[nx.DiGraph, nx.Graph, str],
                    colors: Optional[Union[Dict[int, int], str]] = None,
                    directed: bool=False):
        """
        Load the graph from file or nx object.
        :param graph: Graph from user.
        :param colors: colors for graph vertices.
        :param directed: is graph directed.
        :return: None
        """
        if isinstance(graph, str):
            if os.path.isfile(graph):
                self._graph = (
                    nx.read_edgelist(graph, delimiter=',', create_using=nx.DiGraph()
                    if directed else nx.Graph()))
            else:
                raise GraphMeasuresException("Graph file does not exist.",
                                             GRAPH_FILE_DOES_NOT_EXIST_EXCEPTION)
        elif isinstance(graph, (nx.Graph, nx.DiGraph)):
            if isinstance(graph, nx.Graph) and directed:
                raise ValueError("Expect directed graph, but got undirected graph.")
            if isinstance(graph, nx.DiGraph) and not directed:
                raise ValueError("Expect undirected graph, but got directed graph.")
            self._graph = graph.copy()
        else:
            raise ValueError("Graph must be path to edges list, nx.Graph or nx.DiGraph")

        # save the nodes before the convert to integers which mix their order
        self.nodes_order = list(self._graph.nodes)

        vertices = np.array(sorted([int(i) for i in self._graph.nodes]))
        should_be_vertices = np.array(range(len(vertices)))
        self._mapping = dict(enumerate(self._graph))
        if not np.array_equal(vertices, should_be_vertices):
            if self._verbose:
                self._logger.debug("Relabeling vertices to [0, 1, ..., n-1]")

                with open(os.path.join(self._dir_path, "vertices_mapping.pkl"), "wb") as vertices_mapping_file:
                    pickle.dump(self._mapping, vertices_mapping_file)

        self._graph = nx.convert_node_labels_to_integers(self._graph)
        self._load_colors(colors)
        if self._verbose:
            self._logger.info(str(datetime.datetime.now()) + " , Loaded graph")
            self._logger.debug(f"Graph Size: {len(self._graph)} Nodes, {len(self._graph.edges)} Edges")

    def is_valid_feature(self, feature, all_node_features):
        """
        Is feature a valid feature?
        """
        if self._graph.is_directed():
            return feature in all_node_features and feature not in NOT_EXIST_FEATURE_FOR_DIRECTED_GRAPH

        return feature in all_node_features and feature not in NOT_EXIST_FEATURE_FOR_UNDIRECTED_GRAPH

    def _get_feature_meta(self, features, acc):
        """
        Load features meta data.
        :param features: The features to get meta for/
        :param acc: Are features accelerated.
        :return: None
        """
        if acc:
            features_meta = AcceleratedFeaturesMetadata
            features_meta_kwargs = {"gpu": self._gpu, "device": self._device}
        else:
            features_meta = FeaturesMetadata
            features_meta_kwargs = {}

        all_node_features = features_meta(**features_meta_kwargs).node_level
        valid_features = [feature for feature in features if self.is_valid_feature(feature,
                                                                                   all_node_features)]
        self.unknown_features = [feature for feature in features
                                 if not self.is_valid_feature(feature, all_node_features)]

        self._features = {}
        self._special_features = []
        if self._verbose:
            for key in self.unknown_features:
                self._logger.debug(f"Feature {key} is not available for this graph, ignoring this feature.")

        for key in valid_features:
            if key in ['degree', 'in_degree', 'out_degree', 'additional_features']:
                self._special_features.append(key)
            else:
                self._features[key] = all_node_features[key]

    def get_features(self):
        """
        return features.
        """
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
            self._raw_features = FeatureCalculatorRunner(graph=self._graph, colores_loaded=self._colors_loaded,
                                                         configuration=self._configuration,
                                                         features=self._features, dir_path=self._dir_path,
                                                         logger=self._logger)
            if dumping_specs is not None:
                if 'vertex_names' in dumping_specs:
                    if dumping_specs['vertex_names']:
                        dumping_specs['vertex_names'] = self._mapping
                    else:
                        del dumping_specs['vertex_names']
            self._raw_features.build(should_dump=should_dump, dumping_specs=dumping_specs, force_build=force_build)
            self._other_features = AdditionalFeatureRunner(self._graph, self._special_features, self._dir_path, self._params,
                                                           self._logger)
            self._other_features.build(should_dump=should_dump)
            if self._verbose:
                self._logger.info(str(datetime.datetime.now()) + " , Calculated features")

    @property
    def feature_matrix(self):
        """
        :return: A pandas DataFrame containing the feature values.
        """
        return np.hstack((self._raw_features.to_matrix(mtype=np.array, should_zscore=self.should_zscore),
                          self._other_features.feature_matrix))

    def feature_matrix_with_names(self):
        """
        Feature matrix with names.
        :return: feature matrix with names.
        """
        # return the feature matrix and the order of the features in the matrix
        raw_features, raw_names = self._raw_features.to_matrix(mtype=np.array, should_zscore=self.should_zscore,
                                                               get_features_order=True)
        other_features = self._other_features.feature_matrix
        other_order = self._other_features.features
        return np.hstack((raw_features, other_features)), raw_names + self.build_names_list(other_order)

    @property
    def adjacency_matrix(self):
        """
        Returns the adjacency matrix of the graph.
        :return: adjacency matrix
        """
        return self._adj_matrix

    def build_names_list(self, names):
        """
        This function get a features list, and return a new list
        with each feature time it's output size.
        For example - feature which it's output is two columns will appear twice.
        """
        features_output_size = self._get_features_output_size()
        new_list = []

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

    def _get_features_output_size(self):
        """
        Return a mapping from feature name to its output size for the current graph.

        This resolver combines static sizes (e.g., features that always produce one column)
        with dynamic sizes that depend on the graph's properties, such as whether it is
        directed or the number of nodes.

        Returns:
            Dict[str, int]: Feature name to output size mapping, resolved for the current graph.

        Notes:
            This is not a static constant. It depends on `self._graph` at call time.
        """
        return {
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
