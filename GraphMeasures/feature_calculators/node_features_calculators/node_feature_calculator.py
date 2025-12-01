
class NodeFeatureCalculator(FeatureCalculator):
    # def __init__(self, *args, **kwargs):
    #     super(NodeFeatureCalculator, self).__init__(*args, **kwargs)
    #     self._features = {str(node): None for node in self._gnx}

    def _params_order(self, input_order: list = None):
        if input_order is None:
            return sorted(self._gnx)
        return input_order

    def _get_feature(self, element) -> np.ndarray:
        return np.array(self.features[element])

    def edge_based_node_feature(self):
        nodes_dict = self.features
        edge_dict = {}
        for edge in self._gnx.edges():
            n1_val = np.array(nodes_dict[edge[0]])
            n2_val = np.array(nodes_dict[edge[1]])

            edge_dict[edge] = list(chain(*zip(n1_val - n2_val, np.mean([n1_val, n2_val], axis=0))))
        return edge_dict

    def feature(self, element):
        return self._get_feature(element)
