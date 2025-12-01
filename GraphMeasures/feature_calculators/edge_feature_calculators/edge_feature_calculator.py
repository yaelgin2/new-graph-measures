class EdgeFeatureCalculator(FeatureCalculator):
    # def __init__(self, *args, **kwargs):
    #     super(EdgeFeatureCalculator, self).__init__(*args, **kwargs)
    #     self._features = {str(edge): None for edge in self._gnx.edges()}

    def _params_order(self, input_order: list = None):
        if input_order is None:
            return sorted(self._gnx.edges())
        return input_order

    def _get_feature(self, element) -> np.ndarray:
        return np.array(self.features[element])
