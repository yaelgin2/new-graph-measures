from .src import attraction_basin
from ...feature_calculators.node_features_calculators import NodeFeatureCalculator
from ...graph_features_metadata.feature_meta import FeatureMeta


class AttractorBasinCalculator(NodeFeatureCalculator):
    def __init__(self, *args, alpha=2, **kwargs):
        super(AttractorBasinCalculator, self).__init__(*args, **kwargs)
        self._alpha = alpha
        self._default_val = float('nan')

    def is_relevant(self):
        return self._gnx.is_directed()

    def _calculate(self, include: set):
        self._features = attraction_basin(self._gnx, alpha=self._alpha)


feature_entry = {
    "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),
}
