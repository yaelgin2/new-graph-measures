from abc import ABC, abstractmethod

class FeatureMetaAdapterBase(ABC):
    """
    Base class for feature calculator adapters.

    Adapters wrap calculators that require extra arguments beyond the graph,
    providing a consistent interface for FeatureRunner/FeatureManager.

    Subclasses must implement the `__call__` method, which takes a graph
    and returns the calculated features (e.g., as a matrix or array).
    """

    @abstractmethod
    def __call__(self, graph, configuration, logger):
        """
        Run the wrapped calculator on the given graph.

        Args:
            graph: The graph to calculate features on.

        Returns:
            Calculated features (e.g., numpy array, pandas DataFrame, dict).
        """
        pass