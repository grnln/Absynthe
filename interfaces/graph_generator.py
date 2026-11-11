import abc

from .graph_types import GraphLike


class GraphGenerator(abc.ABC):
    """Abstract base class for generating graphs.

    A graph generator produces a NetworkX graph. Concrete implementations
    may generate graphs from motifs, read them from files, use random
    graph models, or any other method.

    Any NetworkX graph class is accepted downstream: plain undirected
    ``nx.Graph`` instances (e.g. motif composites) as well as directed,
    multi-relational knowledge graphs (``nx.DiGraph`` / ``nx.MultiDiGraph``
    with ``type`` node attributes and ``relation`` edge attributes, see
    ``utils.kg_utils``).
    """

    @abc.abstractmethod
    def generate_graph(self, **kwargs) -> GraphLike:
        """Generate a graph.

        Args:
            **kwargs: Keyword arguments specific to the graph generation.

        Returns:
            The generated graph (any NetworkX graph class).
        """
        pass
