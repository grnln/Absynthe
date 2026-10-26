import abc
import networkx as nx

from typing import Any

class MotifGenerator(abc.ABC):
    """Abstract base class for generating graph motifs.

    A motif is a subgraph with a specific structure. All motifs must be
    returned as a NetworkX graph with node roles.
    """

    @abc.abstractmethod
    def generate_motif(self, start: int, id: int = 0, **kwargs) -> nx.Graph[Any]:
        """Generate a motif.

        Args:
            start (int): Starting index for the nodes in the motif.
            id (int): Identifier for the motif.
            **kwargs: Keyword arguments specific to the motif type.

        Returns:
            nx.Graph: The generated motif as NetworkX graph.
        """
        pass
