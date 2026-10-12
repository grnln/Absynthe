import abc
import networkx as nx

from typing import Any
from interfaces.labeling_result import LabelingResult

class LabelingFunction(abc.ABC):
    """Abstract base class for labeling a graph.

    Implementations decide which fields of LabelingResult to populate:
    node labels (labels), edge labels (edge_labels), graph-level labels
    (graph_labels), or any combination.

    Implementations should NOT mutate the input graph.
    """

    @abc.abstractmethod
    def label(self, graph: nx.Graph[Any]) -> LabelingResult:
        """Label the graph and return results.

        Args:
            graph: The graph to label.

        Returns:
            A LabelingResult with any subset of labels, edge_labels,
            graph_labels, details, and metadata populated.
        """
        pass
