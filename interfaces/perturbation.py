import abc
from typing import Tuple, Dict, Any, Optional
import networkx as nx

from .perturbation_hint import PerturbationHint
from .perturbation_walk import PerturbationWalk

class Perturbation(abc.ABC):
    """Abstract base class for graph perturbations.

    A perturbation is any operation that modifies a graph in a controlled way.
    Examples include removing nodes, adding edges, removing edges, modifying
    node features, etc.

    Each concrete perturbation configures its own parameters (strategy, count,
    probability, etc.) in __init__. The apply method is stateless with respect
    to the graph argument.
    """

    @abc.abstractmethod
    def apply(self, graph: nx.Graph[Any],
              hint: Optional[PerturbationHint] = None
              ) -> Tuple[nx.Graph[Any], Dict[str, Any]]:
        """Apply this perturbation to a graph.

        The implementation must NOT mutate the input graph. It should work
        on a copy.

        Args:
            graph: The input graph to perturb.
            hint: Optional ``PerturbationHint`` produced by a labeling
                function, signaling which nodes/edges form the zone where
                the perturbation should focus. Concrete implementations
                decide how to interpret it; when ``None`` or empty they
                must operate over the whole graph.

        Returns:
            A tuple of (perturbed_graph, changes_dict) where changes_dict
            describes what changed (e.g., removed_nodes, added_edges, etc.).
            The exact keys depend on the perturbation type.
        """
        pass

    @abc.abstractmethod
    def apply_by_walk(self, graph: nx.Graph[Any], walk: PerturbationWalk[Any]) -> Any:
        pass