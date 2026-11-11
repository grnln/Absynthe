from dataclasses import dataclass, field
from typing import Any, Optional, Set, Tuple

from .graph_types import ordered_pair


@dataclass
class PerturbationHint:
    """Hint produced by a labeling function about which zone of a graph
    is relevant for perturbation.

    A perturbation can use this hint to focus its action on the nodes and
    edges the labeler considers meaningful (e.g., the nodes/edges that
    belong to a detected motif), instead of touching arbitrary parts of
    the graph.

    Attributes:
        nodes: Set of node IDs in the relevant zone.
        edges: Set of edge references in the relevant zone. Each reference
            is ``(u, v)`` for simple graphs or ``(u, v, key)`` for
            multigraphs, as produced by :meth:`normalize_edge`. On
            undirected graphs the endpoints are stored in canonical
            ``(min, max)`` order; on directed graphs the order is kept.
    """

    nodes: Set[Any] = field(default_factory=set)
    edges: Set[Tuple] = field(default_factory=set)

    def is_empty(self) -> bool:
        return not self.nodes and not self.edges

    def merge(self, other: "PerturbationHint") -> "PerturbationHint":
        return PerturbationHint(
            nodes=self.nodes | other.nodes,
            edges=self.edges | other.edges,
        )

    @staticmethod
    def normalize_edge(u: Any, v: Any, key: Optional[Any] = None,
                       directed: bool = False) -> Tuple:
        """Build the canonical hint reference for an edge.

        ``normalize_edge(u, v)`` keeps the historical behaviour (undirected,
        endpoints sorted). Pass ``directed=True`` to keep the endpoint order
        and ``key=`` to reference one edge of a multigraph.
        """
        pair = (u, v) if directed else ordered_pair(u, v)
        return pair if key is None else pair + (key,)

    def contains_edge(self, u: Any, v: Any, key: Optional[Any] = None,
                      directed: bool = False) -> bool:
        """Return True if the edge ``(u, v[, key])`` is referenced by the hint.

        A keyless query matches keyed references with the same endpoints
        (any parallel edge of the pair counts as "in the zone").
        """
        pairs = [(u, v)] if directed else [(u, v), (v, u)]
        for pair in pairs:
            if pair in self.edges:
                return True
            if key is not None and pair + (key,) in self.edges:
                return True
        if key is None:
            for ref in self.edges:
                if len(ref) == 3 and (ref[0], ref[1]) in pairs:
                    return True
        return False
