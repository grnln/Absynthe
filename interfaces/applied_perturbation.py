from typing import Any
from .graph_types import GraphLike
from .perturbation import Perturbation
from .perturbation_heuristic import PerturbationHeuristic

class AppliedPerturbation:
    def __init__(
        self,
        perturbation: Perturbation,
        heuristic: PerturbationHeuristic[Any],
        graph: GraphLike,
        original_graph: GraphLike,
        changes: dict[str, Any]
    ) -> None:
        self.perturbation = perturbation
        self.heuristic = heuristic
        self.graph = graph
        self.original_graph = original_graph
        self.changes = changes

        self.changed_nodes = {}
        self.changed_edges = {}
        self.changed_graphs = {}

    def __str__(self) -> str:
        return f'{self.perturbation.__class__.__name__}[{str(self.heuristic.current())}]'

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AppliedPerturbation):
            return False

        return (
            type(self.perturbation) == type(other.perturbation)
            and self.changes == other.changes
        )

    def __hash__(self) -> int:
        return hash((self.perturbation, self.heuristic.current()))

    def set_labeling_changes(
        self,
        changed_nodes: dict[Any, tuple[Any, Any]],
        changed_edges: dict[tuple[Any, Any], tuple[tuple[Any, Any], tuple[Any, Any]]],
        changed_graphs: dict[Any, tuple[Any, Any]]
    ) -> None:
        self.changed_nodes = changed_nodes
        self.changed_edges = changed_edges
        self.changed_graphs = changed_graphs