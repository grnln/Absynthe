import networkx as nx

from typing import Any
from .perturbation import Perturbation
from .perturbation_heuristic import PerturbationHeuristic

class AppliedPerturbation:
    def __init__(
        self,
        perturbation: Perturbation,
        heuristic: PerturbationHeuristic[Any],
        graph: nx.Graph[Any],
        changes: dict[str, Any]
    ) -> None:
        self.perturbation = perturbation
        self.heuristic = heuristic
        self.graph = graph
        self.changes = changes

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