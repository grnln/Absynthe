import networkx as nx

from typing import Any
from .perturbation import Perturbation
from .perturbation_walk import PerturbationWalk

class AppliedPerturbation:
    def __init__(
        self,
        perturbation: Perturbation,
        walk: PerturbationWalk[Any],
        graph: nx.Graph[Any],
        changes: dict[str, Any]
    ) -> None:
        self.perturbation = perturbation
        self.walk = walk
        self.graph = graph
        self.changes = changes

    def __str__(self) -> str:
        return self.perturbation.__class__.__name__ + str(self.changes)

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AppliedPerturbation):
            return False

        return type(self.perturbation) == type(other.perturbation) and self.changes == other.changes