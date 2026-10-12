from __future__ import annotations

import networkx as nx

from typing import Any
from perturbation import Perturbation

class AppliedPerturbation:
    def __init__(
        self,
        perturbation: Perturbation,
        graph: nx.Graph[Any],
        changes: dict[str, Any]
    ) -> None:
        self.perturbation = perturbation
        self.graph = graph
        self.changes = changes

class PerturbationChain:
    def __init__(self, perturbations: list[AppliedPerturbation]) -> None:
        self.perturbations = perturbations

    def __getitem__(self, index: int) -> AppliedPerturbation:
        if index >= len(self.perturbations):
            raise ValueError("Index out of range.")
        
        return self.perturbations[index]

    def __len__(self) -> int:
        return len(self.perturbations)

    def __str__(self) -> str:
        return str(self.perturbations)

    def __add__(self, other: PerturbationChain) -> PerturbationChain:
        return PerturbationChain(self.perturbations + other.perturbations)

    def append(self, perturbation: AppliedPerturbation) -> None:
        self.perturbations.append(perturbation)

    def remove(self, perturbation: AppliedPerturbation) -> None:
        self.perturbations.remove(perturbation)

    def clear(self) -> None:
        self.perturbations.clear()