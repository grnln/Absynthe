from __future__ import annotations

import abc
import networkx as nx

from typing import Any
from abc import abstractmethod
from .perturbation import Perturbation
from .labeling_function import LabelingFunction
from .applied_perturbation import AppliedPerturbation
from .perturbation_heuristic import PerturbationHeuristic

class PerturbationChain(abc.ABC):
    def __init__(self, perturbations: list[AppliedPerturbation]) -> None:
        self.perturbations = perturbations

    def __getitem__(self, index: int) -> AppliedPerturbation:
        return self.perturbations[index]

    @abstractmethod
    def __add__(self, other: PerturbationChain) -> PerturbationChain:
        pass

    def __len__(self) -> int:
        return len(self.perturbations)

    def __str__(self) -> str:
        return f'Perturbation chain of length {len(self)} [\n{",\n".join([f"\t{perturbation}" for perturbation in self.perturbations])}\n]'

    def __repr__(self) -> str:
        return str(self)

    def match_perturbation_set(self, other: PerturbationChain) -> bool:
        if len(other) != len(self):
            return False

        num_equal_elements = 0

        for perturbation in self:
            for other_perturbation in other:
                if perturbation == other_perturbation:
                    num_equal_elements += 1

        return num_equal_elements == len(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PerturbationChain):
            return False
        return self.perturbations == other.perturbations

    def __hash__(self) -> int:
        return hash(tuple(self.perturbations))

    def append(self, perturbation: AppliedPerturbation) -> None:
        self.perturbations.append(perturbation)

    def remove(self, perturbation: AppliedPerturbation) -> None:
        self.perturbations.remove(perturbation)

    def clear(self) -> None:
        self.perturbations.clear()

    @abstractmethod
    def test_potential_perturbation(
        self,
        perturbation: Perturbation,
        heuristic: PerturbationHeuristic[Any],
        labeling_functions: list[LabelingFunction],
        graph: nx.Graph[Any],
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        pass

    @abstractmethod
    def test_perturbation(
        self,
        index: int,
        graph: nx.Graph[Any],
        labeling_functions: list[LabelingFunction],
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        pass

    @abstractmethod
    def check_subchain_validity(self, original_graph: nx.Graph[Any], labeling_functions: list[LabelingFunction]) -> bool:
        pass

    @abstractmethod
    def check_chain_validity(self, graph: nx.Graph[Any], labeling_functions: list[LabelingFunction]) -> bool:
        pass

    @abstractmethod
    def is_valid(self, graph: nx.Graph[Any], labeling_functions: list[LabelingFunction]) -> bool:
        pass