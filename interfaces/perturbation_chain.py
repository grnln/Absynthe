from __future__ import annotations

import networkx as nx

from typing import Any
from interfaces import LabelingFunction
from graph.labeling_helpers import compute_labels
from .applied_perturbation import AppliedPerturbation

class PerturbationChain:
    def __init__(self, perturbations: list[AppliedPerturbation]) -> None:
        self.perturbations = perturbations

    def __getitem__(self, index: int) -> AppliedPerturbation:
        return self.perturbations[index]

    def __len__(self) -> int:
        return len(self.perturbations)

    def __str__(self) -> str:
        return f'Perturbation chain of length {len(self)} [\n{",\n".join([f"\t{perturbation}" for perturbation in self.perturbations])}\n]'

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, other: PerturbationChain) -> PerturbationChain:
        return PerturbationChain(self.perturbations + other.perturbations)

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

    def test_perturbation(
        self,
        index: int,
        graph: nx.Graph[Any],
        labeling_functions: list[LabelingFunction],
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        result = False
        changed_nodes: dict[int, tuple[Any, Any]] | None = None

        if index >= len(self):
            raise ValueError('Perturbation index out of bounds.')

        original_result = compute_labels(graph, labeling_functions)
        candidate_result = original_result

        out = self.perturbations[index].perturbation.apply_heuristic(
            graph, self.perturbations[index].heuristic
        )

        # Check for label changes
        candidate_result = compute_labels(out.graph, labeling_functions)
        candidate_labels = candidate_result.node_labels
        changed_nodes = {
            node: (original_result.node_labels.get(node), candidate_labels[node])
            for node in candidate_labels if candidate_labels[node] != original_result.node_labels.get(node)
        }

        if changed_nodes:
            result = desired_result
        else:
            result = not desired_result

        if not result:
            out = None
        return out

    def _test_subchain(
            self,
            original_graph: nx.Graph[Any],
            labeling_functions: list[LabelingFunction],
            subchain: PerturbationChain,
            subchain_length: int
        ) -> bool:
            out = True
    
            if len(subchain) >= subchain_length:
                result = None
                acc_graph = original_graph
    
                for perturbation in subchain:
                    result = self.test_perturbation(self.perturbations.index(perturbation), acc_graph, labeling_functions)
    
                    if result is not None:
                        acc_graph = result.graph
        
                out = result is not None
            else:
                for perturbation in [p for p in self if p not in subchain]:
                    out = self._test_subchain(
                        original_graph,
                        labeling_functions,
                        subchain + PerturbationChain([perturbation]),
                        subchain_length
                    )
                    
                    if out is False:
                        break;
            return out

    def check_subchain_validity(self, original_graph: nx.Graph[Any], labeling_functions: list[LabelingFunction]) -> bool:
        if len(self) > 2:
            for subchain_length in range(2, len(self)):
                for perturbation in self:
                    if not self._test_subchain(
                        original_graph,
                        labeling_functions,
                        PerturbationChain([perturbation]),
                        subchain_length
                    ):
                        return False
        return len(self) > 1

    def check_chain_validity(self, graph: nx.Graph[Any], labeling_functions: list[LabelingFunction]) -> bool:
        result = None
        acc_graph = graph

        for i in range(len(self)):
            result = self.test_perturbation(i, acc_graph, labeling_functions, i >= (len(self) - 1))

            if result is None:
                break

            acc_graph = result.graph

        return (result is not None)

    def is_valid(self, graph: nx.Graph[Any], labeling_functions: list[LabelingFunction]) -> bool:
        return self.check_chain_validity(graph, labeling_functions) and self.check_subchain_validity(graph, labeling_functions)