from __future__ import annotations

import networkx as nx

from typing import Any
from graph.labeling_helpers import compute_labels
from interfaces import PerturbationChain, Perturbation, LabelingFunction, AppliedPerturbation, PerturbationHeuristic

class BasePerturbationChain(PerturbationChain):
    def __init__(self, perturbations: list[AppliedPerturbation]) -> None:
        super().__init__(perturbations)

    def __add__(self, other: PerturbationChain) -> BasePerturbationChain:
        if not isinstance(other, BasePerturbationChain):
            raise ValueError('Attempting to extend perturbation chain with perturbation chain of incompatible type.')

        return BasePerturbationChain(self.perturbations + other.perturbations)

    def test_potential_perturbation(
        self,
        perturbation: Perturbation,
        heuristic: PerturbationHeuristic[Any],
        labeling_functions: list[LabelingFunction],
        graph: nx.Graph[Any],
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        out = None
        result = False
        changed_nodes: dict[int, tuple[Any, Any]] | None = None

        new_heuristic = heuristic
        original_result = compute_labels(graph, labeling_functions)
        candidate_result = original_result

        while result is False:
            new_heuristic = new_heuristic.next(graph, self)

            if new_heuristic is None:
                break
            
            out = perturbation.apply_heuristic(
                graph, new_heuristic
            )

            # Check for label changes
            candidate_result = compute_labels(out.graph, labeling_functions)
            changed_nodes, changed_edges, changed_graphs = self.get_changes(original_result, candidate_result)

            if changed_nodes or changed_edges or changed_graphs:
                result = desired_result
            else:
                result = not desired_result

        if not result:
            out = None
        return out

    def test_perturbation(
        self,
        index: int,
        graph: nx.Graph[Any],
        labeling_functions: list[LabelingFunction],
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        result = False

        if index >= len(self):
            raise ValueError('Perturbation index out of bounds.')

        original_result = compute_labels(graph, labeling_functions)
        candidate_result = original_result

        out = self.perturbations[index].perturbation.apply_heuristic(
            graph, self.perturbations[index].heuristic
        )

        # Check for label changes
        candidate_result = compute_labels(out.graph, labeling_functions)
        changed_nodes, changed_edges, changed_graphs = self.get_changes(original_result, candidate_result)

        if changed_nodes or changed_edges or changed_graphs:
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
        subchain: BasePerturbationChain,
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
                    subchain + BasePerturbationChain([perturbation]),
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
                        BasePerturbationChain([perturbation]),
                        subchain_length
                    ):
                        return False
        return len(self) > 1

    def _check_chain_validity(self, graph: nx.Graph[Any], labeling_functions: list[LabelingFunction]) -> bool:
        result = None
        acc_graph = graph

        for i in range(len(self)):
            result = self.test_perturbation(i, acc_graph, labeling_functions, i >= (len(self) - 1))

            if result is None:
                break

            acc_graph = result.graph

        return (result is not None)

    def is_valid(self, graph: nx.Graph[Any], labeling_functions: list[LabelingFunction]) -> bool:
        return self._check_chain_validity(graph, labeling_functions) and self.check_subchain_validity(graph, labeling_functions)