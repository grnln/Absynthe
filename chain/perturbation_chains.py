from __future__ import annotations

import networkx as nx

from typing import Any
from graph.labeling_helpers import compute_labels
from interfaces import GraphLike, PerturbationChain, Perturbation, LabelingFunction, LabelingResult, AppliedPerturbation, PerturbationHeuristic

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
        graph: GraphLike,
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        out = None
        result = False
        changed_nodes: dict[int, tuple[Any, Any]] | None = None

        original_result = compute_labels(graph, labeling_functions)
        candidate_result = original_result

        out = perturbation.apply_heuristic(
            graph, heuristic
        )

        # Check for label changes
        candidate_result = compute_labels(out.graph, labeling_functions)
        changed_nodes, changed_edges, changed_graphs = self.get_changes(original_result, candidate_result)
        
        out.set_labeling_changes(changed_nodes, changed_edges, changed_graphs)
        
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
        graph: GraphLike,
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

        out.set_labeling_changes(changed_nodes, changed_edges, changed_graphs)

        if changed_nodes or changed_edges or changed_graphs:
            result = desired_result
        else:
            result = not desired_result

        if not result:
            out = None
        return out

    def _test_subchain(
        self,
        start: int,
        original_graph: GraphLike,
        labeling_functions: list[LabelingFunction],
        subchain: BasePerturbationChain,
        subchain_length: int
    ) -> bool:
        out = True

        if len(subchain) >= subchain_length:
            result = None
            acc_graph = original_graph

            for perturbation in subchain:
                try:
                    result = self.test_perturbation(self.perturbations.index(perturbation), acc_graph, labeling_functions)
                except nx.NetworkXError:
                    result = None
                    break

                if result is not None:
                    acc_graph = result.graph
                else:
                    break
    
            out = result is not None
        else:
            for i in range(start, len(self.perturbations)):
                out = self._test_subchain(
                    i + 1,
                    original_graph,
                    labeling_functions,
                    subchain + BasePerturbationChain([self.perturbations[i]]),
                    subchain_length
                )

                if out == False:
                    break
        return out

    def check_subchain_validity(self, original_graph: GraphLike, labeling_functions: list[LabelingFunction]) -> bool:
        if len(self) > 2:
            for subchain_length in range(2, len(self)):
                if not self._test_subchain(
                    0,
                    original_graph,
                    labeling_functions,
                    BasePerturbationChain([]),
                    subchain_length
                ):
                    return False
        return len(self) > 1

    def _check_chain_validity(self, graph: GraphLike, labeling_functions: list[LabelingFunction]) -> bool:
        result = None
        acc_graph = graph

        for i in range(len(self)):
            result = self.test_perturbation(i, acc_graph, labeling_functions, i >= (len(self) - 1))

            if result is None:
                break

            acc_graph = result.graph

        return (result is not None)

    def is_valid(self, graph: GraphLike, labeling_functions: list[LabelingFunction]) -> bool:
        return self._check_chain_validity(graph, labeling_functions) and self.check_subchain_validity(graph, labeling_functions)

    def chain_type(self) -> str:
        return 'base'

class NewChangedElementsPerturbationChain(PerturbationChain):
    def __init__(self, perturbations: list[AppliedPerturbation]) -> None:
        super().__init__(perturbations)

    def __add__(self, other: PerturbationChain) -> NewChangedElementsPerturbationChain:
        if not isinstance(other, NewChangedElementsPerturbationChain):
            raise ValueError('Attempting to extend perturbation chain with perturbation chain of incompatible type.')

        return NewChangedElementsPerturbationChain(self.perturbations + other.perturbations)

    def test_potential_perturbation(
        self,
        perturbation: Perturbation,
        heuristic: PerturbationHeuristic[Any],
        labeling_functions: list[LabelingFunction],
        graph: GraphLike,
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        out = perturbation.apply_heuristic(
            graph, heuristic
        )
        original_result = compute_labels(graph, labeling_functions)
        candidate_result = compute_labels(out.graph, labeling_functions)

        changed_nodes, changed_edges, changed_graphs = self.get_changes(original_result, candidate_result)
        out.set_labeling_changes(changed_nodes, changed_edges, changed_graphs)
        
        return out

    def test_perturbation(
        self,
        index: int,
        graph: GraphLike,
        labeling_functions: list[LabelingFunction],
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        if index >= len(self):
            raise ValueError('Perturbation index out of bounds.')

        out = self.perturbations[index].perturbation.apply_heuristic(
            graph, self.perturbations[index].heuristic
        )
        original_result = compute_labels(graph, labeling_functions)
        candidate_result = compute_labels(out.graph, labeling_functions)

        changed_nodes, changed_edges, changed_graphs = self.get_changes(original_result, candidate_result)
        out.set_labeling_changes(changed_nodes, changed_edges, changed_graphs)

        return out

    def check_subchain_validity(self, original_graph: GraphLike, labeling_functions: list[LabelingFunction]) -> bool:
        return True

    def _get_individual_changes(
        self,
        graph: GraphLike,
        labeling_functions: list[LabelingFunction],
        original_result: LabelingResult
    ) -> tuple[dict[Any, tuple[Any, Any]], dict[tuple[Any, Any], tuple[Any, Any]], dict[Any, tuple[Any, Any]]]:
        node_changes: dict[Any, tuple[Any, Any]] = dict()
        edge_changes: dict[tuple[Any, Any], tuple[Any, Any]] = dict()
        graph_changes: dict[Any, tuple[Any, Any]] = dict()
        
        for perturbation in self:
            applied = perturbation.perturbation.apply_heuristic(graph, perturbation.heuristic)
            result = compute_labels(applied.graph, labeling_functions)
            changed_nodes, changed_edges, changed_graphs = self.get_changes(original_result, result)

            node_changes.update(changed_nodes)
            edge_changes.update(changed_edges)
            graph_changes.update(changed_graphs)
        return (node_changes, edge_changes, graph_changes)

    def is_valid(self, graph: GraphLike, labeling_functions: list[LabelingFunction]) -> bool:
        if len(self) <= 0:
            return False

        original_result = compute_labels(graph, labeling_functions)
        final_result = compute_labels(self.perturbations[-1].graph, labeling_functions)
        
        all_node_changes, all_edge_changes, all_graph_changes = self.get_changes(original_result, final_result)
        individual_node_changes, individual_edge_changes, individual_graph_changes = self._get_individual_changes(graph, labeling_functions, original_result)

        new_changed_nodes = set(all_node_changes.keys()) - set(individual_node_changes.keys())
        new_changed_edges = set(all_edge_changes.keys()) - set(individual_edge_changes.keys())
        new_changed_graphs = set(all_graph_changes.keys()) - set(individual_graph_changes.keys())

        return (len(new_changed_nodes) > 0 or len(new_changed_edges) > 0 or len(new_changed_graphs) > 0)

    def chain_type(self) -> str:
        return 'new_changed_elements'

chain_types: dict[str, type[PerturbationChain]] = {
    'base': BasePerturbationChain,
    'new_changed_elements': NewChangedElementsPerturbationChain
}