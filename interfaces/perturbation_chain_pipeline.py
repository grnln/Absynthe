import abc
import networkx as nx

from abc import abstractmethod
from typing import Any
from graph.labeling_helpers import compute_labels

from interfaces.perturbation import Perturbation
from interfaces.labeling_function import LabelingFunction

from .perturbation_chain import AppliedPerturbation, PerturbationChain

class PerturbationChainPipeline(abc.ABC):
    def __init__(
        self,
        graph: nx.Graph[Any],
        perturbations: list[Perturbation],
        labeling_functions: list[LabelingFunction],
        max_iterations: int,
        max_length: int
    ) -> None:
        self.graph = graph;
        self.max_length = max_length
        self.max_iterations = max_iterations
        self.perturbations = perturbations
        self.labeling_functions = labeling_functions

    def find(self) -> list[PerturbationChain]:
        out: list[PerturbationChain] = []

        for length in range(2, (self.max_length + 1)):
            out += self._find_chains(length)

        return out

    @abstractmethod
    def _test_subchain(self, chain: PerturbationChain, subchain: PerturbationChain, current_length: int, subchain_length: int) -> bool:
        pass

    @abstractmethod
    def _find_chain(self, chain: PerturbationChain, current_length: int, chain_length: int, chains: list[PerturbationChain]) -> None:
        pass

    def _test_perturbation(
        self,
        perturbation: Perturbation,
        graph: nx.Graph[Any],
        desired_result: bool = False
    ) -> tuple[bool, nx.Graph[Any], dict[str, Any]]:
        attempts = 0
        result = False

        original_result = compute_labels(graph, self.labeling_functions)

        candidate_graph = graph
        candidate_result = original_result

        changes: dict[str, Any] = dict()
        changed_nodes: dict[int, tuple[Any, Any]] | None = None

        while result is False and attempts < self.max_iterations:
            attempts += 1

            # Always apply to the original graph
            candidate_graph, changes = perturbation.apply(
                graph, hint = original_result.hint
            )

            # Check for label changes
            candidate_result = compute_labels(candidate_graph, self.labeling_functions)
            candidate_labels = candidate_result.node_labels
            changed_nodes = {
                node: (original_result.node_labels.get(node), candidate_labels[node])
                for node in candidate_labels if candidate_labels[node] != original_result.node_labels.get(node)
            }

            if changed_nodes:
                result = desired_result
            else:
                result = not desired_result

        return (result, candidate_graph, changes)

    def _find_chains(self, chain_length: int) -> list[PerturbationChain]:
        chains: list[PerturbationChain] = []
    
        for perturbation in self.perturbations:
            result, candidate_graph, changes = self._test_perturbation(perturbation, self.graph)

            if result is False:
                continue

            self._find_chain(
                PerturbationChain([AppliedPerturbation(perturbation, candidate_graph, changes)]),
                1,
                chain_length,
                chains
            )
        return chains