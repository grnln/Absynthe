import abc
import networkx as nx

from typing import Any
from abc import abstractmethod
from graph.labeling_helpers import compute_labels

from interfaces.perturbation import Perturbation
from interfaces.perturbation_heuristic import PerturbationHeuristic
from interfaces.labeling_function import LabelingFunction

from .perturbation_chain import AppliedPerturbation, PerturbationChain

class PerturbationChainPipeline(abc.ABC):
    def __init__(
        self,
        graph: nx.Graph[Any],
        perturbations: list[tuple[Perturbation, PerturbationHeuristic[Any]]],
        labeling_functions: list[LabelingFunction],
        max_length: int,
        max_iterations: int | None = None,
        num_chains: int | None = None
    ) -> None:
        self.graph = graph;
        self.max_length = max_length
        self.max_iterations = max_iterations
        self.num_chains = num_chains
        self.perturbations = [perturbation for perturbation, _ in perturbations]
        self.labeling_functions = labeling_functions
        self.heuristics = {perturbation: heuristic for perturbation, heuristic in perturbations}

    @abstractmethod
    def _find_chains(self, chain_length: int) -> set[PerturbationChain]:
        pass

    def find(self) -> set[PerturbationChain]:
        out: set[PerturbationChain] = set()

        for length in range(2, (self.max_length + 1)):
            out = out.union(self._find_chains(length))
            
        return out

    def _test_perturbation(
        self,
        perturbation: Perturbation,
        heuristic: PerturbationHeuristic[Any],
        chain: PerturbationChain,
        graph: nx.Graph[Any],
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        out = None
        result = False
        changed_nodes: dict[int, tuple[Any, Any]] | None = None

        new_heuristic = heuristic
        original_result = compute_labels(graph, self.labeling_functions)
        candidate_result = original_result

        while result is False:
            new_heuristic = new_heuristic.next(graph, chain)

            if new_heuristic is None:
                break
            
            out = perturbation.apply_heuristic(
                graph, new_heuristic
            )

            # Check for label changes
            candidate_result = compute_labels(out.graph, self.labeling_functions)
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