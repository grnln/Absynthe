import abc
import networkx as nx

from typing import Any
from abc import abstractmethod
from graph.labeling_helpers import compute_labels

from interfaces.perturbation import Perturbation
from interfaces.perturbation_walk import PerturbationWalk
from interfaces.labeling_function import LabelingFunction

from .perturbation_chain import AppliedPerturbation, PerturbationChain

class PerturbationChainPipeline(abc.ABC):
    def __init__(
        self,
        graph: nx.Graph[Any],
        perturbations: list[tuple[Perturbation, PerturbationWalk[Any]]],
        labeling_functions: list[LabelingFunction],
        max_length: int,
        max_iterations: int | None,
        num_chains: int | None
    ) -> None:
        self.graph = graph;
        self.max_length = max_length
        self.max_iterations = max_iterations
        self.num_chains = num_chains
        self.perturbations = [perturbation for perturbation, _ in perturbations]
        self.labeling_functions = labeling_functions
        self.walks = {perturbation: walk for perturbation, walk in perturbations}

    def find(self) -> list[PerturbationChain]:
        out: list[PerturbationChain] = []

        for length in range(2, (self.max_length + 1)):
            out += self._find_chains(length)
        return out

    @abstractmethod
    def _test_subchain(self, chain: PerturbationChain, subchain: PerturbationChain, subchain_length: int) -> bool:
        pass

    @abstractmethod
    def _find_chain(self, chain: PerturbationChain, chain_length: int, chains: list[PerturbationChain], num_chains: int | None) -> bool:
        pass

    def _test_perturbation(
        self,
        perturbation: Perturbation,
        walk: PerturbationWalk[Any],
        chain: PerturbationChain,
        graph: nx.Graph[Any],
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        out = None
        result = False
        changed_nodes: dict[int, tuple[Any, Any]] | None = None

        new_walk = walk
        original_result = compute_labels(graph, self.labeling_functions)
        candidate_result = original_result

        while result is False:
            new_walk = new_walk.next(graph, chain)

            if new_walk is None:
                break
            
            out = perturbation.apply_by_walk(
                graph, new_walk, chain
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

    def _test_applied_perturbation(
        self,
        perturbation: AppliedPerturbation,
        chain: PerturbationChain,
        graph: nx.Graph[Any],
        desired_result: bool = False
    ) -> AppliedPerturbation | None:
        result = False
        changed_nodes: dict[int, tuple[Any, Any]] | None = None

        original_result = compute_labels(graph, self.labeling_functions)
        candidate_result = original_result

        out = perturbation.perturbation.apply_by_walk(
            graph, perturbation.walk, chain
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

    def _find_chains(self, chain_length: int) -> list[PerturbationChain]:
        chains: list[PerturbationChain] = []
    
        for perturbation in self.perturbations:
            num_iterations = 0
            walk: PerturbationWalk[Any] | None = self.walks[perturbation]

            while walk is not None:
                result = self._test_perturbation(perturbation, walk, PerturbationChain([]), self.graph)

                if result is None:
                    walk = walk.next(self.graph, PerturbationChain([]))
                    continue

                walk = result.walk
                self.walks[perturbation] = result.walk

                while self._find_chain(
                    PerturbationChain([result]),
                    chain_length,
                    chains,
                    self.num_chains
                ) and (
                    self.max_iterations is None or num_iterations < self.max_iterations
                ) and (
                    self.num_chains is None or len(chains) < self.num_chains
                ):
                    num_iterations += 1

        return chains