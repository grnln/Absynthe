import abc
import time
import networkx as nx

from typing import Any
from abc import abstractmethod

from interfaces.perturbation import Perturbation
from interfaces.perturbation_heuristic import PerturbationHeuristic
from interfaces.labeling_function import LabelingFunction

from .perturbation_chain import PerturbationChain

class PerturbationChainPipeline(abc.ABC):
    def __init__(
        self,
        graph: nx.Graph[Any],
        perturbations: list[tuple[Perturbation, PerturbationHeuristic[Any]]],
        labeling_functions: list[LabelingFunction],
        chain_type: type[PerturbationChain],
        max_length: int,
        max_iterations: int | None = None,
        num_chains: int | None = None
    ) -> None:
        self.graph = graph
        self.max_length = max_length
        self.chain_type = chain_type
        self.num_chains = num_chains
        self.max_iterations = max_iterations
        self.perturbations = [perturbation for perturbation, _ in perturbations]
        self.labeling_functions = labeling_functions
        self.heuristics = {perturbation: heuristic for perturbation, heuristic in perturbations}
        self.running_time = 0

    @abstractmethod
    def _find_chains(self, chain_length: int) -> set[PerturbationChain]:
        pass

    def find(self) -> set[PerturbationChain]:
        start_time = time.time()
        out: set[PerturbationChain] = set()

        for length in range(2, (self.max_length + 1)):
            out = out.union(self._find_chains(length))

        end_time = time.time()
        self.running_time = (end_time - start_time)
        return out