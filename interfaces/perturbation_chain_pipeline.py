import abc

from typing import Any
from abc import abstractmethod

from .graph_types import GraphLike
from .perturbation import Perturbation
from .labeling_function import LabelingFunction
from .perturbation_chain import PerturbationChain
from .perturbation_heuristic import PerturbationHeuristic

class PerturbationChainPipeline(abc.ABC):
    def __init__(
        self,
        graph: GraphLike,
        perturbations: list[tuple[Perturbation, PerturbationHeuristic[Any]]],
        labeling_functions: list[LabelingFunction],
        chain_type: type[PerturbationChain],
        **kwargs: Any
    ) -> None:
        self.graph = graph
        self.chain_type = chain_type
        self.perturbations = [perturbation for perturbation, _ in perturbations]
        self.labeling_functions = labeling_functions
        self.heuristics = {perturbation: heuristic for perturbation, heuristic in perturbations}

        self.max_length: int = self._get_arg('max_length', 4, **kwargs)
        self.num_chains: int | None = self._get_arg('num_chains', None, **kwargs)
        self.max_iterations: int | None = self._get_arg('max_iterations', None, **kwargs)

        self.num_iterations: dict[int, int] = {length: 0 for length in range(2, self.max_length + 1)}
        self.running_time: dict[int, float] = {length: 0.0 for length in range(2, self.max_length + 1)}
        self.training_time: float = 0.0

    def _get_arg(self, key: str, default: Any, **kwargs: Any) -> Any:
        return kwargs[key] if key in kwargs else default

    @abstractmethod
    def _find_chains(self, chain_length: int) -> set[PerturbationChain]:
        pass

    def find(self) -> set[PerturbationChain]:        
        for length in self.running_time:
            self.running_time[length] = 0.0
            self.num_iterations[length] = 0
            
        out: set[PerturbationChain] = set()

        for length in range(2, (self.max_length + 1)):
            out = out.union(self._find_chains(length))

        return out
