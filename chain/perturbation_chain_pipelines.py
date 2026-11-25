import random, time

from math import sqrt
from typing import Any
from copy import deepcopy
from .perturbation_heuristics import MCTSPerturbationHeuristic
from interfaces import GraphLike, Perturbation, PerturbationHeuristic, PerturbationChainPipeline, PerturbationChain, LabelingFunction
from .perturbation_chains import chain_types

class HeuristicPerturbationChainPipeline(PerturbationChainPipeline):
    def __init__(
        self,
        graph: GraphLike,
        perturbations: list[tuple[Perturbation, PerturbationHeuristic[Any]]],
        labeling_functions: list[LabelingFunction],
        chain_type: type[PerturbationChain],
        **kwargs: Any
    ) -> None:
        super().__init__(graph, perturbations, labeling_functions, chain_type, **kwargs)

    def _find_chain(
        self,
        graph: GraphLike,
        chain: PerturbationChain,
        chain_length: int,
        chains: set[PerturbationChain]
    ) -> None:
        if (self.num_chains is not None and len(chains) >= self.num_chains) or (self.max_iterations is not None and self.num_iterations[chain_length] >= self.max_iterations):
            return
        
        if len(chain) >= chain_length:
            if not chain.check_subchain_validity(self.graph, self.labeling_functions):
                return

            if all([not chain.match_perturbation_set(c) for c in chains]):
                chains.add(chain)
        else:
            for perturbation in self.perturbations:
                heuristic: PerturbationHeuristic[Any] | None = self.heuristics[perturbation]

                while heuristic is not None and (self.max_iterations is None or self.num_iterations[chain_length] < self.max_iterations):
                    self.num_iterations[chain_length] += 1
                    result = chain.test_potential_perturbation(
                        perturbation,
                        heuristic,
                        self.labeling_functions,
                        graph,
                        desired_result = len(chain) >= (chain_length - 1)
                    )

                    if result is None:
                        heuristic = heuristic.next(self.graph, chain)
                        continue

                    heuristic = deepcopy(result.heuristic)
                    new_chain = chain + self.chain_type([result])
                    
                    if any([new_chain.match_perturbation_set(detected_chain) for detected_chain in [c for c in chains if len(c) == len(new_chain)]]):
                        return

                    self._find_chain(
                        result.graph,
                        chain + self.chain_type([result]),
                        chain_length,
                        chains
                    )

    def _find_chains(self, chain_length: int) -> set[PerturbationChain]:
        chains: set[PerturbationChain] = set()
        start_time = time.time()

        self._find_chain(
            self.graph,
            self.chain_type([]),
            chain_length,
            chains
        )
        end_time = time.time()
        self.running_time[chain_length] = end_time - start_time

        for perturbation in self.heuristics:
            self.heuristics[perturbation].reset()

        return chains

class MCTSPerturbationChainPipeline(PerturbationChainPipeline):
    def __init__(
        self,
        graph: GraphLike,
        perturbations: list[tuple[Perturbation, PerturbationHeuristic[Any]]],
        labeling_functions: list[LabelingFunction],
        chain_type: type[PerturbationChain],
        **kwargs: Any
    ) -> None:
        super().__init__(graph, perturbations, labeling_functions, chain_type, **kwargs)
        self.visualize_tree: bool = self._get_arg('visualize_tree', False, **kwargs)
        self.max_training_iterations: int = self._get_arg('max_training_iterations', 1000, **kwargs)
        self.mcts_exploration_param: float = self._get_arg('mcts_exploration_param', sqrt(2.0), **kwargs)

        start_time = time.time()
        self.heuristic: MCTSPerturbationHeuristic = MCTSPerturbationHeuristic(
            graph,
            labeling_functions,
            self.heuristics,
            self.chain_type,
            self.max_length,
            self.max_training_iterations,
            self.mcts_exploration_param,
            self.visualize_tree
        )
        self.training_time = time.time() - start_time
        self.training_found_chains = self.heuristic.get_found_chains()

    def _find_chain(
        self,
        graph: GraphLike,
        chain: PerturbationChain,
        chain_length: int,
        chains: set[PerturbationChain],
        heuristic: MCTSPerturbationHeuristic
    ) -> None:
        if (self.num_chains is not None and len(chains) >= self.num_chains) or (self.max_iterations is not None and self.num_iterations[chain_length] >= self.max_iterations):
            return
        
        if len(chain) >= chain_length:
            if not chain.check_subchain_validity(self.graph, self.labeling_functions):
                return

            if all([not chain.match_perturbation_set(c) for c in chains]):
                chains.add(chain)
        else:
            new_heuristic = heuristic

            while self.max_iterations is None or self.num_iterations[chain_length] < self.max_iterations:
                self.num_iterations[chain_length] += 1
                new_heuristic = new_heuristic.next(graph, chain)

                if new_heuristic is None:
                    break

                new_chain = chain + self.chain_type([new_heuristic.current()])

                if any([new_chain.match_perturbation_set(detected_chain) for detected_chain in [c for c in chains if len(c) == len(new_chain)]]):
                    break

                result = new_chain.test_perturbation(
                    -1,
                    graph,
                    self.labeling_functions,
                    desired_result = len(new_chain) >= chain_length
                )

                if result is None:
                    continue

                self._find_chain(
                    result.graph,
                    new_chain,
                    chain_length,
                    chains,
                    new_heuristic
                )
                
    def _find_chains(self, chain_length: int) -> set[PerturbationChain]:
        chains: set[PerturbationChain] = set()
        chains = chains.union({chain for chain in self.training_found_chains if len(chain) == chain_length})

        if self.num_chains is not None and len(chains) >= self.num_chains:
            return set(random.sample(list(chains), self.num_chains))

        start_time = time.time()

        self._find_chain(
            self.graph,
            self.chain_type([]),
            chain_length,
            chains,
            self.heuristic
        )
        end_time = time.time()
        self.running_time[chain_length] = end_time - start_time

        for perturbation in self.heuristics:
            self.heuristics[perturbation].reset()
        
        self.heuristic.reset()
        return chains

_pipeline_algorithms: dict[str, type[PerturbationChainPipeline]] = {
    'heuristic': HeuristicPerturbationChainPipeline,
    'mcts': MCTSPerturbationChainPipeline
}

def build_pipeline(
    algorithm: str,
    graph: GraphLike,
    perturbations: list[tuple[Perturbation, PerturbationHeuristic[Any]]],
    labeling_functions: list[LabelingFunction],
    chain_type: str,
    **kwargs: Any
) -> PerturbationChainPipeline:
    return _pipeline_algorithms[algorithm](
        graph,
        perturbations,
        labeling_functions,
        chain_types[chain_type],
        **kwargs
    )
