import networkx as nx

from math import sqrt
from typing import Any
from copy import deepcopy
from .perturbation_heuristics import MCTSPerturbationHeuristic
from interfaces import Perturbation, PerturbationHeuristic, PerturbationChainPipeline, PerturbationChain, LabelingFunction

class HeuristicPerturbationChainPipeline(PerturbationChainPipeline):
    def __init__(
        self,
        graph: nx.Graph[Any],
        perturbations: list[tuple[Perturbation, PerturbationHeuristic[Any]]],
        labeling_functions: list[LabelingFunction],
        chain_type: type[PerturbationChain],
        max_length: int = 4,
        max_iterations: int | None = None,
        num_chains: int | None = None,
        ordered: bool = False
    ) -> None:
        super().__init__(graph, perturbations, labeling_functions, chain_type, max_length, max_iterations, num_chains)
        self.ordered = ordered

    def _find_chain(
        self,
        graph: nx.Graph[Any],
        chain: PerturbationChain,
        chain_length: int,
        chains: set[PerturbationChain],
        num_chains: int | None
    ) -> None:
        self.num_iterations += 1
        
        if (num_chains is not None and len(chains) >= num_chains) or (self.max_iterations is not None and self.num_iterations > self.max_iterations):
            return
        
        if len(chain) >= chain_length:
            if not chain.check_subchain_validity(self.graph, self.labeling_functions):
                return

            if self.ordered or (not self.ordered and not any([chain.match_perturbation_set(c) for c in chains])):
                chains.add(chain)
        else:
            for perturbation in self.perturbations:
                heuristic: PerturbationHeuristic[Any] | None = self.heuristics[perturbation]

                while heuristic is not None:
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
                    
                    if any([new_chain == detected_chain for detected_chain in [c for c in chains if len(c) == len(new_chain)]]):
                        return

                    self._find_chain(
                        result.graph,
                        chain + self.chain_type([result]),
                        chain_length,
                        chains,
                        num_chains
                    )

    def _find_chains(self, chain_length: int) -> set[PerturbationChain]:
        chains: set[PerturbationChain] = set()
    
        for perturbation in self.perturbations:
            heuristic: PerturbationHeuristic[Any] | None = self.heuristics[perturbation]

            while heuristic is not None and (
                self.max_iterations is None or self.num_iterations < self.max_iterations
            ) and (
                self.num_chains is None or len(chains) < self.num_chains
            ):
                chain = self.chain_type([])                
                result = chain.test_potential_perturbation(perturbation, heuristic, self.labeling_functions, self.graph)

                if result is None:
                    heuristic = heuristic.next(self.graph, chain)
                    continue

                heuristic = deepcopy(result.heuristic)
                result.heuristic.soft_reset()
                self.heuristics[perturbation] = deepcopy(result.heuristic)
                
                self._find_chain(
                    result.graph,
                    self.chain_type([result]),
                    chain_length,
                    chains,
                    self.num_chains
                )

        for perturbation in self.heuristics:
            self.heuristics[perturbation].reset()

        return chains

class MCTSPerturbationChainPipeline(PerturbationChainPipeline):
    def __init__(
        self,
        graph: nx.Graph[Any],
        perturbations: list[tuple[Perturbation, PerturbationHeuristic[Any]]],
        labeling_functions: list[LabelingFunction],
        chain_type: type[PerturbationChain],
        max_length: int = 4,
        max_training_iterations: int = 1000,
        mcts_exploration_param: float = sqrt(2.0),
        num_chains: int | None = None,
        ordered: bool = False,
        visualize_tree: bool = False
    ) -> None:
        super().__init__(graph, perturbations, labeling_functions, chain_type, max_length, num_chains = num_chains)
        self.ordered = ordered
        self.visualize_tree = visualize_tree
        self.heuristic = MCTSPerturbationHeuristic(
            graph,
            labeling_functions,
            self.heuristics,
            self.chain_type,
            max_length,
            max_training_iterations,
            mcts_exploration_param,
            visualize_tree,
            ordered
        )
        self.training_found_chains = self.heuristic.get_found_chains()

    def _find_chain(
        self,
        chain: PerturbationChain,
        chain_length: int,
        chains: set[PerturbationChain],
        num_chains: int | None,
        heuristic: MCTSPerturbationHeuristic
    ) -> None:
        self.num_iterations += 1
        
        if (num_chains is not None and len(chains) >= num_chains) or (self.max_iterations is not None and self.num_iterations > self.max_iterations):
            return
        
        if len(chain) >= chain_length:
            if not chain.check_subchain_validity(self.graph, self.labeling_functions):
                return

            if self.ordered or (not self.ordered and not any([chain.match_perturbation_set(c) for c in chains])):
                chains.add(chain)
        else:
            new_heuristic = heuristic

            while True:
                new_heuristic = new_heuristic.next(self.graph, chain)
                working_graph = chain[-1].graph if len(chain) > 0 else self.graph

                if new_heuristic is None:
                    return

                new_chain = chain + self.chain_type([new_heuristic.current()])

                if any([new_chain == detected_chain for detected_chain in [c for c in chains if len(c) == len(new_chain)]]):
                    return
                
                result = new_chain.test_perturbation(
                    -1,
                    working_graph,
                    self.labeling_functions,
                    desired_result = len(new_chain) >= chain_length
                )

                if result is None:
                    continue

                self._find_chain(new_chain, chain_length, chains, num_chains, new_heuristic)

    def _find_chains(self, chain_length: int) -> set[PerturbationChain]:
        chains: set[PerturbationChain] = set()
        chains = chains.union(self.training_found_chains)
    
        for perturbation in self.perturbations:
            heuristic = self.heuristic

            while heuristic is not None and (
                self.max_iterations is None or self.num_iterations < self.max_iterations
            ) and (
                self.num_chains is None or len(chains) < self.num_chains
            ):
                heuristic = heuristic.next(self.graph, self.chain_type([]))

                if heuristic is None:
                    break

                chain = self.chain_type([heuristic.current()])

                if chain.test_perturbation(-1, self.graph, self.labeling_functions) is None:
                    heuristic = heuristic.next(self.graph, self.chain_type([]))
                    continue

                new_heuristic = deepcopy(heuristic)
                new_heuristic.soft_reset()

                self._find_chain(
                    chain,
                    chain_length,
                    chains,
                    self.num_chains,
                    new_heuristic
                )

        for perturbation in self.heuristics:
            self.heuristics[perturbation].reset()
        
        self.heuristic.reset()
        return chains