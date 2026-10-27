import networkx as nx

from typing import Any
from copy import deepcopy
from interfaces import Perturbation, PerturbationWalk, PerturbationChainPipeline, PerturbationChain, LabelingFunction

class BruteForcePerturbationChainPipeline(PerturbationChainPipeline):
    def __init__(
        self,
        graph: nx.Graph[Any],
        perturbations: list[tuple[Perturbation, PerturbationWalk[Any]]],
        labeling_functions: list[LabelingFunction],
        max_length: int = 4,
        max_iterations: int | None = None,
        num_chains: int | None = None,
        ordered: bool = False
    ) -> None:
        super().__init__(graph, perturbations, labeling_functions, max_length, max_iterations, num_chains)
        self.ordered = ordered

    def _test_subchain(
        self,
        chain: PerturbationChain,
        subchain: PerturbationChain,
        subchain_length: int
    ) -> bool:
        out = True

        if len(subchain) >= subchain_length:
            result = None
            acc_graph = self.graph

            for perturbation in subchain:
                result = self._test_applied_perturbation(perturbation, subchain, acc_graph)

                if result is not None:
                    acc_graph = result.graph
    
            out = result is not None
        else:
            for perturbation in [p for p in chain if p not in subchain]:
                out = self._test_subchain(
                    chain,
                    subchain + PerturbationChain([perturbation]),
                    subchain_length
                )
                
                if out is False:
                    break;
        return out        

    def _find_chain(
        self,
        graph: nx.Graph[Any],
        chain: PerturbationChain,
        chain_length: int,
        chains: list[PerturbationChain],
        num_chains: int | None
    ) -> None:
        if num_chains is not None and len(chains) >= num_chains:
            return
        
        if len(chain) >= chain_length:
            result = None

            if chain_length > 2:
                for subchain_length in range(2, chain_length):
                    for perturbation in chain:
                        if not self._test_subchain(
                            chain,
                            PerturbationChain([perturbation]),
                            subchain_length
                        ):
                            return
            
            if self.ordered or (not self.ordered and not any([chain == c for c in chains])):
                chains.append(chain)
        else:
            for perturbation in self.perturbations:
                walk: PerturbationWalk[Any] | None = self.walks[perturbation]

                while walk is not None:
                    result = self._test_perturbation(perturbation, walk, chain, graph, desired_result = len(chain) >= (chain_length - 1))

                    if result is None:
                        walk = walk.next(self.graph, chain)
                        continue

                    walk = deepcopy(result.walk)

                    self._find_chain(
                        result.graph,
                        chain + PerturbationChain([result]),
                        chain_length,
                        chains,
                        num_chains
                    )