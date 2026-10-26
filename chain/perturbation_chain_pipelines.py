import networkx as nx

from typing import Any

from interfaces import Perturbation, PerturbationWalk, PerturbationChainPipeline, PerturbationChain, LabelingFunction

class BruteForcePerturbationChainPipeline(PerturbationChainPipeline):
    def __init__(
        self,
        graph: nx.Graph[Any],
        perturbations: list[tuple[Perturbation, PerturbationWalk[Any]]],
        labeling_functions: list[LabelingFunction],
        max_length: int = 4
    ) -> None:
        super().__init__(graph, perturbations, labeling_functions, max_length)

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
        chain: PerturbationChain,
        chain_length: int,
        chains: list[PerturbationChain]
    ) -> bool:
        out = True
        
        if len(chain) >= chain_length:
            result = None
            acc_graph = self.graph

            for perturbation in chain:
                result  = self._test_applied_perturbation(perturbation, chain, acc_graph)

                if result is not None:
                    acc_graph = result.graph
    
            if result is None:
                chains.append(chain)
        else:
            for perturbation in self.perturbations:
                discard = False
                walk = self.walks[perturbation]

                result = self._test_perturbation(perturbation, walk, chain, self.graph)

                if result is None:
                    new_walk = walk.next(self.graph, chain)

                    if new_walk is not None:
                        self.walks[perturbation] = new_walk
                        continue
                    else:
                        return False

                candidate_chain = chain + PerturbationChain([result])

                # Discard the perturbation if one of the possible sub-chains involving it changes labels
                for subchain_length in range(2, min(len(candidate_chain), chain_length)):
                    for perturb in candidate_chain:
                        if self._test_subchain(
                            candidate_chain,
                            PerturbationChain([perturb]),
                            subchain_length,
                        ) is False:
                            discard = True

                if discard is True:
                    continue;

                self.walks[perturbation] = result.walk

                out = self._find_chain(
                    candidate_chain,
                    chain_length,
                    chains
                )
        return out