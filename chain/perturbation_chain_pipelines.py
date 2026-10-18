import networkx as nx

from typing import Any

from interfaces import Perturbation, PerturbationWalk, PerturbationChainPipeline, AppliedPerturbation, PerturbationChain, LabelingFunction

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
                result, acc_graph, _ = self._test_perturbation(perturbation.perturbation, perturbation.walk, acc_graph)
    
            out = result is True
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
    ) -> None:
        if len(chain) >= chain_length:
            result = None
            acc_graph = self.graph

            for perturbation in chain:
                result, acc_graph, _ = self._test_perturbation(perturbation.perturbation, perturbation.walk, acc_graph)
    
            if result is True:
                chains.append(chain)
        else:
            for perturbation, walk in self.perturbations:
                discard = False
                result, candidate_graph, changes = self._test_perturbation(perturbation, walk, self.graph)

                if result is False:
                    continue

                candidate_chain = chain + PerturbationChain([AppliedPerturbation(perturbation, walk, candidate_graph, changes)])

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

                self._find_chain(
                    chain + PerturbationChain([AppliedPerturbation(perturbation, walk, candidate_graph, changes)]),
                    chain_length,
                    chains
                )