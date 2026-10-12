import networkx as nx

from typing import Any

from interfaces.perturbation import Perturbation
from interfaces.labeling_function import LabelingFunction

from interfaces.perturbation_chain_pipeline import PerturbationChainPipeline
from interfaces.perturbation_chain import AppliedPerturbation, PerturbationChain

class BruteForcePerturbationChainPipeline(PerturbationChainPipeline):
    def __init__(
        self,
        graph: nx.Graph[Any],
        perturbations: list[Perturbation],
        labeling_functions: list[LabelingFunction],
        max_iterations: int = 40,
        max_length: int = 4
    ) -> None:
        super().__init__(graph, perturbations, labeling_functions, max_iterations, max_length)

    def test_subchain(
        self,
        chain: PerturbationChain,
        subchain: PerturbationChain,
        current_length: int,
        subchain_length: int
    ) -> bool:
        out = True
        
        if current_length >= subchain_length:
            result = None
            acc_graph = self.graph

            for perturbation in subchain:
                result, acc_graph, _ = self._test_perturbation(perturbation.perturbation, acc_graph)
    
            out = result is True
        else:
            for perturbation in [p for p in chain if p not in subchain]:
                out = self._test_subchain(
                    chain,
                    subchain + PerturbationChain([perturbation]),
                    len(subchain) + 1,
                    subchain_length
                )
                
                if out is False:
                    break;
        return out        

    def find_chain(
        self,
        chain: PerturbationChain,
        current_length: int,
        chain_length: int,
        chains: list[PerturbationChain]
    ) -> None:
        if current_length >= chain_length:
            result = None
            acc_graph = self.graph

            for perturbation in chain:
                result, acc_graph, _ = self._test_perturbation(perturbation.perturbation, acc_graph)
    
            if result is True:
                chains.append(chain)
        else:
            for perturbation in self.perturbations:
                discard = False
                result, candidate_graph, changes = self._test_perturbation(perturbation, self.graph)

                if result is False:
                    continue

                candidate_chain = chain + PerturbationChain([AppliedPerturbation(perturbation, candidate_graph, changes)])

                # Discard the perturbation if one of the possible sub-chains involving it changes labels
                for subchain_length in range(2, min(len(candidate_chain), chain_length)):
                    for perturb in candidate_chain:
                        if self.test_subchain(
                            candidate_chain,
                            PerturbationChain([perturb]),
                            1,
                            subchain_length,
                        ) is False:
                            discard = True

                if discard is True:
                    continue;

                self.find_chain(
                    chain + PerturbationChain([AppliedPerturbation(perturbation, candidate_graph, changes)]),
                    len(chain) + 1,
                    chain_length,
                    chains
                )