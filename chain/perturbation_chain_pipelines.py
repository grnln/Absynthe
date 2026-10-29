import graphviz
import networkx as nx

from typing import Any
from copy import deepcopy
from datetime import datetime
from .mcts_tree import MCTSTree
from interfaces import Perturbation, PerturbationWalk, PerturbationChainPipeline, PerturbationChain, LabelingFunction

class HeuristicPerturbationChainPipeline(PerturbationChainPipeline):
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

class MCTSPerturbationChainPipeline(PerturbationChainPipeline):
    def __init__(
        self,
        graph: nx.Graph[Any],
        perturbations: list[tuple[Perturbation, PerturbationWalk[Any]]],
        labeling_functions: list[LabelingFunction],
        random_walks: dict[Perturbation, PerturbationWalk[Any]],
        max_length: int = 4,
        max_mcts: int = 10,
        max_mcts_iterations: int = 1000,
        num_chains: int | None = None,
        ordered: bool = False,
        visualize_tree: bool = False
    ) -> None:
        super().__init__(graph, perturbations, labeling_functions, max_length, num_chains = num_chains)
        self.ordered = ordered
        self.max_mcts = max_mcts
        self.random_walks = random_walks
        self.visualize_tree = visualize_tree
        self.max_mcts_iterations = max_mcts_iterations

    def _check_chain(self, chain: PerturbationChain) -> bool:
        result = None
        acc_graph = self.graph

        for i in range(len(chain)):
            result = self._test_applied_perturbation(chain[i], acc_graph, i >= (len(chain) - 1))

            if result is None:
                break

            acc_graph = result.graph

        return (result is not None)

    def _check_subchain(self, chain: PerturbationChain) -> bool:
        if len(chain) > 2:
            for subchain_length in range(2, len(chain)):
                for perturbation in chain:
                    if not self._test_subchain(
                        chain,
                        PerturbationChain([perturbation]),
                        subchain_length
                    ):
                        return False
        return True

    def _train_tree(self, tree: MCTSTree) -> None:
        for _ in range(self.max_mcts_iterations):
            node = tree
            
            while not node.is_leaf() and node.is_expanded():
                node = node.best_child()
        
            if not node.is_leaf() and not node.is_expanded():
                node.expand()

            if len(node.children) <= 0:
                node = tree
        
            valid_chain = node.simulate(self._test_perturbation, self._test_subchain)
            node.backpropagate(valid_chain)

    def _visualize_tree(self, tree: MCTSTree) -> None:
        dot = graphviz.Graph(f'MCTS_{datetime.now().strftime("%Y%m%d_%H%M%S")}', format = 'png')
        tree.visualize(dot)
        dot.render(directory = 'visualizations')

    def _build_chain(self, tree: MCTSTree, chain_length: int) -> PerturbationChain | None:
        candidate_chain = PerturbationChain([tree.perturbation])
        
        for _ in range(chain_length - 1):
            if len(tree.children) <= 0:
                break
                        
            tree = tree.most_simulated_child()
            candidate_chain.append(tree.perturbation)

        if len(candidate_chain) != chain_length:
            return None
        
        return candidate_chain

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

        tree = MCTSTree(
            None,
            chain[0].perturbation,
            self.walks,
            self.random_walks,
            self.graph,
            chain,
            chain_length
        )
        self._train_tree(tree)
        
        for _ in range(self.max_mcts):
            candidate_chain = self._build_chain(tree, chain_length)
            
            if candidate_chain is None:
                continue

            if not self._check_chain(candidate_chain):
                continue

            if not self._check_subchain(candidate_chain):
                continue

            if self.ordered or (not self.ordered and not any([candidate_chain == c for c in chains])):
                chains.append(candidate_chain)

        if self.visualize_tree:
            self._visualize_tree(tree)