import math
import random
import graphviz
import networkx as nx

from copy import deepcopy
from typing import Any, Callable
from interfaces import Perturbation, PerturbationWalk, PerturbationChain, AppliedPerturbation

class MCTSTree:
    def __init__(
        self,
        parent: MCTSTree | None,
        perturbation: Perturbation,
        walks: dict[Perturbation, PerturbationWalk[Any]],
        random_walks: dict[Perturbation, PerturbationWalk[Any]],
        graph: nx.Graph[Any],
        chain: PerturbationChain,
        chain_length: int,
        exploration_param: float = math.sqrt(2.0)
    ) -> None:
        self.level = 0
        self.score = 0
        self.num_simulations = 0
        self.exploration_param = exploration_param

        self.walks = walks
        self.parent = parent
        self.chain_length = chain_length
        self.random_walks = random_walks

        self.children: list[MCTSTree] = []
        self.expanded: dict[Perturbation, bool] = {perturbation: False for perturbation in walks}
        
        self.perturbation = perturbation.apply_by_walk(graph, walks[perturbation])
        self.chain = chain + PerturbationChain([self.perturbation])

    def perturbed_element(self) -> Any:
        return self.perturbation.walk.current()

    def add_child(self, child: MCTSTree) -> None:
        child.level = self.level + 1
        self.children.append(child)

    def expand(self) -> None:
        for perturbation in [p for p in self.expanded if self.expanded[p] == False]:
            walks = {p: self.walks[p] for p in self.walks}
            walk = walks[perturbation]

            while True:
                walk = walk.next(self.perturbation.graph, self.chain)

                if walk is None:
                    break
                
                walks[perturbation] = walk
                
                child = MCTSTree(
                    self,
                    perturbation,
                    walks,
                    self.random_walks,
                    self.perturbation.graph,
                    self.chain,
                    self.chain_length
                )
                
                if child not in self.children:
                    self.add_child(child)
                    break

            if walk is None:
                self.expanded[perturbation] = True

    def simulate(
        self,
        test_perturbation: Callable[[Perturbation, PerturbationWalk[Any], PerturbationChain, nx.Graph[Any], bool], AppliedPerturbation | None],
        test_subchain: Callable[[PerturbationChain, PerturbationChain, int], bool]
    ) -> bool:
        step = 0
        chain = deepcopy(self.chain)

        acc_graph = self.perturbation.graph
        num_steps = self.chain_length - (self.level + 1)

        while step < num_steps:
            perturbation = random.choice(list(self.walks.keys()))
            walk = self.random_walks[perturbation].next(acc_graph, chain)

            if walk is None:
                return False

            result = test_perturbation(perturbation, self.random_walks[perturbation], chain, acc_graph, step >= (num_steps - 1))

            if result is None:
                return False

            chain.append(result)
            acc_graph = result.graph

            self.random_walks[perturbation] = result.walk
            step += 1

        if self.chain_length > 2:
            for subchain_length in range(2, self.chain_length):
                for perturbation in chain:
                    if not test_subchain(
                        chain,
                        PerturbationChain([perturbation]),
                        subchain_length
                    ):
                        return False

        for perturbation in self.walks:
            self.random_walks[perturbation].reset()
        return True

    def backpropagate(self, valid_chain: bool) -> None:
        self.num_simulations += 1

        if valid_chain:
            self.score += 1

        if self.parent is not None:
            self.parent.backpropagate(valid_chain)

    def _uct(self, child: MCTSTree) -> float:
        return (
            (child.score / child.num_simulations)
            + (self.exploration_param * math.sqrt(math.log(self.num_simulations) / child.num_simulations))
        )

    def best_child(self) -> MCTSTree:
        for child in self.children:
            if child.num_simulations == 0:
                return child

        return max(self.children, key = lambda c: self._uct(c))

    def most_simulated_child(self) -> MCTSTree:
        return max(self.children, key = lambda c: c.num_simulations)

    def is_expanded(self) -> bool:
        return all(self.expanded.values())

    def is_leaf(self) -> bool:
        return ((self.level + 1) >= self.chain_length) or (self.is_expanded() and len(self.children) <= 0)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MCTSTree):
            return False
        
        return type(self.perturbation) == type(other.perturbation) and self.perturbed_element() == other.perturbed_element()

    def visualize(self, dot: graphviz.Dot) -> None:
        dot.node(f'{self.level}. {self.perturbation}')

        if self.parent is not None:
            dot.edge(f'{self.parent.level}. {self.parent.perturbation}', f'{self.level}. {self.perturbation}')

        for child in self.children:
            child.visualize(dot)