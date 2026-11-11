import math
import random
import graphviz
import networkx as nx

from copy import deepcopy
from typing import Any
from interfaces import LabelingFunction
from interfaces import Perturbation, PerturbationHeuristic, PerturbationChain

class MCTSTree:
    def __init__(
        self,
        parent: MCTSTree | None,
        chain: PerturbationChain,
        chain_type: type[PerturbationChain],
        expansion_heuristics: dict[Perturbation, PerturbationHeuristic[Any]],
        graph: nx.Graph[Any],
        exploration_param: float = math.sqrt(2.0)
    ) -> None:
        self.level = 0
        self.score = 0
        self.num_simulations = 0
        self.exploration_param = exploration_param

        self.chain = chain
        self.graph = graph
        self.parent = parent
        self.chain_type = chain_type
        self.expansion_heuristics = expansion_heuristics
        self.initial_heuristics = {p: deepcopy(self.expansion_heuristics[p]) for p in self.expansion_heuristics}
        self.simulation_heuristics = {p: deepcopy(self.expansion_heuristics[p]) for p in self.expansion_heuristics}

        self.children: list[MCTSTree] = []
        self.expanded: dict[Perturbation, bool] = {perturbation: False for perturbation in self.expansion_heuristics}
        
    def add_child(self, child: MCTSTree) -> None:
        child.level = self.level + 1
        self.children.append(child)

    def expand(self) -> None:
        for perturbation in [p for p in self.expanded if self.expanded[p] == False]:
            heuristic = self.expansion_heuristics[perturbation]

            while True:
                graph = self.chain[-1].graph if len(self.chain) > 0 else self.graph
                heuristic = heuristic.next(graph, self.chain)

                if heuristic is None:
                    self.expanded[perturbation] = True
                    break
                
                self.expansion_heuristics[perturbation] = heuristic

                result = perturbation.apply_heuristic(graph, deepcopy(heuristic))
                new_chain = self.chain + self.chain_type([result])                

                heuristics = {p: deepcopy(self.initial_heuristics[p]) for p in self.initial_heuristics}
                heuristics[perturbation].add(heuristic.current())

                child = MCTSTree(
                    self,
                    new_chain,
                    self.chain_type,
                    heuristics,
                    graph
                )
                
                if child not in self.children:
                    self.add_child(child)
                    break

    def simulate(
        self,
        graph: nx.Graph[Any],
        labeling_functions: list[LabelingFunction],
        chain_length: int,
        chains: set[PerturbationChain],
        ordered: bool
    ) -> bool:
        step = 0
        chain = deepcopy(self.chain)

        acc_graph = self.chain[-1].graph if len(self.chain) > 0 else self.graph
        num_steps = chain_length - self.level

        while step < num_steps:
            perturbation = random.choice(list(self.expansion_heuristics.keys()))
            result = chain.test_potential_perturbation(
                perturbation,
                self.simulation_heuristics[perturbation],
                labeling_functions, acc_graph, step >= (num_steps - 1)
            )

            if result is None:
                return False

            chain.append(result)
            acc_graph = result.graph

            self.simulation_heuristics[perturbation] = result.heuristic
            step += 1

        self.simulation_heuristics = {p: deepcopy(self.expansion_heuristics[p]) for p in self.expansion_heuristics}

        if chain.is_valid(graph, labeling_functions):
            if ordered or (not ordered and not any([chain.match_perturbation_set(c) for c in chains])):
                chains.add(chain)
            return True

        return False

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
        return (self.is_expanded() and len(self.children) <= 0)

    def visualize(self, dot: Any) -> None:
        dot.node(f'{self.chain}')

        if self.parent is not None:
            dot.edge(f'{self.parent.chain}', f'{self.chain}')

        for child in self.children:
            child.visualize(dot)

    def perturbation(self) -> Perturbation | None:
        return self.chain[-1].perturbation if len(self.chain) > 0 else None

    def perturbed_element(self) -> Any:
        return self.chain[-1].heuristic.current() if len(self.chain) > 0 else None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MCTSTree):
            return False
        
        return type(self.perturbation()) == type(other.perturbation()) and self.perturbed_element() == other.perturbed_element()