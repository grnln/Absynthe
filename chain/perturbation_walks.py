import networkx as nx

from typing import Any
from interfaces import PerturbationWalk, PerturbationChain

class NaturalOrderNodePerturbationWalk(PerturbationWalk[Any]):
    def __init__(self) -> None:
        self.traversed_nodes: list[Any] = []

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationWalk[Any] | None:
        index = 0
        node_list = sorted(list(graph.nodes))
        
        if len(self.traversed_nodes) > 0:
            last_node = self.traversed_nodes[-1]
            index = node_list.index(last_node) + 1

        if (index >= len(node_list)):
            return None

        node = node_list[index]
        new_walk = NaturalOrderNodePerturbationWalk()
        new_walk.traversed_nodes.append(node)
        return new_walk

    def current(self) -> Any:
        return self.traversed_nodes[-1]

    def reset(self) -> None:
        self.traversed_nodes = []

    def soft_reset(self) -> None:
        if len(self.traversed_nodes) > 0:
            self.traversed_nodes = [self.traversed_nodes[-1]]

    def has_next(self, graph: nx.Graph[Any]) -> bool:
        return (len(self.traversed_nodes) < len(list(graph.nodes)))
        
class NaturalOrderEdgePerturbationWalk(PerturbationWalk[Any]):
    def __init__(self) -> None:
        self.traversed_edges: list[Any] = []

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationWalk[Any] | None:
        index = 0
        edge_list = sorted(list(graph.edges), key = lambda x: x[0])

        if len(self.traversed_edges) > 0:
            last_edge = self.traversed_edges[-1]
            index = edge_list.index(last_edge) + 1

        if (index >= len(edge_list)):
            return None

        edge = edge_list[index]
        new_walk = NaturalOrderEdgePerturbationWalk()
        new_walk.traversed_edges.append(edge)
        return new_walk

    def current(self) -> Any:
        return self.traversed_edges[-1]

    def reset(self) -> None:
        self.traversed_edges = []

    def soft_reset(self) -> None:
        if len(self.traversed_edges) > 0:
            self.traversed_edges = [self.traversed_edges[-1]]

    def has_next(self, graph: nx.Graph[Any]) -> bool:
        return (len(self.traversed_edges) < len(list(graph.edges)))