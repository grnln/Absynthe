import networkx as nx

from math import sqrt
from copy import deepcopy
from typing import Any, Callable
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
        new_walk = deepcopy(self)
        new_walk.traversed_nodes.append(node)
        return new_walk

    def current(self) -> Any:
        return self.traversed_nodes[-1]

    def reset(self) -> None:
        self.traversed_nodes = []

    def soft_reset(self) -> None:
        if len(self.traversed_nodes) > 0:
            self.traversed_nodes = [self.traversed_nodes[-1]]

class NaturalOrderEdgePerturbationWalk(PerturbationWalk[tuple[Any, Any]]):
    def __init__(self) -> None:
        self.traversed_edges: list[tuple[Any, Any]] = []

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationWalk[tuple[Any, Any]] | None:
        index = 0
        edge_list = sorted(list(graph.edges), key = lambda x: x[0])

        if len(self.traversed_edges) > 0:
            last_edge = self.traversed_edges[-1]
            index = edge_list.index(last_edge) + 1

        if (index >= len(edge_list)):
            return None
        
        edge = edge_list[index]
        new_walk = deepcopy(self)
        new_walk.traversed_edges.append(edge)
        return new_walk

    def current(self) -> tuple[Any, Any]:
        return self.traversed_edges[-1]

    def reset(self) -> None:
        self.traversed_edges = []

    def soft_reset(self) -> None:
        if len(self.traversed_edges) > 0:
            self.traversed_edges = [self.traversed_edges[-1]]

node_perturbation_types: set[str] = {
    'removed_nodes'
}

def node_euclidean_distance(a: Any, b: Any) -> float:
    return abs(float(b) - float(a))

class ProximityNodePerturbationWalk(PerturbationWalk[Any]):
    def __init__(self, distance_func: Callable[[Any, Any], float] = node_euclidean_distance) -> None:
        self.traversed_nodes: list[Any] = []
        self.distance_func = distance_func

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationWalk[Any] | None:
        last_perturbation = None
        last_perturbation_type = ''
        
        new_walk = deepcopy(self)
        candidate_nodes = [n for n in list(graph.nodes) if n not in self.traversed_nodes]
        
        for perturbation in reversed(chain):
            for perturbation_type in node_perturbation_types:
                if perturbation_type in perturbation.changes:
                    last_perturbation_type = perturbation_type
                    last_perturbation = perturbation
                    break

        if last_perturbation is None:
            if len(candidate_nodes) > 0:
                new_walk.traversed_nodes.append(candidate_nodes[0])
                return new_walk
            else:
                return None

        last_affected_nodes = [affected_node['id'] for affected_node in last_perturbation.changes[last_perturbation_type]]
        
        candidate_nodes.sort(
            key = lambda candidate_node:
                sum([self.distance_func(candidate_node, affected_node) for affected_node in last_affected_nodes])
                / len(last_affected_nodes)
        )
        
        if len(candidate_nodes) > 0:
            new_walk.traversed_nodes.append(candidate_nodes[0])
            return new_walk
        else:
            return None

    def current(self) -> tuple[Any, Any]:
        return self.traversed_nodes[-1]

    def reset(self) -> None:
        self.traversed_nodes = []

    def soft_reset(self) -> None:
        if len(self.traversed_nodes) > 0:
            self.traversed_nodes = [self.traversed_nodes[-1]]

edge_perturbation_types: set[str] = {
    'added_edges',
    'removed_edges'
}

def edge_euclidean_distance(a: tuple[Any, Any], b: tuple[Any, Any]) -> float:
    a_u = float(a[0])
    a_v = float(a[1])

    b_u = float(b[0])
    b_v = float(b[1])
    return sqrt(((b_u - a_u) * (b_u - a_u)) + ((b_v - a_v) * (b_v - a_v)))

class ProximityEdgePerturbationWalk(PerturbationWalk[tuple[Any, Any]]):
    def __init__(self, distance_func: Callable[[tuple[Any, Any], tuple[Any, Any]], float] = edge_euclidean_distance) -> None:
        self.traversed_edges: list[tuple[Any, Any]] = []
        self.distance_func = distance_func

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationWalk[tuple[Any, Any]] | None:
        last_perturbation = None
        last_perturbation_type = ''
        
        new_walk = deepcopy(self)
        candidate_edges = [e for e in list(graph.edges) if e not in self.traversed_edges]
        
        for perturbation in reversed(chain):
            for perturbation_type in edge_perturbation_types:
                if perturbation_type in perturbation.changes:
                    last_perturbation_type = perturbation_type
                    last_perturbation = perturbation
                    break

        if last_perturbation is None:
            if len(candidate_edges) > 0:
                new_walk.traversed_edges.append(candidate_edges[0])
                return new_walk
            else:
                return None

        last_affected_edges = [(affected_edge['u'], affected_edge['v']) for affected_edge in last_perturbation.changes[last_perturbation_type]]
        
        candidate_edges.sort(
            key = lambda candidate_edge:
                sum([self.distance_func(candidate_edge, affected_edge) for affected_edge in last_affected_edges])
                / len(last_affected_edges)
        )
        
        if len(candidate_edges) > 0:
            new_walk.traversed_edges.append(candidate_edges[0])
            return new_walk
        else:
            return None

    def current(self) -> tuple[Any, Any]:
        return self.traversed_edges[-1]

    def reset(self) -> None:
        self.traversed_edges = []

    def soft_reset(self) -> None:
        if len(self.traversed_edges) > 0:
            self.traversed_edges = [self.traversed_edges[-1]]