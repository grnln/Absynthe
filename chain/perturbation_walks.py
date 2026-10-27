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

node_perturbation_types: set[str] = {
    'removed_nodes'
}

def node_graph_distance(a: Any, b: Any, graph: nx.Graph[Any]) -> float:
    return nx.shortest_path_length(graph, source = a, target = b) # type: ignore

def node_euclidean_distance(a: Any, b: Any, graph: nx.Graph[Any]) -> float:
    return abs(float(b) - float(a))

class ProximityNodePerturbationWalk(PerturbationWalk[Any]):
    def __init__(self, graph: nx.Graph[Any], distance_func: Callable[[Any, Any, nx.Graph[Any]], float] = node_graph_distance) -> None:
        self.graph = graph.copy()
        self.traversed_nodes: list[Any] = []
        self.distance_func = distance_func

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationWalk[Any] | None:
        perturbed_nodes_raw: list[dict[str, Any]] = []
        
        new_walk = deepcopy(self)
        candidate_nodes = [n for n in list(graph.nodes) if n not in self.traversed_nodes]
        
        for perturbation in reversed(chain):
            for perturbation_type in node_perturbation_types:
                if perturbation_type in perturbation.changes:
                    perturbed_nodes_raw.extend(perturbation.changes[perturbation_type])

        if len(perturbed_nodes_raw) == 0:
            if len(candidate_nodes) > 0:
                new_walk.traversed_nodes.append(candidate_nodes[0])
                return new_walk
            else:
                return None

        perturbed_nodes = [perturbed_node['id'] for perturbed_node in perturbed_nodes_raw]
        
        candidate_nodes.sort(
            key = lambda candidate_node:
                sum([self.distance_func(candidate_node, perturbed_node, self.graph) for perturbed_node in perturbed_nodes])
                / len(perturbed_nodes)
        )
        
        if len(candidate_nodes) > 0:
            new_walk.traversed_nodes.append(candidate_nodes[0])
            return new_walk
        else:
            return None

    def current(self) -> tuple[Any, Any]:
        return self.traversed_nodes[-1]
    
edge_perturbation_types: set[str] = {
    'added_edges',
    'removed_edges'
}

def edge_graph_distance(a: tuple[Any, Any], b: tuple[Any, Any], graph: nx.Graph[Any]) -> float:
    return min([node_graph_distance(u, v, graph) for v in b for u in a])

def edge_euclidean_distance(a: tuple[Any, Any], b: tuple[Any, Any], graph: nx.Graph[Any]) -> float:
    a_u = float(a[0])
    a_v = float(a[1])

    b_u = float(b[0])
    b_v = float(b[1])
    return sqrt(((b_u - a_u) * (b_u - a_u)) + ((b_v - a_v) * (b_v - a_v)))

class ProximityEdgePerturbationWalk(PerturbationWalk[tuple[Any, Any]]):
    def __init__(self, graph: nx.Graph[Any], distance_func: Callable[[tuple[Any, Any], tuple[Any, Any], nx.Graph[Any]], float] = edge_graph_distance) -> None:
        self.graph = graph.copy()
        self.traversed_edges: list[tuple[Any, Any]] = []
        self.distance_func = distance_func

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationWalk[tuple[Any, Any]] | None:
        perturbed_edges_raw: list[dict[str, Any]] = []

        new_walk = deepcopy(self)
        candidate_edges = [e for e in list(graph.edges) if e not in self.traversed_edges]
        
        for perturbation in reversed(chain):
            for perturbation_type in edge_perturbation_types:
                if perturbation_type in perturbation.changes:
                    perturbed_edges_raw.extend(perturbation.changes[perturbation_type])
                    
        if len(perturbed_edges_raw) == 0:
            if len(candidate_edges) > 0:
                new_walk.traversed_edges.append(candidate_edges[0])
                return new_walk
            else:
                return None

        perturbed_edges = [(perturbed_edge['u'], perturbed_edge['v']) for perturbed_edge in perturbed_edges_raw]
        
        candidate_edges.sort(
            key = lambda candidate_edge:
                sum([self.distance_func(candidate_edge, perturbed_edge, self.graph) for perturbed_edge in perturbed_edges])
                / len(perturbed_edges)
        )
        
        if len(candidate_edges) > 0:
            new_walk.traversed_edges.append(candidate_edges[0])
            return new_walk
        else:
            return None

    def current(self) -> tuple[Any, Any]:
        return self.traversed_edges[-1]

class CentralityNodePerturbationWalk(PerturbationWalk[Any]):
    def __init__(self, graph: nx.Graph[Any]):
        self.traversed_nodes: list[Any] = []
        self.initial_centrality = nx.betweenness_centrality(graph)

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationWalk[Any] | None:
        centrality = nx.edge_betweenness_centrality(graph)
        candidate_nodes: list[tuple[Any, float]] = []

        for node in centrality:
            if node in self.initial_centrality and node not in self.traversed_nodes:
                candidate_nodes.append((node, (centrality[node] - self.initial_centrality[node])))

        candidate_nodes.sort(key = lambda x: x[1], reverse = True)

        if len(candidate_nodes) > 0:
            new_walk = deepcopy(self)
            new_walk.traversed_nodes.append(candidate_nodes[0])
            return new_walk
        else:
            return None

    def current(self) -> Any:
        return self.traversed_nodes[-1]

class CentralityEdgePerturbationWalk(PerturbationWalk[tuple[Any, Any]]):
    def __init__(self, graph: nx.Graph[Any]):
        self.traversed_edges: list[tuple[Any, Any]] = []
        self.initial_centrality = nx.edge_betweenness_centrality(graph)

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationWalk[tuple[Any, Any]] | None:
        centrality = nx.edge_betweenness_centrality(graph)
        candidate_edges: list[tuple[tuple[Any, Any], float]] = []

        for edge in centrality:
            if edge in self.initial_centrality and edge not in self.traversed_edges:
                candidate_edges.append((edge, (centrality[edge] - self.initial_centrality[edge])))

        candidate_edges.sort(key = lambda x: x[1], reverse = True)

        if len(candidate_edges) > 0:
            new_walk = deepcopy(self)
            new_walk.traversed_edges.append(candidate_edges[0][0])
            return new_walk
        else:
            return None

    def current(self) -> tuple[Any, Any]:
        return self.traversed_edges[-1]