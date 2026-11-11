from __future__ import annotations

import random, graphviz, os
import networkx as nx

from math import sqrt
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from .mcts_tree import MCTSTree
from typing import Any, Callable
from interfaces import LabelingFunction
from interfaces import Perturbation, PerturbationHeuristic, PerturbationChain, AppliedPerturbation

class NaturalOrderNodePerturbationHeuristic(PerturbationHeuristic[Any]):
    def __init__(self) -> None:
        self.traversed_nodes: list[Any] = []

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationHeuristic[Any] | None:
        candidate_nodes = sorted([n for n in list(graph.nodes) if n not in self.traversed_nodes])

        if len(candidate_nodes) > 0:
            new_heuristic = deepcopy(self)
            new_heuristic.traversed_nodes.append(candidate_nodes[0])
            return new_heuristic
        else:
            return None
 
    def current(self) -> Any:
        return self.traversed_nodes[-1]

    def reset(self) -> None:
        self.traversed_nodes.clear()

    def soft_reset(self) -> None:
        if len(self.traversed_nodes) > 1:
            node = self.traversed_nodes[-1]

            self.traversed_nodes.clear()
            self.traversed_nodes.append(node)
    
    def add(self, element: Any) -> None:
        self.traversed_nodes.append(element)

class NaturalOrderEdgePerturbationHeuristic(PerturbationHeuristic[tuple[Any, Any]]):
    def __init__(self) -> None:
        self.traversed_edges: list[tuple[Any, Any]] = []

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationHeuristic[tuple[Any, Any]] | None:
        candidate_edges = sorted([e for e in list(graph.edges) if e not in self.traversed_edges], key = lambda x: x[0])

        if len(candidate_edges) > 0:
            new_heuristic = deepcopy(self)
            new_heuristic.traversed_edges.append(candidate_edges[0])
            return new_heuristic
        else:
            return None
            
    def current(self) -> tuple[Any, Any]:
        return self.traversed_edges[-1]

    def reset(self) -> None:
        self.traversed_edges.clear()

    def soft_reset(self) -> None:
        if len(self.traversed_edges) > 1:
            edge = self.traversed_edges[-1]
            
            self.traversed_edges.clear()
            self.traversed_edges.append(edge)

    def add(self, element: tuple[Any, Any]) -> None:
        self.traversed_edges.append(element)

node_perturbation_types: set[str] = {
    'removed_nodes'
}

def node_graph_distance(a: Any, b: Any, graph: nx.Graph[Any]) -> float:
    return nx.shortest_path_length(graph, source = a, target = b) # type: ignore

def node_euclidean_distance(a: Any, b: Any, graph: nx.Graph[Any]) -> float:
    return abs(float(b) - float(a))

class ProximityNodePerturbationHeuristic(PerturbationHeuristic[Any]):
    def __init__(self, graph: nx.Graph[Any], distance_func: Callable[[Any, Any, nx.Graph[Any]], float] = node_graph_distance) -> None:
        self.graph = graph.copy()
        self.traversed_nodes: list[Any] = []
        self.distance_func = distance_func

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationHeuristic[Any] | None:
        perturbed_nodes_raw: list[dict[str, Any]] = []
        
        new_heuristic = deepcopy(self)
        candidate_nodes = [n for n in list(graph.nodes) if n not in self.traversed_nodes]
        
        for perturbation in reversed(chain):
            for perturbation_type in node_perturbation_types:
                if perturbation_type in perturbation.changes:
                    perturbed_nodes_raw.extend(perturbation.changes[perturbation_type])

        if len(perturbed_nodes_raw) == 0:
            if len(candidate_nodes) > 0:
                new_heuristic.traversed_nodes.append(candidate_nodes[0])
                return new_heuristic
            else:
                return None

        perturbed_nodes = [perturbed_node['id'] for perturbed_node in perturbed_nodes_raw]
        
        candidate_nodes.sort(
            key = lambda candidate_node:
                sum([self.distance_func(candidate_node, perturbed_node, self.graph) for perturbed_node in perturbed_nodes])
                / len(perturbed_nodes)
        )
        
        if len(candidate_nodes) > 0:
            new_heuristic.traversed_nodes.append(candidate_nodes[0])
            return new_heuristic
        else:
            return None

    def current(self) -> tuple[Any, Any]:
        return self.traversed_nodes[-1]

    def reset(self) -> None:
        self.traversed_nodes.clear()

    def soft_reset(self) -> None:
        if len(self.traversed_nodes) > 1:
            node = self.traversed_nodes[-1]
            
            self.traversed_nodes.clear()
            self.traversed_nodes.append(node)

    def add(self, element: Any) -> None:
        self.traversed_nodes.append(element)

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

class ProximityEdgePerturbationHeuristic(PerturbationHeuristic[tuple[Any, Any]]):
    def __init__(self, graph: nx.Graph[Any], distance_func: Callable[[tuple[Any, Any], tuple[Any, Any], nx.Graph[Any]], float] = edge_graph_distance) -> None:
        self.graph = graph.copy()
        self.traversed_edges: list[tuple[Any, Any]] = []
        self.distance_func = distance_func

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationHeuristic[tuple[Any, Any]] | None:
        perturbed_edges_raw: list[dict[str, Any]] = []

        new_heuristic = deepcopy(self)
        candidate_edges = [e for e in list(graph.edges) if e not in self.traversed_edges]
        
        for perturbation in reversed(chain):
            for perturbation_type in edge_perturbation_types:
                if perturbation_type in perturbation.changes:
                    perturbed_edges_raw.extend(perturbation.changes[perturbation_type])
                    
        if len(perturbed_edges_raw) == 0:
            if len(candidate_edges) > 0:
                new_heuristic.traversed_edges.append(candidate_edges[0])
                return new_heuristic
            else:
                return None

        perturbed_edges = [(perturbed_edge['u'], perturbed_edge['v']) for perturbed_edge in perturbed_edges_raw]
        
        candidate_edges.sort(
            key = lambda candidate_edge:
                sum([self.distance_func(candidate_edge, perturbed_edge, self.graph) for perturbed_edge in perturbed_edges])
                / len(perturbed_edges)
        )
        
        if len(candidate_edges) > 0:
            new_heuristic.traversed_edges.append(candidate_edges[0])
            return new_heuristic
        else:
            return None

    def current(self) -> tuple[Any, Any]:
        return self.traversed_edges[-1]

    def reset(self) -> None:
        self.traversed_edges.clear()


    def soft_reset(self) -> None:
        if len(self.traversed_edges) > 1:
            edge = self.traversed_edges[-1]
            
            self.traversed_edges.clear()
            self.traversed_edges.append(edge)

    def add(self, element: tuple[Any, Any]) -> None:
        self.traversed_edges.append(element)

class CentralityNodePerturbationHeuristic(PerturbationHeuristic[Any]):
    def __init__(self, graph: nx.Graph[Any]):
        self.traversed_nodes: list[Any] = []
        self.initial_centrality = nx.betweenness_centrality(graph)

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationHeuristic[Any] | None:
        centrality = nx.edge_betweenness_centrality(graph)
        candidate_nodes: list[tuple[Any, float]] = []

        for node in centrality:
            if node in self.initial_centrality and node not in self.traversed_nodes:
                candidate_nodes.append((node, (centrality[node] - self.initial_centrality[node])))

        candidate_nodes.sort(key = lambda x: x[1], reverse = True)

        if len(candidate_nodes) > 0:
            new_heuristic = deepcopy(self)
            new_heuristic.traversed_nodes.append(candidate_nodes[0])
            return new_heuristic
        else:
            return None

    def current(self) -> Any:
        return self.traversed_nodes[-1]

    def reset(self) -> None:
        self.traversed_nodes.clear()

    def soft_reset(self) -> None:
        if len(self.traversed_nodes) > 1:
            node = self.traversed_nodes[-1]
            
            self.traversed_nodes.clear()
            self.traversed_nodes.append(node)

    def add(self, element: Any) -> None:
        self.traversed_nodes.append(element)

class CentralityEdgePerturbationHeuristic(PerturbationHeuristic[tuple[Any, Any]]):
    def __init__(self, graph: nx.Graph[Any]):
        self.traversed_edges: list[tuple[Any, Any]] = []
        self.initial_centrality = nx.edge_betweenness_centrality(graph)

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationHeuristic[tuple[Any, Any]] | None:
        centrality = nx.edge_betweenness_centrality(graph)
        candidate_edges: list[tuple[tuple[Any, Any], float]] = []

        for edge in centrality:
            if edge in self.initial_centrality and edge not in self.traversed_edges:
                candidate_edges.append((edge, (centrality[edge] - self.initial_centrality[edge])))

        candidate_edges.sort(key = lambda x: x[1], reverse = True)

        if len(candidate_edges) > 0:
            new_heuristic = deepcopy(self)
            new_heuristic.traversed_edges.append(candidate_edges[0][0])
            return new_heuristic
        else:
            return None

    def current(self) -> tuple[Any, Any]:
        return self.traversed_edges[-1]

    def reset(self) -> None:
        self.traversed_edges.clear()

    def soft_reset(self) -> None:
        if len(self.traversed_edges) > 1:
            edge = self.traversed_edges[-1]
            
            self.traversed_edges.clear()
            self.traversed_edges.append(edge)

    def add(self, element: tuple[Any, Any]) -> None:
        self.traversed_edges.append(element)

class ConnectedComponentsNodePerturbationHeuristic(PerturbationHeuristic[Any]):
    def __init__(self):
        self.traversed_nodes: list[Any] = []

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationHeuristic[Any] | None:
        last_perturbed_node = None
        last_perturbed_graph = graph
        candidate_nodes = [n for n in list(graph.nodes) if n not in self.traversed_nodes]
        new_heuristic = deepcopy(self)
        
        for perturbation in reversed(chain):
            for perturbation_type in edge_perturbation_types:
                if perturbation_type in perturbation.changes:
                    node_dict = perturbation.changes[perturbation_type]
                    last_perturbed_node = node_dict[-1]['id']
                    last_perturbed_graph = perturbation.graph
                    break

        if last_perturbed_node is None:
            if len(candidate_nodes) > 0:
                new_heuristic.traversed_nodes.append(candidate_nodes[0])
                return new_heuristic
            else:
                return None

        component = nx.node_connected_component(last_perturbed_graph, last_perturbed_node)
        candidate_nodes = [n for n in candidate_nodes if n in component]

        if len(candidate_nodes) <= 0:
            return None

        node_distances = {n: node_graph_distance(last_perturbed_node, n, last_perturbed_graph) for n in candidate_nodes}
        new_heuristic.traversed_nodes.append(min(candidate_nodes, key = lambda n: node_distances[n]))
        return new_heuristic

    def current(self) -> Any:
        return self.traversed_nodes[-1]

    def reset(self) -> None:
        self.traversed_nodes.clear()

    def soft_reset(self) -> None:
        if len(self.traversed_nodes) > 1:
            node = self.traversed_nodes[-1]
            
            self.traversed_nodes.clear()
            self.traversed_nodes.append(node)

    def add(self, element: Any) -> None:
        self.traversed_nodes.append(element)

class ConnectedComponentsEdgePerturbationHeuristic(PerturbationHeuristic[tuple[Any, Any]]):
    def __init__(self):
        self.traversed_edges: list[tuple[Any, Any]] = []

    def _weigh_edge(self, edge: tuple[Any, Any], component: set[Any]) -> int:
        u, v = edge
        weight = 0

        if u in component:
            weight += 1
        
        if v in component:
            weight += 1

        return weight

    def _try_connected_component(self, graph: nx.Graph[Any], node: Any, edge_distances: dict[tuple[Any, Any], float]) -> tuple[tuple[Any, Any], int] | None:
        component = nx.node_connected_component(graph, node)
        edge_weights = {edge: self._weigh_edge(edge, component) for edge in edge_distances}

        if all([weight <= 0 for weight in edge_weights.values()]):
            return None
        
        edge = min(edge_distances.keys(), key = lambda e: (-edge_weights[e], edge_distances[e]))
        return edge, edge_weights[edge]

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationHeuristic[tuple[Any, Any]] | None:
        last_perturbed_edge = None
        last_perturbed_graph = graph
        candidate_edges = [e for e in list(graph.edges) if e not in self.traversed_edges]
        new_heuristic = deepcopy(self)
        
        for perturbation in reversed(chain):
            for perturbation_type in edge_perturbation_types:
                if perturbation_type in perturbation.changes:
                    edge_dict = perturbation.changes[perturbation_type]
                    last_perturbed_edge = (edge_dict[-1]['u'], edge_dict[-1]['v'])
                    last_perturbed_graph = perturbation.graph
                    break

        if last_perturbed_edge is None:
            if len(candidate_edges) > 0:
                new_heuristic.traversed_edges.append(candidate_edges[0])
                return new_heuristic
            else:
                return None
        
        edge_distances = {e: edge_graph_distance(last_perturbed_edge, e, last_perturbed_graph) for e in candidate_edges}

        result_u = self._try_connected_component(last_perturbed_graph, last_perturbed_edge[0], edge_distances)
        result_v = self._try_connected_component(last_perturbed_graph, last_perturbed_edge[0], edge_distances)

        if result_u is not None:
            edge, weight = result_u

            if result_v is not None:
                edge_v, weight_v = result_v

                if weight_v > weight:
                    edge = edge_v
            new_heuristic.traversed_edges.append(edge)
        elif result_v is not None:
            edge, _ = result_v
            new_heuristic.traversed_edges.append(edge)
        else:
            if len(candidate_edges) > 0:
                new_heuristic.traversed_edges.append(candidate_edges[0])
            else:
                new_heuristic = None
        return new_heuristic

    def current(self) -> tuple[Any, Any]:
        return self.traversed_edges[-1]
    
    def reset(self) -> None:
        self.traversed_edges.clear()

    def soft_reset(self) -> None:
        if len(self.traversed_edges) > 1:
            edge = self.traversed_edges[-1]
            
            self.traversed_edges.clear()
            self.traversed_edges.append(edge)

    def add(self, element: tuple[Any, Any]) -> None:
        self.traversed_edges.append(element)

class RandomNodePerturbationHeuristic(PerturbationHeuristic[Any]):
    def __init__(self) -> None:
        self.traversed_nodes: list[Any] = []

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationHeuristic[Any] | None:
        candidate_nodes = sorted([n for n in list(graph.nodes) if n not in self.traversed_nodes])

        if len(candidate_nodes) > 0:
            new_heuristic = deepcopy(self)
            new_heuristic.traversed_nodes.append(random.choice(candidate_nodes))
            return new_heuristic
        else:
            return None
 
    def current(self) -> Any:
        return self.traversed_nodes[-1]

    def reset(self) -> None:
        self.traversed_nodes.clear()

    def soft_reset(self) -> None:
        if len(self.traversed_nodes) > 1:
            node = self.traversed_nodes[-1]

            self.traversed_nodes.clear()
            self.traversed_nodes.append(node)

    def add(self, element: Any) -> None:
        self.traversed_nodes.append(element)

class RandomEdgePerturbationHeuristic(PerturbationHeuristic[tuple[Any, Any]]):
    def __init__(self) -> None:
        self.traversed_edges: list[tuple[Any, Any]] = []

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> PerturbationHeuristic[tuple[Any, Any]] | None:
        candidate_edges = sorted([e for e in list(graph.edges) if e not in self.traversed_edges], key = lambda x: x[0])

        if len(candidate_edges) > 0:
            new_heuristic = deepcopy(self)
            new_heuristic.traversed_edges.append(random.choice(candidate_edges))
            return new_heuristic
        else:
            return None
            
    def current(self) -> tuple[Any, Any]:
        return self.traversed_edges[-1]

    def reset(self) -> None:
        self.traversed_edges.clear()

    def soft_reset(self) -> None:
        if len(self.traversed_edges) > 1:
            edge = self.traversed_edges[-1]
            
            self.traversed_edges.clear()
            self.traversed_edges.append(edge)

    def add(self, element: tuple[Any, Any]) -> None:
        self.traversed_edges.append(element)

class MCTSPerturbationHeuristic(PerturbationHeuristic[AppliedPerturbation]):
    def __init__(
        self,
        graph: nx.Graph[Any],
        labeling_functions: list[LabelingFunction],
        expansion_heuristics: dict[Perturbation, PerturbationHeuristic[Any]],
        chain_type: type[PerturbationChain],
        max_length: int = 4,
        max_training_iterations: int = 100,
        mcts_exploration_param: float = sqrt(2.0),
        visualize_tree: bool = False,
        ordered: bool = False
    ) -> None:
        self.graph = graph
        self.chain_type = chain_type
        self.labeling_functions = labeling_functions

        self.max_length = max_length
        self.max_training_iterations = max_training_iterations

        self.tree = MCTSTree(
            None,
            self.chain_type([]),
            self.chain_type,
            expansion_heuristics,
            graph,
            mcts_exploration_param
        )
        self.ordered = ordered
        self.chains = self._train_tree(self.tree)
        self.traversed_nodes: dict[int, list[MCTSTree]] = {}

        print(f'Found {len(self.chains)} chains during training{":" if len(self.chains) > 0 else "."}')
        print(self.chains)

        if visualize_tree:
            self._visualize_tree(self.tree)

    def get_found_chains(self) -> set[PerturbationChain]:
        return self.chains

    def _train_tree(self, tree: MCTSTree) -> set[PerturbationChain]:
        chains: set[PerturbationChain] = set()
        
        for _ in range(self.max_training_iterations):
            node = tree
            
            while not node.is_leaf() and node.is_expanded():
                node = node.best_child()
        
            if not node.is_leaf() and not node.is_expanded():
                node.expand()

            if len(node.children) > 0:
                node = node.children[-1]

            valid_chain = node.simulate(
                self.graph,
                self.labeling_functions,
                random.randint(2, self.max_length),
                chains,
                self.ordered
            )
            node.backpropagate(valid_chain)
        return chains

    def _visualize_tree(self, tree: MCTSTree) -> None:
        output_dir = Path(os.getcwd()) / Path('visualizations')
        
        if not os.path.isdir(output_dir):
            print('Creating visualizations directory...')
            os.makedirs(output_dir, exist_ok = True)
        else:
            print('Clearing existing visualizations directory...')
            
            for _, _, filenames in os.walk(output_dir):
                for filename in filenames:
                    os.remove(output_dir / Path(filename))
        
        dot = graphviz.Graph(f'MCTS_{datetime.now().strftime("%Y%m%d_%H%M%S")}', format = 'png')
        tree.visualize(dot)
        dot.render(directory = 'visualizations')

    def next(self, graph: nx.Graph[Any], chain: PerturbationChain) -> MCTSPerturbationHeuristic | None:
        node = self.tree

        for perturbation in chain:
            for child in node.children:
                if perturbation == child.chain[-1]:
                    node = child

        candidate_nodes = node.children

        if len(chain) in self.traversed_nodes:
            candidate_nodes = [child for child in candidate_nodes if child not in self.traversed_nodes[len(chain)]]

        if len(candidate_nodes) < 1:
            return None

        new_heuristic = deepcopy(self)
        candidate_nodes.sort(key = lambda c: c.num_simulations, reverse = True)
        selected_node = candidate_nodes[0]

        if len(chain) in self.traversed_nodes:
            new_heuristic.traversed_nodes[len(chain)].append(selected_node)
        else:
            new_heuristic.traversed_nodes[len(chain)] = [selected_node]
        return new_heuristic

    def current(self) -> AppliedPerturbation:
        return self.traversed_nodes[max(self.traversed_nodes.keys())][-1].chain[-1]

    def reset(self) -> None:
        self.traversed_nodes = {}

    def soft_reset(self) -> None:
        items = list(self.traversed_nodes.items())
        
        if len(items) > 1:
            items.sort(key = lambda i: i[0])
            last_item = (items[-1][0], items[-1][1][-1])
            self.traversed_nodes = {last_item[0]: [last_item[1]]}

    def add(self, element: AppliedPerturbation) -> None:
        raise NotImplementedError('Method is not implemented for this class.')