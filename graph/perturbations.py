import random
from typing import Optional, Dict, Any, Tuple, List
import networkx as nx
from interfaces import Perturbation, PerturbationHint
from .perturbation_strategies import STRATEGY_MAP
from utils.rng import get_rng

from interfaces import PerturbationChain, PerturbationWalk

# ---------------------------------------------------------------------------
# Hint helpers
# ---------------------------------------------------------------------------

def _edge_in_zone(u: int, v: int, hint: Optional[PerturbationHint]) -> bool:
    """An edge belongs to the hint zone if at least one endpoint is in
    ``hint.nodes`` or the edge itself is in ``hint.edges``. With no hint
    (or an empty one) every edge qualifies.
    """
    if hint is None or hint.is_empty():
        return True
    if u in hint.nodes or v in hint.nodes:
        return True
    return PerturbationHint.normalize_edge(u, v) in hint.edges


# ---------------------------------------------------------------------------
# Utility functions (used internally by the perturbation classes below)
# ---------------------------------------------------------------------------

def remove_nodes(graph: nx.Graph[Any],
                 n: int,
                 strategy: str = 'motif',
                 params: Optional[Dict[Any, Any]] = None,
                 rng: Optional[random.Random] = None,
                 hint: Optional[PerturbationHint] = None) -> Tuple[nx.Graph[Any], List[int]]:
    """Remove n nodes according to the requested strategy and return (new_graph, removed_nodes).

    Args:
        graph: The input graph.
        n: Number of nodes to remove.
        strategy: Strategy for selecting nodes to remove.
        params: Additional parameters for the strategy.
        rng: Random number generator.
        hint: Optional perturbation hint. When provided and non-empty, the
            candidate pool is restricted to ``hint.nodes``.

    Returns:
        Tuple of (perturbed_graph, list_of_removed_nodes).
    """
    params = params or {}
    rng = rng or get_rng()

    G = graph.copy()
    if hint is not None and hint.nodes:
        nodes = [u for u in G.nodes() if u in hint.nodes]
    else:
        nodes = list(G.nodes())
    n = min(n, len(nodes))

    if n == 0:
        return G, []

    strat = STRATEGY_MAP.get((strategy or 'random').lower())
    if not strat:
        raise ValueError(f"Unknown strategy: {strategy}")

    to_remove = strat(G.subgraph(nodes), n, params, rng)
    G.remove_nodes_from(to_remove)
    return G, to_remove

def perturb_edges(graph: nx.Graph[Any],
                  p_remove: float = 0.0,
                  p_add: float = 0.0,
                  add_num: Optional[int] = None,
                  rng: Optional[random.Random] = None,
                  hint: Optional[PerturbationHint] = None) -> nx.Graph[Any]:
    """Perturb edges by removing existing edges and/or adding new ones.

    Args:
        graph: The input graph.
        p_remove: Probability to remove each existing edge.
        p_add: Probability for each non-edge to be added (used if add_num is None).
        add_num: Explicit number of edges to add (if provided, p_add is ignored).
        rng: Random number generator.
        hint: Optional perturbation hint. When provided and non-empty, only
            edges/non-edges with at least one endpoint in ``hint.nodes``
            (or an existing edge listed in ``hint.edges``) are considered.
    """
    rng = rng or get_rng()
    G = graph.copy()

    if p_remove > 0:
        for u, v in list(G.edges()):
            if not _edge_in_zone(u, v, hint):
                continue
            if rng.random() < p_remove:
                G.remove_edge(u, v)

    non_edges = [(u, v) for u, v in nx.non_edges(G) if _edge_in_zone(u, v, hint)]
    if add_num is not None:
        add_num = min(add_num, len(non_edges))
        additions = rng.sample(non_edges, add_num) if add_num > 0 else []
        G.add_edges_from(additions)
    elif p_add > 0:
        for u, v in non_edges:
            if rng.random() < p_add:
                G.add_edge(u, v)

    return G


# ---------------------------------------------------------------------------
# Concrete Perturbation implementations
# ---------------------------------------------------------------------------

class RemoveNodesPerturbation(Perturbation):
    """Remove nodes from a graph using a configurable strategy."""

    folder_name = "remove_nodes"

    def __init__(self, num_nodes: int, strategy: str = 'random',
                 params: Optional[Dict[Any, Any]] = None,
                 rng: Optional[random.Random] = None,
                 folder_name: Optional[str] = None):
        self.num_nodes = num_nodes
        self.strategy = strategy
        self.params = params
        self.rng = rng
        if folder_name is not None:
            self.folder_name = folder_name

    def _get_removed_info(self, graph: nx.Graph[Any], removed: list[Any]) -> list[dict[str, Any]]:
        removed_info: list[dict[str, Any]] = []
        
        for n in removed:
            incident: list[dict[str, Any]] = [
                {
                    "u": n,
                    "v": nb,
                    "attrs": dict(graph.edges[n, nb])
                }
                for nb in graph.neighbors(n)
            ]
            removed_info.append({
                "id": n,
                "attrs": dict(graph.nodes[n]),
                "edges": incident,
            })
        return removed_info

    def apply(self, graph: nx.Graph[Any],
              hint: Optional[PerturbationHint] = None
              ) -> Tuple[nx.Graph[Any], Dict[str, Any]]:
        new_graph, removed = remove_nodes(
            graph, self.num_nodes, self.strategy,
            self.params, self.rng, hint
        )
        removed_info = self._get_removed_info(graph, removed)
        return new_graph, {"removed_nodes": removed_info}

    def apply_by_walk(self, graph: nx.Graph[Any], walk: PerturbationWalk[Any], chain: PerturbationChain) -> tuple[nx.Graph[Any], dict[str, Any]]:
        nodes = list[Any]()

        for _ in range(self.num_nodes):
            nodes.append(walk.next(chain))

        working_graph = graph.copy()
        working_graph.remove_nodes_from(nodes)
        return (working_graph, {"removed_nodes": self._get_removed_info(graph, nodes)})

class RemoveEdgesPerturbation(Perturbation):
    """Remove edges from a graph with a given probability."""

    folder_name = "remove_edges"

    def __init__(self, p_remove: float = 0.1,
                 rng: Optional[random.Random] = None,
                 folder_name: Optional[str] = None,
                 num_edges: int = 2):
        self.p_remove = p_remove
        self.rng = rng
        self.num_edges = num_edges

        if folder_name is not None:
            self.folder_name = folder_name

    def _get_removed_info(self, graph: nx.Graph[Any], removed: set[tuple[Any, Any]]) -> list[dict[str, Any]]:
        removed_info: list[dict[str, Any]] = [
            {
                "u": u,
                "v": v,
                "attrs": dict(graph.edges[u, v])
            }
            for u, v in removed
        ]
        return removed_info

    def apply(self, graph: nx.Graph[Any],
              hint: Optional[PerturbationHint] = None
              ) -> Tuple[nx.Graph[Any], Dict[str, Any]]:
        new_graph = perturb_edges(
            graph, p_remove=self.p_remove, rng=self.rng, hint=hint
        )
        removed = set(graph.edges()) - set(new_graph.edges())
        removed_info = self._get_removed_info(graph, removed)
        return new_graph, {"removed_edges": removed_info}

    def apply_by_walk(self, graph: nx.Graph[Any], walk: PerturbationWalk[tuple[Any, Any]], chain: PerturbationChain) -> tuple[nx.Graph[Any], dict[str, Any]]:
        edges = set[tuple[Any, Any]]()

        for _ in range(self.num_edges):
            edges.add(walk.next(chain))

        working_graph = graph.copy()

        for u, v in edges:
            working_graph.remove_edge(u, v)
        
        return (working_graph, {"removed_nodes": self._get_removed_info(graph, edges)})

class AddEdgesPerturbation(Perturbation):
    """Add random edges to a graph."""

    folder_name = "add_edges"

    def __init__(self, p_add: float = 0.0, add_num: Optional[int] = None,
                 rng: Optional[random.Random] = None,
                 folder_name: Optional[str] = None):
        self.p_add = p_add
        self.add_num = add_num
        self.rng = rng
        if folder_name is not None:
            self.folder_name = folder_name

    def _get_added_info(self, graph: nx.Graph[Any], new_graph: nx.Graph[Any]) -> list[dict[str, Any]]:
        added = set(new_graph.edges()) - set(graph.edges())
        added_info = [{"u": u, "v": v} for u, v in added]
        return added_info

    def apply(self, graph: nx.Graph[Any],
              hint: Optional[PerturbationHint] = None
              ) -> Tuple[nx.Graph[Any], Dict[str, Any]]:
        new_graph = perturb_edges(
            graph, p_add=self.p_add, add_num=self.add_num,
            rng=self.rng, hint=hint,
        )
        return new_graph, {"added_edges": self._get_added_info(graph, new_graph)}

    def apply_by_walk(self, graph: nx.Graph[Any], walk: PerturbationWalk[tuple[Any, Any]], chain: PerturbationChain) -> tuple[nx.Graph[Any], dict[str, Any]]:
        edges = set[tuple[Any, Any]]()

        if self.add_num is None:
            raise ValueError("The number of edges to add must be provided to apply the perturbation based on a walk.")

        for _ in range(self.add_num):
            edges.add(walk.next(chain))

        working_graph = graph.copy()

        for u, v in edges:
            working_graph.add_edge(u, v)
            
        return (working_graph, {"added_edges": self._get_added_info(graph, working_graph)})

class EdgePerturbation(Perturbation):
    """Combined edge perturbation: remove and add edges simultaneously."""

    folder_name = "edge_perturbation"

    def __init__(self, p_remove: float = 0.0, p_add: float = 0.0,
                 add_num: Optional[int] = None,
                 rng: Optional[random.Random] = None,
                 folder_name: Optional[str] = None):
        self.p_remove = p_remove
        self.p_add = p_add
        self.add_num = add_num
        self.rng = rng
        
        if folder_name is not None:
            self.folder_name = folder_name

    def _get_change_info(self, graph: nx.Graph[Any], new_graph: nx.Graph[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        before_edges = set(graph.edges())
        after_edges = set(new_graph.edges())
        
        removed_edges = before_edges - after_edges
        added_edges = after_edges - before_edges

        return [
            {"u": u, "v": v, "attrs": dict(graph.edges[u, v])}
            for u, v in removed_edges
        ], [
            {"u": u, "v": v} for u, v in added_edges
        ]

    def apply(self, graph: nx.Graph[Any],
              hint: Optional[PerturbationHint] = None
              ) -> Tuple[nx.Graph[Any], Dict[str, Any]]:
        new_graph = perturb_edges(
            graph, p_remove=self.p_remove, p_add=self.p_add,
            add_num=self.add_num, rng=self.rng, hint=hint,
        )
        removed_edges, added_edges = self._get_change_info(graph, new_graph)
        return new_graph, {
            "removed_edges": removed_edges,
            "added_edges": added_edges
        }

    def apply_by_walk(self, graph: nx.Graph[Any], walk: PerturbationWalk[tuple[Any, Any]], chain: PerturbationChain) -> tuple[nx.Graph[Any], dict[str, Any]]:
            edges = set[tuple[tuple[Any, Any], tuple[Any, Any]]]()
    
            if self.add_num is None:
                raise ValueError("The number of edges must be provided to apply the perturbation based on a walk.")
    
            for _ in range(self.add_num):
                edges.add(walk.next(chain))

            working_graph = graph.copy()
    
            for add, remove in edges:
                for u, v in add:
                    working_graph.add_edge(u, v)

                for u, v in remove:
                    working_graph.add_edge(u, v)

            removed_edges, added_edges = self._get_change_info(graph, working_graph)
            return working_graph, {
                "removed_edges": removed_edges,
                "added_edges": added_edges
            }
