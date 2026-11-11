import random
import networkx as nx

from typing import Optional, Dict, Any, Tuple, List
from interfaces import Perturbation, PerturbationHint, AppliedPerturbation, GraphLike

from .perturbation_strategies import STRATEGY_MAP
from utils.kg_utils import (
    EDGE_RELATION_ATTR,
    add_edge,
    edge_record,
    incident_edges,
    is_kg,
    iter_edges,
    non_edges,
    relation_vocab,
    remove_edge,
)
from utils.rng import get_rng

from interfaces import PerturbationChain, PerturbationHeuristic

# ---------------------------------------------------------------------------
# Hint helpers
# ---------------------------------------------------------------------------

def _edge_in_zone(u: Any, v: Any, hint: Optional[PerturbationHint],
                  key: Optional[Any] = None, directed: bool = False) -> bool:
    """An edge belongs to the hint zone if at least one endpoint is in
    ``hint.nodes`` or the edge itself is in ``hint.edges``. With no hint
    (or an empty one) every edge qualifies.
    """
    if hint is None or hint.is_empty():
        return True
    if u in hint.nodes or v in hint.nodes:
        return True
    return hint.contains_edge(u, v, key, directed=directed)


def _relation_allowed(attrs: Dict[str, Any], relations: Optional[List[str]]) -> bool:
    if not relations:
        return True
    return str(attrs.get(EDGE_RELATION_ATTR)) in relations


# ---------------------------------------------------------------------------
# Utility functions (used internally by the perturbation classes below)
# ---------------------------------------------------------------------------

def remove_nodes(graph: GraphLike,
                 n: int,
                 strategy: str = 'motif',
                 params: Optional[Dict[Any, Any]] = None,
                 rng: Optional[random.Random] = None,
                 hint: Optional[PerturbationHint] = None) -> Tuple[GraphLike, List[Any]]:
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


def perturb_edges(graph: GraphLike,
                  p_remove: float = 0.0,
                  p_add: float = 0.0,
                  add_num: Optional[int] = None,
                  rng: Optional[random.Random] = None,
                  hint: Optional[PerturbationHint] = None,
                  relations: Optional[List[str]] = None,
                  add_relation: Optional[str] = None,
                  ) -> Tuple[GraphLike, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Perturb edges by removing existing edges and/or adding new ones.

    Works on any NetworkX graph class. Direction is honoured on directed
    graphs and parallel edges are handled individually on multigraphs.

    Args:
        graph: The input graph.
        p_remove: Probability to remove each existing edge.
        p_add: Probability for each non-edge to be added (used if add_num is None).
        add_num: Explicit number of edges to add (if provided, p_add is ignored).
        rng: Random number generator.
        hint: Optional perturbation hint. When provided and non-empty, only
            edges/non-edges with at least one endpoint in ``hint.nodes``
            (or an existing edge listed in ``hint.edges``) are considered.
        relations: Optional list of relation types; when given, only edges
            whose ``relation`` attribute is in the list can be removed.
        add_relation: Relation type assigned to added edges on knowledge
            graphs. Defaults to a random relation from the graph's vocabulary.

    Returns:
        ``(perturbed_graph, removed_records, added_records)`` where the
        records are the reversible change dicts ``{"u", "v"[, "key"], "attrs"}``.
    """
    rng = rng or get_rng()
    G = graph.copy()
    directed = G.is_directed()
    removed: List[Dict[str, Any]] = []
    added: List[Dict[str, Any]] = []

    if p_remove > 0:
        for u, v, k, d in list(iter_edges(G)):
            if not _edge_in_zone(u, v, hint, k, directed):
                continue
            if not _relation_allowed(d, relations):
                continue
            if rng.random() < p_remove:
                removed.append(edge_record(G, u, v, k))
                remove_edge(G, u, v, k)

    candidates = [(u, v) for u, v in non_edges(G) if _edge_in_zone(u, v, hint, None, directed)]
    chosen: List[Tuple[Any, Any]] = []
    if add_num is not None:
        add_num = min(add_num, len(candidates))
        chosen = rng.sample(candidates, add_num) if add_num > 0 else []
    elif p_add > 0:
        chosen = [(u, v) for u, v in candidates if rng.random() < p_add]

    if chosen:
        vocab = relation_vocab(G) if (is_kg(G) and add_relation is None) else []
        for u, v in chosen:
            attrs: Dict[str, Any] = {}
            if add_relation is not None:
                attrs[EDGE_RELATION_ATTR] = add_relation
            elif vocab:
                attrs[EDGE_RELATION_ATTR] = rng.choice(vocab)
            key = add_edge(G, u, v, **attrs)
            added.append(edge_record(G, u, v, key, attrs=attrs, with_attrs=bool(attrs)))

    return G, removed, added


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
            incident = [
                edge_record(graph, u, v, k)
                for u, v, k, _ in incident_edges(graph, n)
            ]
            removed_info.append({
                "id": n,
                "attrs": dict(graph.nodes[n]),
                "edges": incident,
            })
        return removed_info

    def apply(self, graph: GraphLike,
              hint: Optional[PerturbationHint] = None
              ) -> Tuple[GraphLike, Dict[str, Any]]:
        new_graph, removed = remove_nodes(
            graph, self.num_nodes, self.strategy,
            self.params, self.rng, hint
        )
        removed_info = self._get_removed_info(graph, removed)
        return new_graph, {"removed_nodes": removed_info}

    def apply_heuristic(self, graph: nx.Graph[Any], heuristic: PerturbationHeuristic[Any]) -> AppliedPerturbation:
        nodes = list[Any]()

        for _ in range(self.num_nodes):
            node = heuristic.current()
            nodes.append(node)

        working_graph = graph.copy()
        working_graph.remove_nodes_from(nodes)

        return AppliedPerturbation(
            self,
            heuristic,
            working_graph,
            {"removed_nodes": self._get_removed_info(graph, nodes)}
        )

class RemoveEdgesPerturbation(Perturbation):
    """Remove edges from a graph with a given probability.

    On knowledge graphs, ``relations`` restricts removal to the listed
    relation types (e.g. ``["works_at"]``).
    """

    folder_name = "remove_edges"

    def __init__(self, p_remove: float = 0.1,
                rng: Optional[random.Random] = None,
                folder_name: Optional[str] = None,
                relations: Optional[List[str]] = None,
                num_edges: int = 2):
        self.p_remove = p_remove
        self.rng = rng
        self.num_edges = num_edges
        self.relations = relations

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

    def apply(self, graph: GraphLike,
              hint: Optional[PerturbationHint] = None
              ) -> Tuple[GraphLike, Dict[str, Any]]:
        new_graph, removed, _ = perturb_edges(
            graph, p_remove=self.p_remove, rng=self.rng, hint=hint,
            relations=self.relations,
        )
        return new_graph, {"removed_edges": removed}

    def apply_heuristic(self, graph: nx.Graph[Any], heuristic: PerturbationHeuristic[tuple[Any, Any]]) -> AppliedPerturbation:
        edges = set[tuple[Any, Any]]()

        for _ in range(self.num_edges):            
            edge = heuristic.current()
            edges.add(edge)

        working_graph = graph.copy()

        for u, v in edges:
            working_graph.remove_edge(u, v)
        
        return AppliedPerturbation(
            self,
            heuristic,
            working_graph,
            {"removed_edges": self._get_removed_info(graph, edges)}
        )

class AddEdgesPerturbation(Perturbation):
    """Add random edges to a graph.

    On knowledge graphs, added edges receive ``add_relation`` (or a random
    relation from the graph's vocabulary when not given).
    """

    folder_name = "add_edges"

    def __init__(self, p_add: float = 0.0, add_num: Optional[int] = None,
                 rng: Optional[random.Random] = None,
                 folder_name: Optional[str] = None,
                 add_relation: Optional[str] = None):
        self.p_add = p_add
        self.add_num = add_num
        self.rng = rng
        self.add_relation = add_relation
        if folder_name is not None:
            self.folder_name = folder_name

    def _get_added_info(self, graph: nx.Graph[Any], new_graph: nx.Graph[Any]) -> list[dict[str, Any]]:
        added = set(new_graph.edges()) - set(graph.edges())
        added_info = [{"u": u, "v": v} for u, v in added]
        return added_info

    def apply(self, graph: GraphLike,
        hint: Optional[PerturbationHint] = None
    ) -> Tuple[GraphLike, Dict[str, Any]]:
        new_graph, _, added = perturb_edges(
            graph, p_add=self.p_add, add_num=self.add_num,
            rng=self.rng, hint=hint, add_relation=self.add_relation,
        )
        return new_graph, {"added_edges": added}

    def apply_heuristic(self, graph: nx.Graph[Any], heuristic: PerturbationHeuristic[tuple[Any, Any]]) -> AppliedPerturbation:
        edges = set[tuple[Any, Any]]()

        if self.add_num is None:
            raise ValueError("The number of edges to add must be provided to apply the perturbation based on a heuristic.")

        for _ in range(self.add_num):
            edge = heuristic.current()
            edges.add(edge)

        working_graph = graph.copy()

        for u, v in edges:
            working_graph.add_edge(u, v)

        return AppliedPerturbation(
            self,
            heuristic,
            working_graph,
            {"added_edges": self._get_added_info(graph, working_graph)}
        )

class EdgePerturbation(Perturbation):
    """Combined edge perturbation: remove and add edges simultaneously."""

    folder_name = "edge_perturbation"

    def __init__(self, p_remove: float = 0.0, p_add: float = 0.0,
                 add_num: Optional[int] = None,
                 rng: Optional[random.Random] = None,
                 folder_name: Optional[str] = None,
                 relations: Optional[List[str]] = None,
                 add_relation: Optional[str] = None):
        self.p_remove = p_remove
        self.p_add = p_add
        self.add_num = add_num
        self.rng = rng
        self.relations = relations
        self.add_relation = add_relation
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

    def apply(self, graph: GraphLike,
              hint: Optional[PerturbationHint] = None
              ) -> Tuple[GraphLike, Dict[str, Any]]:
        new_graph, removed, added = perturb_edges(
            graph, p_remove=self.p_remove, p_add=self.p_add,
            add_num=self.add_num, rng=self.rng, hint=hint,
            relations=self.relations, add_relation=self.add_relation,
        )
        return new_graph, {
            "removed_edges": removed,
            "added_edges": added,
        }

    def apply_heuristic(self, graph: nx.Graph[Any], heuristic: PerturbationHeuristic[tuple[Any, Any]]) -> AppliedPerturbation:
            edges = set[tuple[tuple[Any, Any], tuple[Any, Any]]]()
    
            if self.add_num is None:
                raise ValueError("The number of edges must be provided to apply the perturbation based on a heuristic.")
    
            for _ in range(self.add_num):
                edge = heuristic.current()
                edges.add(edge)

            working_graph = graph.copy()
    
            for add, remove in edges:
                for u, v in add:
                    working_graph.add_edge(u, v)

                for u, v in remove:
                    working_graph.add_edge(u, v)

            removed_edges, added_edges = self._get_change_info(graph, working_graph)
            return AppliedPerturbation(
                self,
                heuristic,
                working_graph,
                {
                    "removed_edges": removed_edges,
                    "added_edges": added_edges
                }
            )
