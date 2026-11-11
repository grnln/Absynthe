from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from networkx.algorithms import isomorphism as iso

from interfaces import GraphLike
from utils.kg_utils import EDGE_RELATION_ATTR, NODE_TYPE_ATTR, edge_key, iter_edges


def _relation_counter(attrs: Dict[str, Any], multi: bool) -> Counter:
    """Count relation types of one host/motif edge slot.

    On multigraphs the isomorphism matcher passes a dict of ``{key: attrs}``
    for all parallel edges between two nodes; on simple graphs it passes the
    attrs dict itself. Edges without a ``relation`` count as ``None``.
    """
    dicts = attrs.values() if multi else [attrs]
    return Counter(
        str(d[EDGE_RELATION_ATTR]) if EDGE_RELATION_ATTR in d else None
        for d in dicts
    )


def make_relation_edge_match(host_multi: bool, motif_multi: bool) -> Callable[[Dict, Dict], bool]:
    """Edge matcher with containment semantics for relation-typed motifs.

    Every relation required by the motif must be present between the host
    nodes (with at least the same multiplicity); motif edges without a
    ``relation`` are wildcards that match any remaining host edge.
    """
    def match(host_attrs: Dict, motif_attrs: Dict) -> bool:
        hc = _relation_counter(host_attrs, host_multi)
        mc = _relation_counter(motif_attrs, motif_multi)
        wildcards = mc.pop(None, 0)
        if any(hc.get(rel, 0) < count for rel, count in mc.items()):
            return False
        remaining = sum(hc.values()) - sum(mc.values())
        return remaining >= wildcards
    return match


def _type_node_match(host_attrs: Dict, motif_attrs: Dict) -> bool:
    """Motif nodes with a ``type`` only match host nodes of the same type."""
    if NODE_TYPE_ATTR not in motif_attrs:
        return True
    return host_attrs.get(NODE_TYPE_ATTR) == motif_attrs[NODE_TYPE_ATTR]

from typing import Any

def assign_labels_to_motif(
    graph: GraphLike, motif: GraphLike, motif_name: str,
) -> List[Dict[str, Any]]:
    """Detect monomorphic subgraphs matching the motif and label nodes.

    Additional edges not present in the motif are allowed (subgraph
    monomorphism, not induced isomorphism).

    The host graph may be any NetworkX graph class:

    * An undirected motif is matched structurally on a directed host by
      ignoring edge direction (the undirected view of the host).
    * A directed motif is matched with direction on a directed host.
    * If motif nodes carry a ``type`` attribute, only host nodes with the
      same type match; if motif edges carry a ``relation`` attribute, the
      host edges must contain those relations (see
      :func:`make_relation_edge_match`). Motifs without those attributes
      match purely structurally, as before.

    Args:
        graph: Reference graph (modified in place — labels are set).
        motif: Structural motif to search for.
        motif_name: Label applied to detected nodes.

    Returns:
        A list of dicts, one per unique detected occurrence.  Each dict
        contains ``motif_name``, ``nodes`` (frozenset of node IDs) and
        ``edges`` (frozenset of edge references: ``(u, v)`` tuples, or
        ``(u, v, key)`` on multigraphs).
    """
    plain_host = not graph.is_directed() and not graph.is_multigraph()

    host = graph
    if graph.is_directed() and not motif.is_directed():
        host = graph.to_undirected(as_view=True)
    elif motif.is_directed() and not graph.is_directed():
        raise ValueError("A directed motif cannot be matched against an undirected graph")

    type_aware = any(NODE_TYPE_ATTR in d for _, d in motif.nodes(data=True))
    relation_aware = any(EDGE_RELATION_ATTR in d for _, _, d in motif.edges(data=True))

    node_match: Optional[Callable] = _type_node_match if type_aware else None
    edge_match: Optional[Callable] = (
        make_relation_edge_match(host.is_multigraph(), motif.is_multigraph())
        if relation_aware else None
    )

    Matcher = iso.DiGraphMatcher if host.is_directed() else iso.GraphMatcher
    matcher = Matcher(host, motif, node_match=node_match, edge_match=edge_match)

    seen_node_sets = set()
    instances: List[Dict[str, Any]] = []

    for subgraph in matcher.subgraph_monomorphisms_iter():
        node_set = frozenset(subgraph.keys())

        # Avoid labeling the same group multiple times (automorphisms)
        if node_set in seen_node_sets:
            continue
        seen_node_sets.add(node_set)

        for node in node_set:
            graph.nodes[node]['label'] = motif_name

        if plain_host:
            instance_edges = frozenset(graph.subgraph(node_set).edges())
        else:
            instance_edges = frozenset(
                edge_key(graph, u, v, k)
                for u, v, k, _ in iter_edges(graph.subgraph(node_set))
            )
        instances.append({
            "motif_name": motif_name,
            "nodes": node_set,
            "edges": instance_edges,
        })

    return instances
