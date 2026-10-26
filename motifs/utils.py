import networkx as nx

from typing import Any

def assign_labels_to_motif(
    graph: nx.Graph[Any], motif: nx.Graph[Any], motif_name: str,
) -> list[dict[str, Any]]:
    """Detect monomorphic subgraphs matching the motif and label nodes.

    Additional edges not present in the motif are allowed (subgraph
    monomorphism, not induced isomorphism).

    Args:
        graph: Reference graph (modified in place — labels are set).
        motif: Structural motif to search for.
        motif_name: Label applied to detected nodes.

    Returns:
        A list of dicts, one per unique detected occurrence.  Each dict
        contains ``motif_name``, ``nodes`` (frozenset of node IDs) and
        ``edges`` (frozenset of edge tuples).
    """
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph, motif,
        node_match=None,
        edge_match=None
    )

    seen_node_sets: set[frozenset[Any]] = set()
    instances: list[dict[str, Any]] = []

    for subgraph in matcher.subgraph_monomorphisms_iter():
        node_set = frozenset(subgraph.keys())

        # Avoid labeling the same group multiple times (automorphisms)
        if node_set in seen_node_sets:
            continue
        seen_node_sets.add(node_set)

        for node in node_set:
            graph.nodes[node]['label'] = motif_name

        instance_edges = frozenset(graph.subgraph(node_set).edges())
        instances.append({
            "motif_name": motif_name,
            "nodes": node_set,
            "edges": instance_edges,
        })

    return instances