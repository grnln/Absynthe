"""Convert between NetworkX graphs and Cytoscape elements JSON."""
from typing import Any, Dict, List

from interfaces import GraphLike
from utils.kg_utils import (
    EDGE_RELATION_ATTR,
    NODE_TYPE_ATTR,
    entity_type_counts,
    graph_kind,
    iter_edges,
    relation_counts,
)
from web.backend.models.common import CytoscapeElement
from web.backend.models.graph_models import GraphStats

NODE_ATTRS = (
    "label",
    "motif",
    "motif_id",
    NODE_TYPE_ATTR,
    "expected_ground_truth",
    "observed_ground_truth",
)


def graph_to_elements(graph: GraphLike) -> List[CytoscapeElement]:
    """Convert a NetworkX graph (any class) to a list of Cytoscape element dicts.

    Edges carry ``relation`` / ``label`` when present and ``key`` on
    multigraphs, so parallel relations between the same pair stay distinct.
    """
    elements: List[CytoscapeElement] = []

    for node, attrs in graph.nodes(data=True):
        data: Dict[str, Any] = {"id": str(node)}
        # copy all recognised string attributes
        for key in NODE_ATTRS:
            val = attrs.get(key, "")
            data[key] = str(val) if val is not None else ""
        elements.append(CytoscapeElement(group="nodes", data=data))

    multigraph = graph.is_multigraph()
    for u, v, k, attrs in iter_edges(graph):
        edge_id = f"{u}-{v}-{k}" if multigraph else f"{u}-{v}"
        data = {"id": edge_id, "source": str(u), "target": str(v)}
        if multigraph:
            data["key"] = str(k)
        if EDGE_RELATION_ATTR in attrs:
            data["relation"] = str(attrs[EDGE_RELATION_ATTR])
        if "label" in attrs:
            data["label"] = str(attrs["label"])
        elements.append(CytoscapeElement(group="edges", data=data))

    return elements


def graph_stats(graph: GraphLike) -> GraphStats:
    """Summary statistics shared by the generate / upload endpoints."""
    motif_counts: Dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        motif_name = data.get("motif", "")
        if motif_name:
            motif_counts[motif_name] = motif_counts.get(motif_name, 0) + 1

    kind = graph_kind(graph)
    return GraphStats(
        num_nodes=graph.number_of_nodes(),
        num_edges=graph.number_of_edges(),
        motif_counts=motif_counts,
        directed=kind["directed"],
        multigraph=kind["multigraph"],
        kg=kind["kg"],
        relation_counts=relation_counts(graph),
        entity_type_counts=entity_type_counts(graph),
    )


def label_distribution(graph: GraphLike, attr: str = "label") -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        lbl = str(data.get(attr, "unknown"))
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts
