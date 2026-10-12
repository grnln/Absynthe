import networkx as nx

from typing import Any, List, Dict
from interfaces.labeling_function import LabelingFunction, LabelingResult
from interfaces.perturbation_hint import PerturbationHint

def compute_labels(graph: nx.Graph[Any], labeling_functions: List[LabelingFunction]) -> LabelingResult:
    """Compute labels using all labeling functions.
    
    Later labeling functions overwrite earlier ones for the same node.
    Edge labels, graph labels and metadata are merged across all functions. The
    perturbation hint is merged across functions; if no labeling
    function provides one, it is derived from ``details`` so that
    legacy labelers (which only fill ``motif_nodes`` / ``motif_edges``
    per node) still produce a usable hint for downstream perturbations.
    """
    merged = LabelingResult(node_labels={})

    for lf in labeling_functions:
        result = lf.label(graph)
        merged.node_labels.update(result.node_labels)
        merged.edge_labels.update(result.edge_labels)
        merged.graph_labels.update(result.graph_labels)
        for node, det in result.details.items():
            merged.details.setdefault(node, {}).update(det)
        merged.metadata.update(result.metadata)
        if result.hint is not None and not result.hint.is_empty():
            merged.hint = (
                result.hint if merged.hint is None
                else merged.hint.merge(result.hint)
            )
    
    if merged.hint is None or merged.hint.is_empty():
        derived = derive_hint_from_details(merged.details)
        merged.hint = derived if not derived.is_empty() else None
    return merged

def derive_hint_from_details(details: Dict[int, Dict[str, Any]]) -> PerturbationHint:
    """Derive a PerturbationHint from per-node ``details``.

    Looks for ``motif_nodes`` and ``motif_edges`` entries and
    accumulates them, normalizing edges to ``(min, max)``.
    """
    hint = PerturbationHint()
    for det in details.values():
        motif_nodes = det.get("motif_nodes")
        if motif_nodes:
            hint.nodes.update(motif_nodes)
        motif_edges = det.get("motif_edges")
        if motif_edges:
            for edge in motif_edges:
                u, v = edge[0], edge[1]
                hint.edges.add(PerturbationHint.normalize_edge(u, v))
    return hint