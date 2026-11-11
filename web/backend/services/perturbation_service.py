"""Perturbation service."""
from typing import Any, Dict, List, Optional

import web.backend.services  # noqa: F401

from interfaces import LabelingFunction
from utils.kg_utils import EDGE_RELATION_ATTR, apply_labeling_result
from web.backend.models.perturbation_models import (
    ChangedNode,
    EdgeChange,
    EdgePerturbInfo,
    PerturbationPreview,
    PerturbationRequest,
    PerturbationResponse,
    TripleChange,
)
from web.backend.services import graph_store, serialization
from web.backend.services.registry import build_labeling_functions, build_perturbation, count_spec


def _apply_observed_labels(graph, labelers: List[LabelingFunction]) -> None:
    """Compute and store observed labels on the perturbed graph."""
    for labeler in labelers:
        apply_labeling_result(graph, labeler.label(graph), "observed_ground_truth")


def _map_changed_nodes(changed_raw: dict) -> List[ChangedNode]:
    return [
        ChangedNode(node_id=str(node_id), old_label=str(old), new_label=str(new))
        for node_id, (old, new) in changed_raw.items()
    ]


def _opt_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _edge_change(rec: Dict[str, Any]) -> EdgeChange:
    """Map a reversible edge record ``{"u", "v"[, "key"][, "attrs"]}``."""
    return EdgeChange(
        source=str(rec["u"]),
        target=str(rec["v"]),
        key=_opt_str(rec.get("key")),
        relation=_opt_str(rec.get("attrs", {}).get(EDGE_RELATION_ATTR)),
    )


def _triple_change(rec: Dict[str, Any]) -> TripleChange:
    return TripleChange(
        source=str(rec["u"]),
        target=str(rec["v"]),
        key=_opt_str(rec.get("key")),
        relation=_opt_str(rec.get("attrs", {}).get(EDGE_RELATION_ATTR)),
        new_source=str(rec["new_u"]),
        new_target=str(rec["new_v"]),
        new_key=_opt_str(rec.get("new_key")),
        new_relation=_opt_str(rec.get("new_attrs", {}).get(EDGE_RELATION_ATTR)),
    )


def _map_edge_info(changes: dict) -> Dict[str, EdgePerturbInfo]:
    removed_edges = [_edge_change(rec) for rec in changes.get("removed_edges", [])]
    added_edges = [_edge_change(rec) for rec in changes.get("added_edges", [])]
    corrupted = [_triple_change(rec) for rec in changes.get("corrupted_triples", [])]
    if not removed_edges and not added_edges and not corrupted:
        return {}
    return {
        "applied": EdgePerturbInfo(
            removed_edges=removed_edges,
            added_edges=added_edges,
            corrupted_triples=corrupted,
        )
    }


def apply(request: PerturbationRequest) -> PerturbationResponse:
    """Apply configured perturbations and return preview diff payloads."""
    from graph.perturbation_engine import PerturbationPipeline
    from utils.distributions import resolve_count
    from utils.rng import get_rng, set_seed, reset_rng

    if request.seed is not None:
        set_seed(request.seed)

    try:
        original_graph = graph_store.get(request.graph_id)
        original_elements = serialization.graph_to_elements(original_graph)

        labelers = build_labeling_functions(request.labeling_functions)
        if not request.perturbations:
            raise ValueError("At least one perturbation must be configured.")

        previews: List[PerturbationPreview] = []
        primary_graph = None
        primary_preview = None

        for idx, perturb_config in enumerate(request.perturbations):
            # The preview works on a single graph: sample the count once here
            # so the reported desired_count matches what the pipeline targets.
            desired_count = resolve_count(
                count_spec(perturb_config.count, perturb_config.count_distribution), get_rng()
            )
            pipeline = PerturbationPipeline(
                perturbations=[(build_perturbation(perturb_config), desired_count)],
                labeling_functions=labelers,
                max_iterations=request.max_iterations,
            )
            results = pipeline.apply_and_check(original_graph)

            if results:
                result = results[0]
                perturbed_graph = result["perturbed_graph"]
                _apply_observed_labels(perturbed_graph, labelers)
                changes = result.get("changes", {})
                changed_nodes = _map_changed_nodes(result.get("changed_nodes", {}))
                success = True
                message = ""
            else:
                perturbed_graph = original_graph.copy()
                _apply_observed_labels(perturbed_graph, labelers)
                changes = {}
                changed_nodes = []
                success = False
                message = (
                    f"No label changes produced for perturbation "
                    f"'{perturb_config.type}'."
                )

            preview = PerturbationPreview(
                config_index=idx,
                perturbation_type=perturb_config.type,
                desired_count=desired_count,
                success=success,
                message=message,
                original_elements=original_elements,
                perturbed_elements=serialization.graph_to_elements(perturbed_graph),
                removed_nodes=[str(node["id"]) for node in changes.get("removed_nodes", [])],
                changed_nodes=changed_nodes,
                edge_perturb_info=_map_edge_info(changes),
            )
            previews.append(preview)

            if primary_preview is None and success:
                primary_preview = preview
                primary_graph = perturbed_graph

        if primary_preview is None:
            primary_preview = previews[0]
            primary_graph = original_graph.copy()
            _apply_observed_labels(primary_graph, labelers)

        perturbed_id = graph_store.store(primary_graph)
        any_success = any(preview.success for preview in previews)

        return PerturbationResponse(
            original_graph_id=request.graph_id,
            perturbed_graph_id=perturbed_id,
            original_elements=primary_preview.original_elements,
            perturbed_elements=primary_preview.perturbed_elements,
            removed_nodes=primary_preview.removed_nodes,
            changed_nodes=primary_preview.changed_nodes,
            edge_perturb_info=primary_preview.edge_perturb_info,
            previews=previews,
            success=any_success,
            message="" if any_success else "No perturbation caused label changes.",
        )
    finally:
        if request.seed is not None:
            reset_rng()
