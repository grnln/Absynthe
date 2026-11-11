"""Label assignment service."""
import web.backend.services  # noqa: F401

from typing import List, Tuple

from interfaces import GraphLike
from utils.kg_utils import apply_labeling_result
from web.backend.models.graph_models import LabelingFunctionConfig
from web.backend.services import graph_store, serialization
from web.backend.services.registry import build_labeling_functions


def _apply_labels(
    graph: GraphLike,
    labeling_configs: List[LabelingFunctionConfig],
    attribute_name: str,
) -> None:
    labelers = build_labeling_functions(labeling_configs)
    for labeler in labelers:
        apply_labeling_result(graph, labeler.label(graph), attribute_name)


def assign(
    graph_id: str,
    labeling_functions: List[LabelingFunctionConfig],
) -> Tuple[str, list, dict]:
    """Assign expected labels. Mutates stored graph in place."""
    graph = graph_store.get(graph_id)
    _apply_labels(graph, labeling_functions, "expected_ground_truth")
    graph_store.update(graph_id, graph)

    elements = serialization.graph_to_elements(graph)
    dist = serialization.label_distribution(graph, attr="expected_ground_truth")
    return graph_id, elements, dist


def reassign(
    graph_id: str,
    labeling_functions: List[LabelingFunctionConfig],
) -> Tuple[str, list, dict]:
    """Assign observed labels after perturbation. Mutates stored graph in place."""
    graph = graph_store.get(graph_id)
    _apply_labels(graph, labeling_functions, "observed_ground_truth")
    graph_store.update(graph_id, graph)

    elements = serialization.graph_to_elements(graph)
    dist = serialization.label_distribution(graph, attr="observed_ground_truth")
    return graph_id, elements, dist
