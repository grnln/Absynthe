from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .perturbation_hint import PerturbationHint


@dataclass
class LabelingResult:
    """Rich result from a labeling function.

    Attributes:
        node_labels: Mapping from node ID to its label (node-level labeling).
        edge_labels: Mapping from edge reference to its label (edge-level
            labeling). A reference is ``(u, v)`` for simple graphs or
            ``(u, v, key)`` for multigraphs, as built by
            ``utils.kg_utils.edge_key``.
        graph_labels: Named graph-level labels, keyed by label name.
            Multiple labeling functions can contribute different graph-level
            labels without overwriting each other.
        details: Per-node extra information. Each labeling function
            decides what to include (e.g., motif nodes/edges involved
            in the labeling of each node).
        metadata: Arbitrary global information the labeling function
            wants to communicate (e.g., list of detected motif instances).
        hint: Optional ``PerturbationHint`` the labeling function emits
            for downstream perturbations, signaling which nodes/edges
            are part of the relevant zone. If left as ``None``, the
            perturbation pipeline derives a hint automatically from
            ``details`` (``motif_nodes`` / ``motif_edges``).
    """

    node_labels: Dict[Any, Any]
    edge_labels: Dict[Tuple, Any] = field(default_factory=dict)
    graph_labels: Dict[str, Any] = field(default_factory=dict)
    details: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hint: Optional[PerturbationHint] = None
