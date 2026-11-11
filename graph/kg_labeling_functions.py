"""Labeling functions for multi-relational knowledge graphs."""
from typing import Any, Dict, List, Optional

from interfaces import LabelingFunction, PerturbationHint, GraphLike
from interfaces.labeling_result import LabelingResult
from utils.kg_utils import (
    EDGE_RELATION_ATTR,
    NODE_TYPE_ATTR,
    edge_key,
    entity_type_counts,
    iter_edges,
    relation_counts,
)


class EntityTypeLabelingFunction(LabelingFunction):
    """Label nodes by entity type and edges by relation type.

    Optionally, ``patterns`` derive node labels from the relations a node
    participates in, so that removing or corrupting a triple changes the
    label of its endpoints (which is what the perturbation pipeline needs to
    accept a perturbation)::

        EntityTypeLabelingFunction(patterns={
            "employee": {"relation": "works_at", "direction": "out"},
            "employer": {"relation": "works_at", "direction": "in", "node_type": "Company"},
        })

    Each pattern has a ``relation`` (required), a ``direction`` (``"out"``:
    label the head, ``"in"``: label the tail, ``"any"``: both; ignored on
    undirected graphs) and an optional ``node_type`` filter. Later patterns
    overwrite earlier ones. Nodes matching no pattern keep their entity
    type as label (or ``default`` when they have none).

    The result carries ``edge_labels`` keyed by ``utils.kg_utils.edge_key``
    and a ``PerturbationHint`` covering the edges matched by the patterns and
    both of their endpoints (perturbing either endpoint can change the
    label of the other), so that hint-aware perturbations such as
    ``RemoveNodesPerturbation(strategy="by_attribute")`` can target them.
    """

    def __init__(
        self,
        node_attr: str = NODE_TYPE_ATTR,
        edge_attr: str = EDGE_RELATION_ATTR,
        default: str = "unknown",
        patterns: Optional[Dict[str, Dict[str, Any]]] = None,
        label_edges: bool = True,
    ):
        self.node_attr = node_attr
        self.edge_attr = edge_attr
        self.default = default
        self.patterns = patterns or {}
        self.label_edges = label_edges
        for name, spec in self.patterns.items():
            if "relation" not in spec:
                raise ValueError(f"Pattern '{name}' must define a 'relation'")

    def label(self, graph: GraphLike) -> LabelingResult:
        directed = graph.is_directed()
        node_labels: Dict[Any, Any] = {
            n: str(d.get(self.node_attr, self.default))
            for n, d in graph.nodes(data=True)
        }
        edge_labels: Dict[tuple, Any] = {}
        if self.label_edges:
            for u, v, k, d in iter_edges(graph):
                if self.edge_attr in d:
                    edge_labels[edge_key(graph, u, v, k)] = str(d[self.edge_attr])

        details: Dict[Any, Dict[str, Any]] = {}
        hint = PerturbationHint()
        for name, spec in self.patterns.items():
            relation = str(spec["relation"])
            direction = spec.get("direction", "any") if directed else "any"
            node_type = spec.get("node_type")
            for u, v, k, d in iter_edges(graph):
                if str(d.get(self.edge_attr)) != relation:
                    continue
                if direction == "out":
                    targets: List[Any] = [u]
                elif direction == "in":
                    targets = [v]
                else:
                    targets = [u, v]
                if node_type is not None:
                    targets = [n for n in targets if graph.nodes[n].get(self.node_attr) == node_type]
                if not targets:
                    continue
                ref = edge_key(graph, u, v, k)
                for n in targets:
                    node_labels[n] = name
                    det = details.setdefault(n, {"patterns": set(), "pattern_edges": set()})
                    det["patterns"].add(name)
                    det["pattern_edges"].add(ref)
                hint.nodes.update((u, v))
                hint.edges.add(PerturbationHint.normalize_edge(u, v, k, directed=directed))

        return LabelingResult(
            node_labels=node_labels,
            edge_labels=edge_labels,
            graph_labels={},
            details=details,
            metadata={
                "entity_type_counts": entity_type_counts(graph, self.node_attr),
                "relation_counts": relation_counts(graph, self.edge_attr),
            },
            hint=hint if not hint.is_empty() else None,
        )
