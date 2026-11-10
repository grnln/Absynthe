"""Plugin registry and builders for web-configurable modules."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import web.backend.services  # noqa: F401
from interfaces import LabelingFunction, Perturbation
from utils.distributions import CountSpec, to_distribution
from web.backend.models.graph_models import LabelingFunctionConfig
from web.backend.models.perturbation_models import PerturbationConfig

# UI schemas
MOTIF_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "cycle",
        "label": "Cycle",
        "description": "A ring of n nodes",
        "params": [{"name": "len_cycle", "label": "Length", "type": "int", "default": 4, "min": 3}],
    },
    {
        "type": "house",
        "label": "House",
        "description": "5-node house (square base + triangular roof)",
        "params": [],
    },
    {
        "type": "chain",
        "label": "Chain",
        "description": "A linear path of n nodes",
        "params": [{"name": "length", "label": "Length", "type": "int", "default": 3, "min": 2}],
    },
    {
        "type": "star",
        "label": "Star",
        "description": "Central hub with n leaf nodes",
        "params": [{"name": "num_leaves", "label": "Leaves", "type": "int", "default": 3, "min": 1}],
    },
    {
        "type": "gate",
        "label": "Gate",
        "description": "Entry node, two parallel arms, exit node",
        "params": [{"name": "arm_length", "label": "Arm length", "type": "int", "default": 1, "min": 1}],
    },
    {
        "type": "grid",
        "label": "Grid",
        "description": "Rectangular lattice of rows x cols nodes",
        "params": [
            {"name": "rows", "label": "Rows", "type": "int", "default": 3, "min": 2},
            {"name": "cols", "label": "Columns", "type": "int", "default": 3, "min": 2},
        ],
    },
    {
        "type": "clique",
        "label": "Clique",
        "description": "Complete graph K_n: every pair of n nodes is connected",
        "params": [{"name": "num_nodes", "label": "Nodes", "type": "int", "default": 4, "min": 2}],
    },
]

COMPOSITION_SCHEMAS: List[Dict[str, Any]] = [
    {"id": "sequential", "label": "Sequential", "description": "Linear chain between motifs", "params": []},
    {
        "id": "er",
        "label": "Erdos-Renyi",
        "description": "Random edges between motif pairs with probability p",
        "params": [{"name": "p", "label": "Probability", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0}],
    },
    {
        "id": "ba",
        "label": "Barabasi-Albert",
        "description": "Scale-free preferential attachment",
        "params": [{"name": "m", "label": "Attachments (m)", "type": "int", "default": 1, "min": 1}],
    },
    {
        "id": "sbm",
        "label": "Stochastic Block Model",
        "description": "Community-structured composition",
        "params": [
            {"name": "p_in", "label": "p_in", "type": "float", "default": 0.6},
            {"name": "p_out", "label": "p_out", "type": "float", "default": 0.05},
            {"name": "blocks", "label": "Blocks (comma-separated)", "type": "string", "default": ""},
        ],
    },
    {
        "id": "star",
        "label": "Star",
        "description": "One central motif connected to all others",
        "params": [{"name": "center", "label": "Center motif index", "type": "int", "default": 0, "min": 0}],
    },
    {
        "id": "hierarchical",
        "label": "Hierarchical",
        "description": "Motifs grouped into clusters with inter-group connections",
        "params": [{"name": "groups", "label": "Groups", "type": "int", "default": 2, "min": 1}],
    },
]

STRATEGY_SCHEMAS: List[Dict[str, Any]] = [
    {"id": "random", "label": "Random", "description": "Pick nodes at random", "params": []},
    {"id": "motif", "label": "Motif", "description": "Remove nodes from the same motif instance", "params": []},
    {
        "id": "degree",
        "label": "Degree",
        "description": "Highest or lowest degree nodes",
        "params": [{"name": "mode", "label": "Mode", "type": "select", "options": ["high", "low"], "default": "high"}],
    },
    {"id": "centrality", "label": "Centrality", "description": "Highest betweenness centrality", "params": []},
    {
        "id": "role",
        "label": "Role",
        "description": "Nodes matching a specific role attribute",
        "params": [{"name": "role", "label": "Role value", "type": "string", "default": ""}],
    },
    {
        "id": "by_attribute",
        "label": "By Attribute",
        "description": "Nodes matching a custom attribute value",
        "params": [
            {"name": "attr", "label": "Attribute name", "type": "string", "default": "motif"},
            {"name": "value", "label": "Attribute value", "type": "string", "default": ""},
        ],
    },
]

DISTRIBUTION_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "uniform",
        "label": "Uniform",
        "description": "Equally likely between min and max",
        "params": [
            {"name": "min", "label": "Min", "type": "int", "default": 1, "min": 1},
            {"name": "max", "label": "Max", "type": "int", "default": 5, "min": 1},
        ],
    },
    {
        "type": "normal",
        "label": "Normal",
        "description": "Bell curve around mean, clamped to >= 1",
        "params": [
            {"name": "mean", "label": "Mean", "type": "float", "default": 3.0, "min": 0.1},
            {"name": "std", "label": "Std Dev", "type": "float", "default": 1.0, "min": 0.1},
        ],
    },
    {
        "type": "poisson",
        "label": "Poisson",
        "description": "Count events with average rate lambda",
        "params": [
            {"name": "lambda", "label": "Lambda", "type": "float", "default": 3.0, "min": 0.1},
        ],
    },
]

LABELING_FUNCTION_SCHEMAS: List[Dict[str, Any]] = [
    {
        "id": "motif_labeling",
        "label": "Motif Labeling",
        "description": "Assign labels based on motif subgraph membership.",
        "params": [
            {
                "name": "motif_order",
                "label": "Motif order (comma-separated)",
                "type": "string",
                "default": "",
            }
        ],
    },
    {
        "id": "entity_type_labeling",
        "label": "Entity Type Labeling (KG)",
        "description": (
            "Label nodes by entity type and edges by relation type; optional "
            "patterns derive node labels from the relations a node takes part in."
        ),
        "params": [
            {"name": "node_attr", "label": "Node type attribute", "type": "string", "default": "type"},
            {"name": "edge_attr", "label": "Edge relation attribute", "type": "string", "default": "relation"},
            {"name": "default", "label": "Default label", "type": "string", "default": "unknown"},
            {
                "name": "patterns",
                "label": "Patterns (JSON: {name: {relation, direction, node_type}})",
                "type": "string",
                "default": "",
            },
        ],
    },
]

KG_SCHEMA_SCHEMA: Dict[str, Any] = {
    "id": "kg_schema",
    "label": "Synthetic knowledge graph",
    "description": "Sample a multi-relational KG from entity types and relation specs (domain, range, p | count).",
    "params": [
        {"name": "entity_types", "label": "Entity types (JSON: {type: count})", "type": "string", "default": "{}"},
        {"name": "relations", "label": "Relations (JSON list of {name, domain, range, p | count})", "type": "string", "default": "[]"},
        {"name": "directed", "label": "Directed", "type": "select", "options": ["true", "false"], "default": "true"},
        {"name": "multigraph", "label": "Multigraph", "type": "select", "options": ["true", "false"], "default": "true"},
    ],
}

_RELATIONS_PARAM = {
    "name": "relations",
    "label": "Relation types (comma-separated, KG only)",
    "type": "string",
    "default": "",
}

PERTURBATION_SCHEMAS: List[Dict[str, Any]] = [
    {
        "id": "remove_nodes",
        "label": "Remove Nodes",
        "description": "Remove nodes using a selection strategy.",
        "params": [
            {"name": "num_nodes", "label": "Nodes to remove", "type": "int", "default": 1, "min": 1},
            {
                "name": "strategy",
                "label": "Strategy",
                "type": "select",
                "options": [s["id"] for s in STRATEGY_SCHEMAS],
                "default": "random",
            },
            {"name": "mode", "label": "Degree mode", "type": "select", "options": ["high", "low"], "default": "high"},
            {"name": "role", "label": "Role filter", "type": "string", "default": ""},
            {"name": "attr", "label": "Attribute key", "type": "string", "default": ""},
            {"name": "value", "label": "Attribute value", "type": "string", "default": ""},
        ],
    },
    {
        "id": "remove_edges",
        "label": "Remove Edges",
        "description": "Randomly remove edges with probability p_remove (optionally only of given relation types).",
        "params": [
            {"name": "p_remove", "label": "p_remove", "type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
            _RELATIONS_PARAM,
        ],
    },
    {
        "id": "add_edges",
        "label": "Add Edges",
        "description": "Add edges by probability or explicit count.",
        "params": [
            {"name": "p_add", "label": "p_add", "type": "float", "default": 0.05, "min": 0.0, "max": 1.0},
            {"name": "add_num", "label": "add_num", "type": "int", "default": 0, "min": 0},
        ],
    },
    {
        "id": "edge_perturbation",
        "label": "Edge Perturbation",
        "description": "Combined edge removal and addition.",
        "params": [
            {"name": "p_remove", "label": "p_remove", "type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
            {"name": "p_add", "label": "p_add", "type": "float", "default": 0.05, "min": 0.0, "max": 1.0},
            {"name": "add_num", "label": "add_num", "type": "int", "default": 0, "min": 0},
            _RELATIONS_PARAM,
        ],
    },
    {
        "id": "corrupt_triples",
        "label": "Corrupt Triples (KG)",
        "description": "Replace the relation, head or tail of triples (negative-sampling style), reversibly.",
        "params": [
            {"name": "num_triples", "label": "Triples to corrupt", "type": "int", "default": 1, "min": 1},
            {
                "name": "mode",
                "label": "Mode",
                "type": "select",
                "options": ["relation", "head", "tail", "any"],
                "default": "relation",
            },
            _RELATIONS_PARAM,
            {
                "name": "same_type",
                "label": "Replacement endpoint of the same entity type",
                "type": "select",
                "options": ["true", "false"],
                "default": "true",
            },
        ],
    },
]


def _parse_csv_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [chunk.strip() for chunk in str(raw).split(",") if chunk.strip()]


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_json_param(raw: Any, default: Any) -> Any:
    """Accept an already-parsed JSON value or a JSON string (UI forms send strings)."""
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw).strip()
    if not text:
        return default
    import json

    return json.loads(text)


def normalize_composition_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    if not params:
        return {}
    normalized = dict(params)
    blocks = normalized.get("blocks")
    if isinstance(blocks, str):
        values = [chunk.strip() for chunk in blocks.split(",") if chunk.strip()]
        normalized["blocks"] = [int(chunk) for chunk in values if chunk.isdigit()]
    return normalized


def count_spec(fixed: int, distribution: Any = None) -> CountSpec:
    """Pick the runtime count: the ``IntDistribution`` when a distribution
    model is given (sampled per graph downstream), else the fixed int."""
    dist = to_distribution(distribution)
    return dist if dist is not None else fixed


def build_graph_kwargs(request: Any) -> Dict[str, Any]:
    """Keyword arguments for ``generate_graph`` / ``generate_dataset`` built
    from a ``GraphGenerateRequest`` or ``DatasetGenerateRequest``.

    ``num_extra_vertices`` / ``num_extra_edges`` become ``IntDistribution``
    objects when the request carries ``*_distribution`` fields, so motif
    generators re-sample them for every graph.
    """
    return {
        "num_extra_vertices": count_spec(
            request.num_extra_vertices, request.num_extra_vertices_distribution
        ),
        "num_extra_edges": count_spec(
            request.num_extra_edges, request.num_extra_edges_distribution
        ),
        "composition": request.composition,
        "composition_params": normalize_composition_params(request.composition_params),
    }


def build_labeling_functions(
    configs: List[LabelingFunctionConfig] | None,
) -> List[LabelingFunction]:
    from graph.labeling_functions import MotifLabelingFunction

    if not configs:
        return [MotifLabelingFunction()]

    labeling_functions: List[LabelingFunction] = []
    for cfg in configs:
        cfg_type = (cfg.type or "").lower()
        params = cfg.params or {}
        if cfg_type == "motif_labeling":
            motif_order = _parse_csv_list(params.get("motif_order"))
            labeling_functions.append(
                MotifLabelingFunction(motif_order=motif_order or None)
            )
        elif cfg_type == "entity_type_labeling":
            from graph.kg_labeling_functions import EntityTypeLabelingFunction

            labeling_functions.append(
                EntityTypeLabelingFunction(
                    node_attr=str(params.get("node_attr") or "type"),
                    edge_attr=str(params.get("edge_attr") or "relation"),
                    default=str(params.get("default") or "unknown"),
                    patterns=_parse_json_param(params.get("patterns"), {}) or None,
                    label_edges=_to_bool(params.get("label_edges"), True),
                )
            )
        else:
            raise ValueError(f"Unknown labeling function: {cfg.type}")
    return labeling_functions


def build_perturbation(config: PerturbationConfig) -> Perturbation:
    from graph.perturbations import (
        AddEdgesPerturbation,
        EdgePerturbation,
        RemoveEdgesPerturbation,
        RemoveNodesPerturbation,
    )

    cfg_type = (config.type or "").lower()
    params = config.params or {}
    folder_name = config.folder_name

    if cfg_type == "remove_nodes":
        strategy = str(params.get("strategy", "random"))
        num_nodes = max(1, _to_int(params.get("num_nodes"), 1))
        strategy_params: Dict[str, Any] = {}
        if strategy == "degree":
            strategy_params["mode"] = str(params.get("mode", "high"))
        elif strategy == "role":
            role = str(params.get("role", "")).strip()
            if role:
                strategy_params["role"] = role
        elif strategy == "by_attribute":
            attr = str(params.get("attr", "")).strip()
            value = str(params.get("value", "")).strip()
            if attr:
                strategy_params["attr"] = attr
            if value:
                strategy_params["value"] = value
        return RemoveNodesPerturbation(
            num_nodes=num_nodes,
            strategy=strategy,
            params=strategy_params or None,
            folder_name=folder_name,
        )

    relations = _parse_csv_list(params.get("relations")) or None
    add_relation = str(params.get("add_relation")).strip() if params.get("add_relation") else None

    if cfg_type == "remove_edges":
        return RemoveEdgesPerturbation(
            p_remove=_to_float(params.get("p_remove"), 0.1),
            folder_name=folder_name,
            relations=relations,
        )

    if cfg_type == "add_edges":
        add_num = _to_int(params.get("add_num"), 0)
        return AddEdgesPerturbation(
            p_add=_to_float(params.get("p_add"), 0.05),
            add_num=add_num if add_num > 0 else None,
            folder_name=folder_name,
            add_relation=add_relation,
        )

    if cfg_type == "edge_perturbation":
        add_num = _to_int(params.get("add_num"), 0)
        return EdgePerturbation(
            p_remove=_to_float(params.get("p_remove"), 0.1),
            p_add=_to_float(params.get("p_add"), 0.05),
            add_num=add_num if add_num > 0 else None,
            folder_name=folder_name,
            relations=relations,
            add_relation=add_relation,
        )

    if cfg_type == "corrupt_triples":
        from graph.kg_perturbations import CorruptTriplesPerturbation

        return CorruptTriplesPerturbation(
            num_triples=max(1, _to_int(params.get("num_triples"), 1)),
            mode=str(params.get("mode") or "relation"),
            relations=relations,
            same_type=_to_bool(params.get("same_type"), True),
            folder_name=folder_name,
        )

    raise ValueError(f"Unknown perturbation type: {config.type}")


def build_perturbations(
    configs: List[PerturbationConfig] | None,
) -> List[Tuple[Perturbation, CountSpec]]:
    """``(perturbation, count)`` pairs; ``count`` is the ``IntDistribution``
    when the config has ``count_distribution`` (sampled per base graph)."""
    if not configs:
        return []
    return [
        (build_perturbation(cfg), count_spec(cfg.count, cfg.count_distribution))
        for cfg in configs
    ]


def build_graph_generator(request: "DatasetGenerateRequest"):
    """Build the graph generator (folder, KG schema, random-motif, or fixed-motif) from a request.

    Mirrors the dispatch already used in dataset_service.run_generation so the same
    logic is shared between the web service and the CLI loader.
    """
    from graph.composite_graph_generator import MotifComposite
    from graph.folder_graph_generator import (
        FolderGraphGenerator,
        IterationOrder,
        ExhaustionPolicy,
    )

    if request.folder_source:
        return FolderGraphGenerator(
            folder_path=request.folder_source.folder_path,
            iteration_order=IterationOrder(request.folder_source.iteration_order),
            exhaustion_policy=ExhaustionPolicy(request.folder_source.exhaustion_policy),
            as_undirected=request.folder_source.as_undirected,
            rdf_options=request.folder_source.rdf_options or None,
        )

    if request.kg_schema:
        from graph.schema_kg_generator import SchemaKGGenerator

        schema = request.kg_schema
        return SchemaKGGenerator(
            entity_types=dict(schema.entity_types),
            relations=[r.model_dump(exclude_none=True) for r in schema.relations],
            directed=schema.directed,
            multigraph=schema.multigraph,
            allow_self_loops=schema.allow_self_loops,
        )

    if any(m.count_distribution for m in request.motifs):
        from graph.random_motif_composite import RandomMotifComposite

        motif_configs = [
            (m.to_list(), m.count, to_distribution(m.count_distribution))
            for m in request.motifs
        ]
        return RandomMotifComposite(motif_configs=motif_configs)

    motif_lists = []
    for m in request.motifs:
        motif_lists.extend([m.to_list()] * m.count)
    return MotifComposite(motifs=motif_lists)
