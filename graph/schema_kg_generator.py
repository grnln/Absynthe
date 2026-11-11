"""Synthetic knowledge-graph generator driven by a small schema.

Example::

    gen = SchemaKGGenerator(
        entity_types={"Person": 10, "Company": 3},
        relations=[
            {"name": "works_at", "domain": "Person", "range": "Company", "p": 0.3},
            {"name": "knows", "domain": "Person", "range": "Person", "count": 12},
        ],
    )
    G = gen.generate_graph()   # nx.MultiDiGraph with type/relation attributes
"""
from typing import Any, Dict, List, Optional

import networkx as nx

from interfaces import GraphGenerator, GraphLike
from utils.rng import get_rng
from utils.kg_utils import EDGE_RELATION_ATTR, KG_FLAG, NODE_TYPE_ATTR, add_edge


class SchemaKGGenerator(GraphGenerator):
    """Sample a multi-relational knowledge graph from an entity/relation schema.

    This is one possible ``GraphGenerator`` for knowledge graphs; real KGs
    can be loaded instead with ``FolderGraphGenerator`` (RDF / GraphML).

    Args:
        entity_types: ``{type_name: number_of_entities}``.
        relations: One dict per relation type with ``name``, ``domain``
            (head entity type), ``range`` (tail entity type) and either
            ``p`` (Bernoulli probability per candidate pair) or ``count``
            (exact number of triples sampled without replacement).
        directed: Produce a directed graph (default) or an undirected one.
        multigraph: Allow several relations between the same pair
            (default). With ``False`` a later relation overwrites an
            earlier one on the same pair.
        node_id_format: Format string for node ids, receiving ``type`` and ``i``.
        allow_self_loops: Allow ``(x, r, x)`` triples for relations whose
            domain and range coincide.
    """

    def __init__(
        self,
        entity_types: Dict[str, int],
        relations: List[Dict[str, Any]],
        directed: bool = True,
        multigraph: bool = True,
        node_id_format: str = "{type}_{i}",
        allow_self_loops: bool = False,
    ):
        if not entity_types:
            raise ValueError("entity_types must not be empty")
        for rel in relations:
            for field in ("name", "domain", "range"):
                if field not in rel:
                    raise ValueError(f"Relation spec {rel} is missing '{field}'")
            for side in ("domain", "range"):
                if rel[side] not in entity_types:
                    raise ValueError(
                        f"Relation '{rel['name']}': unknown entity type '{rel[side]}'"
                    )
        self.entity_types = dict(entity_types)
        self.relations = [dict(r) for r in relations]
        self.directed = directed
        self.multigraph = multigraph
        self.node_id_format = node_id_format
        self.allow_self_loops = allow_self_loops

    def _empty_graph(self) -> GraphLike:
        if self.directed:
            return nx.MultiDiGraph() if self.multigraph else nx.DiGraph()
        return nx.MultiGraph() if self.multigraph else nx.Graph()

    def generate_graph(self, **kwargs) -> GraphLike:
        """Sample one knowledge graph. Extra kwargs (``num_extra_vertices``,
        ``composition``...) passed by the dataset generator are ignored."""
        rng = kwargs.get("rng") or get_rng()
        G = self._empty_graph()
        G.graph[KG_FLAG] = True

        nodes_by_type: Dict[str, List[str]] = {}
        for etype, count in self.entity_types.items():
            ids = [self.node_id_format.format(type=etype, i=i) for i in range(int(count))]
            G.add_nodes_from((n, {NODE_TYPE_ATTR: etype}) for n in ids)
            nodes_by_type[etype] = ids

        for rel in self.relations:
            heads = nodes_by_type[rel["domain"]]
            tails = nodes_by_type[rel["range"]]
            pairs = [
                (h, t) for h in heads for t in tails
                if self.allow_self_loops or h != t
            ]
            if rel.get("count") is not None:
                k = min(int(rel["count"]), len(pairs))
                chosen = rng.sample(pairs, k) if k > 0 else []
            else:
                p = float(rel.get("p", 0.1))
                chosen = [pair for pair in pairs if rng.random() < p]
            for h, t in chosen:
                add_edge(G, h, t, **{EDGE_RELATION_ATTR: rel["name"]})

        return G
