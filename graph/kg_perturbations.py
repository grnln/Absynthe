"""Perturbations specific to multi-relational knowledge graphs."""
import random
from typing import Any, Dict, List, Optional, Tuple

from interfaces import Perturbation, PerturbationHint, GraphLike
from utils.rng import get_rng
from utils.kg_utils import (
    EDGE_RELATION_ATTR,
    NODE_TYPE_ATTR,
    add_edge,
    has_edge,
    iter_edges,
    relation_vocab,
    remove_edge,
)
from .perturbations import _edge_in_zone, _relation_allowed


class CorruptTriplesPerturbation(Perturbation):
    """Corrupt triples ``(head, relation, tail)`` the way negative sampling does.

    Modes:
        ``"relation"``: replace the relation type by another one from the
            graph's vocabulary. With ``same_type`` (default), relations that
            already occur between the same pair of entity types are
            preferred, so the corrupted triple stays schema-consistent
            (``Person -works_at-> Company`` becomes ``Person -founded->
            Company`` rather than ``Person -cites-> Company``); the full
            vocabulary is used only when no such relation exists.
        ``"head"`` / ``"tail"``: replace one endpoint by another node
            (of the same entity ``type`` when ``same_type`` and types exist).
        ``"any"``: pick one of the above at random per triple.

    Corrupting never duplicates an already existing triple. The change is
    reversible: each record stores the old and new triple (with edge keys
    on multigraphs) under ``"corrupted_triples"``.

    On graphs without ``relation`` attributes the ``relation`` mode produces
    no change; ``head`` / ``tail`` degrade to plain edge rewiring.
    """

    folder_name = "corrupt_triples"

    def __init__(
        self,
        num_triples: int = 1,
        mode: str = "relation",
        relations: Optional[List[str]] = None,
        same_type: bool = True,
        rng: Optional[random.Random] = None,
        folder_name: Optional[str] = None,
    ):
        if mode not in ("relation", "head", "tail", "any"):
            raise ValueError(f"Unknown corruption mode: {mode}")
        self.num_triples = num_triples
        self.mode = mode
        self.relations = relations
        self.same_type = same_type
        self.rng = rng
        if folder_name is not None:
            self.folder_name = folder_name

    # -- helpers -----------------------------------------------------------

    def _candidate_nodes(self, G: GraphLike, endpoint: Any, nodes: List[Any]) -> List[Any]:
        etype = G.nodes[endpoint].get(NODE_TYPE_ATTR)
        if self.same_type and etype is not None:
            return [n for n in nodes if G.nodes[n].get(NODE_TYPE_ATTR) == etype]
        return nodes

    @staticmethod
    def _triple_exists(G: GraphLike, u: Any, v: Any, relation: Any) -> bool:
        if not G.has_edge(u, v):
            return False
        if not G.is_multigraph():
            return True
        return any(
            str(d.get(EDGE_RELATION_ATTR)) == str(relation)
            for d in G[u][v].values()
        )

    # -- Perturbation ------------------------------------------------------

    def apply(self, graph: GraphLike,
              hint: Optional[PerturbationHint] = None
              ) -> Tuple[GraphLike, Dict[str, Any]]:
        rng = self.rng or get_rng()
        G = graph.copy()
        directed = G.is_directed()

        candidates = [
            (u, v, k, dict(d))
            for u, v, k, d in iter_edges(G)
            if _edge_in_zone(u, v, hint, k, directed) and _relation_allowed(d, self.relations)
        ]
        n = min(self.num_triples, len(candidates))
        chosen = rng.sample(candidates, n) if n > 0 else []

        vocab = relation_vocab(G)
        nodes = list(G.nodes())
        records: List[Dict[str, Any]] = []

        # Relations observed per (head type, tail type), used to keep
        # relation swaps schema-consistent when entity types are available.
        pair_relations: Dict[Tuple[Any, Any], set] = {}
        if self.same_type:
            for a, b, _, d in iter_edges(G):
                ta, tb = G.nodes[a].get(NODE_TYPE_ATTR), G.nodes[b].get(NODE_TYPE_ATTR)
                if ta is not None and tb is not None and EDGE_RELATION_ATTR in d:
                    pair_relations.setdefault((ta, tb), set()).add(str(d[EDGE_RELATION_ATTR]))

        for u, v, key, attrs in chosen:
            mode = self.mode
            if mode == "any":
                mode = rng.choice(["relation", "head", "tail"])

            new_u, new_v, new_attrs = u, v, dict(attrs)
            if mode == "relation":
                current = str(attrs.get(EDGE_RELATION_ATTR))
                options = [
                    r for r in vocab
                    if r != current and not self._triple_exists(G, u, v, r)
                ]
                type_pair = (G.nodes[u].get(NODE_TYPE_ATTR), G.nodes[v].get(NODE_TYPE_ATTR))
                consistent = [r for r in options if r in pair_relations.get(type_pair, ())]
                if consistent:
                    options = consistent
                if not options:
                    continue
                new_attrs[EDGE_RELATION_ATTR] = rng.choice(options)
            else:
                endpoint, other = (u, v) if mode == "head" else (v, u)
                pool = [
                    x for x in self._candidate_nodes(G, endpoint, nodes)
                    if x != endpoint and x != other
                ]
                relation = attrs.get(EDGE_RELATION_ATTR)
                pool = [
                    x for x in pool
                    if not self._triple_exists(G, *((x, v) if mode == "head" else (u, x)), relation)
                ]
                if not pool:
                    continue
                x = rng.choice(pool)
                if mode == "head":
                    new_u = x
                else:
                    new_v = x

            if not has_edge(G, u, v, key):
                continue
            remove_edge(G, u, v, key)
            new_key = add_edge(G, new_u, new_v, **new_attrs)

            rec: Dict[str, Any] = {
                "u": u, "v": v, "attrs": attrs,
                "new_u": new_u, "new_v": new_v, "new_attrs": new_attrs,
            }
            if G.is_multigraph():
                rec["key"] = key
                rec["new_key"] = new_key
            records.append(rec)

        return G, {"corrupted_triples": records}
