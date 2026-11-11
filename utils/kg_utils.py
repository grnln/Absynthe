"""Helpers that make the pipeline agnostic to the NetworkX graph class.

Absynthe accepts plain undirected graphs (``nx.Graph``) as well as directed
and multi-relational knowledge graphs (``nx.DiGraph``, ``nx.MultiGraph``,
``nx.MultiDiGraph``). This module centralises the few operations whose API
differs between those classes (edge iteration, edge keys, incident edges) and
defines the attribute conventions used for knowledge graphs:

===================  ==============  ==================================================
Where                Name            Meaning
===================  ==============  ==================================================
node attribute       ``type``        entity type (string)
node attribute       ``types``       all entity types, ``"|"``-joined (RDF loader only)
edge attribute       ``relation``    relation type (short name, e.g. ``foaf:knows``)
edge attribute       ``predicate``   full predicate IRI (RDF loader only)
``graph.graph``      ``kg``          True when the graph is a knowledge graph
``graph.graph``      ``multigraph``  stamped on save so GraphML reloads as a multigraph
===================  ==============  ==================================================

Edge references
---------------
Several parts of the framework need a hashable reference to a single edge
(labeling results, perturbation hints, metadata). :func:`edge_key` builds it:

* simple undirected graph: ``(min(u, v), max(u, v))``
* simple directed graph:   ``(u, v)``
* multigraphs:             the pair above plus the edge key, ``(u, v, key)``

Reversible change records (``{"u", "v", "attrs"}``) gain a ``"key"`` entry
only on multigraphs, so the metadata of plain datasets is unchanged.
"""
from typing import Any, Dict, Iterator, List, Optional, Tuple

import networkx as nx

from interfaces.graph_types import GraphLike, ordered_pair

NODE_TYPE_ATTR = "type"
NODE_TYPES_ATTR = "types"
EDGE_RELATION_ATTR = "relation"
EDGE_PREDICATE_ATTR = "predicate"
KG_FLAG = "kg"
MULTIGRAPH_FLAG = "multigraph"

EdgeRef = Tuple  # (u, v) or (u, v, key)


# ---------------------------------------------------------------------------
# Graph kind
# ---------------------------------------------------------------------------

def is_kg(G: GraphLike) -> bool:
    """True if the graph was flagged as a knowledge graph by its source."""
    return bool(G.graph.get(KG_FLAG, False))


def graph_kind(G: GraphLike) -> Dict[str, bool]:
    return {
        "directed": bool(G.is_directed()),
        "multigraph": bool(G.is_multigraph()),
        "kg": is_kg(G),
    }


def is_plain(G: GraphLike) -> bool:
    """True for the historical case: undirected, simple, not a KG."""
    return not (G.is_directed() or G.is_multigraph() or is_kg(G))


def stamp_graph_kind(G: GraphLike) -> None:
    """Record on ``G.graph`` what GraphML cannot express by itself.

    GraphML stores directedness natively, but a multigraph without parallel
    edges is read back as a simple graph. The ``multigraph`` flag lets
    ``load_graph_file`` restore the right class.
    """
    if G.is_multigraph():
        G.graph[MULTIGRAPH_FLAG] = True


# ---------------------------------------------------------------------------
# Edge references
# ---------------------------------------------------------------------------

def edge_key(G: GraphLike, u: Any, v: Any, key: Optional[Any] = None) -> EdgeRef:
    """Canonical hashable reference of an edge of ``G`` (see module docstring)."""
    pair = (u, v) if G.is_directed() else ordered_pair(u, v)
    if G.is_multigraph():
        return pair + (key,)
    return pair


def unpack_ref(ref: EdgeRef) -> Tuple[Any, Any, Optional[Any]]:
    """Split an edge reference (2- or 3-tuple, or list) into ``(u, v, key)``."""
    if len(ref) >= 3:
        return ref[0], ref[1], ref[2]
    return ref[0], ref[1], None


# ---------------------------------------------------------------------------
# Uniform edge access
# ---------------------------------------------------------------------------

def iter_edges(G: GraphLike, nbunch: Any = None) -> Iterator[Tuple[Any, Any, Optional[Any], Dict]]:
    """Yield ``(u, v, key, attrs)`` for every edge, ``key`` being ``None`` on simple graphs.

    The iteration order is exactly that of ``G.edges()`` so that callers
    consume the random number generator in the same sequence as before.
    """
    if G.is_multigraph():
        for u, v, k, d in G.edges(nbunch, keys=True, data=True):
            yield u, v, k, d
    else:
        for u, v, d in G.edges(nbunch, data=True):
            yield u, v, None, d


def has_edge(G: GraphLike, u: Any, v: Any, key: Optional[Any] = None) -> bool:
    if G.is_multigraph() and key is not None:
        return G.has_edge(u, v, key)
    return G.has_edge(u, v)


def edge_attrs(G: GraphLike, u: Any, v: Any, key: Optional[Any] = None) -> Dict[str, Any]:
    """Attribute dict of one edge (the first parallel edge when ``key`` is None)."""
    if G.is_multigraph():
        if key is None:
            key = next(iter(G[u][v]))
        return G.edges[u, v, key]
    return G.edges[u, v]


def remove_edge(G: GraphLike, u: Any, v: Any, key: Optional[Any] = None) -> None:
    if G.is_multigraph():
        G.remove_edge(u, v, key=key)
    else:
        G.remove_edge(u, v)


def add_edge(G: GraphLike, u: Any, v: Any, key: Optional[Any] = None, **attrs) -> Optional[Any]:
    """Add an edge and return the key actually used (``None`` on simple graphs)."""
    if G.is_multigraph():
        return G.add_edge(u, v, key=key, **attrs)
    G.add_edge(u, v, **attrs)
    return None


def incident_edges(G: GraphLike, n: Any) -> List[Tuple[Any, Any, Optional[Any], Dict]]:
    """All edges touching ``n`` as ``(u, v, key, attrs)`` with the real endpoint order.

    Undirected graphs yield ``(n, neighbour, ...)`` in adjacency order (the
    same order as ``G.neighbors(n)``); directed graphs yield out-edges then
    in-edges, so a removed node can be restored with its exact edges.
    """
    if not G.is_directed():
        return list(iter_edges(G, [n]))
    result = []
    if G.is_multigraph():
        result.extend((u, v, k, d) for u, v, k, d in G.out_edges(n, keys=True, data=True))
        result.extend((u, v, k, d) for u, v, k, d in G.in_edges(n, keys=True, data=True))
    else:
        result.extend((u, v, None, d) for u, v, d in G.out_edges(n, data=True))
        result.extend((u, v, None, d) for u, v, d in G.in_edges(n, data=True))
    return result


def edge_record(G: GraphLike, u: Any, v: Any, key: Optional[Any] = None,
                attrs: Optional[Dict[str, Any]] = None,
                with_attrs: bool = True) -> Dict[str, Any]:
    """Reversible change record for an edge: ``{"u", "v"[, "key"][, "attrs"]}``.

    ``key`` is only included on multigraphs. ``attrs`` defaults to the edge's
    current attributes; pass ``with_attrs=False`` to omit them.
    """
    rec: Dict[str, Any] = {"u": u, "v": v}
    if G.is_multigraph():
        rec["key"] = key
    if with_attrs:
        rec["attrs"] = dict(edge_attrs(G, u, v, key) if attrs is None else attrs)
    return rec


def non_edges(G: GraphLike) -> Iterator[Tuple[Any, Any]]:
    """Node pairs without an edge, honouring direction (wraps ``nx.non_edges``)."""
    return nx.non_edges(G)


# ---------------------------------------------------------------------------
# Vocabulary / statistics
# ---------------------------------------------------------------------------

def relation_vocab(G: GraphLike, attr: str = EDGE_RELATION_ATTR) -> List[str]:
    return sorted({str(d[attr]) for _, _, _, d in iter_edges(G) if attr in d})


def relation_counts(G: GraphLike, attr: str = EDGE_RELATION_ATTR) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, _, _, d in iter_edges(G):
        if attr in d:
            rel = str(d[attr])
            counts[rel] = counts.get(rel, 0) + 1
    return counts


def entity_type_counts(G: GraphLike, attr: str = NODE_TYPE_ATTR) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, d in G.nodes(data=True):
        if attr in d:
            t = str(d[attr])
            counts[t] = counts.get(t, 0) + 1
    return counts


def to_multigraph(G: GraphLike) -> GraphLike:
    """Return a multigraph copy of ``G`` (no-op copy if it already is one).

    An ``id`` edge attribute left behind by ``nx.read_graphml`` becomes the
    edge key.
    """
    if G.is_multigraph():
        return G.copy()
    M = nx.MultiDiGraph() if G.is_directed() else nx.MultiGraph()
    M.graph.update(G.graph)
    M.add_nodes_from(G.nodes(data=True))
    for u, v, _, d in iter_edges(G):
        d = dict(d)
        key = d.pop("id", None)
        M.add_edge(u, v, key=key, **d)
    return M


# ---------------------------------------------------------------------------
# Triples
# ---------------------------------------------------------------------------

def looks_like_iri(x: Any) -> bool:
    s = str(x)
    return "://" in s or s.startswith("urn:")


def graph_to_triples(G: GraphLike, relation_attr: str = EDGE_RELATION_ATTR,
                     default_relation: str = "edge") -> List[Tuple[str, str, str]]:
    """Flatten the graph into ``(subject, relation, object)`` string triples."""
    triples = []
    for u, v, _, d in iter_edges(G):
        triples.append((str(u), str(d.get(relation_attr, default_relation)), str(v)))
    return triples


def write_triples(G: GraphLike, path: str, fmt: str = "tsv",
                  base_iri: str = "urn:absynthe:",
                  relation_attr: str = EDGE_RELATION_ATTR,
                  default_relation: str = "edge") -> None:
    """Write the graph's edges as triples.

    ``fmt="tsv"`` writes one ``subject<TAB>relation<TAB>object`` line per
    edge (the format used by link-prediction benchmarks). ``fmt="nt"`` writes
    N-Triples through rdflib; identifiers that are not IRIs are prefixed with
    ``base_iri``, and the full ``predicate`` IRI is used when present.
    """
    if fmt == "tsv":
        with open(path, "w", encoding="utf-8") as f:
            for s, p, o in graph_to_triples(G, relation_attr, default_relation):
                f.write(f"{s}\t{p}\t{o}\n")
        return
    if fmt != "nt":
        raise ValueError(f"Unsupported triple format: {fmt}")

    import rdflib

    def to_iri(x: Any, prefix: str = "") -> rdflib.URIRef:
        s = str(x)
        return rdflib.URIRef(s if looks_like_iri(s) else base_iri + prefix + s)

    rg = rdflib.Graph()
    for u, v, _, d in iter_edges(G):
        if EDGE_PREDICATE_ATTR in d:
            pred = rdflib.URIRef(str(d[EDGE_PREDICATE_ATTR]))
        else:
            pred = to_iri(d.get(relation_attr, default_relation), prefix="rel/")
        rg.add((to_iri(u), pred, to_iri(v)))
    rg.serialize(destination=path, format="nt", encoding="utf-8")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def apply_labeling_result(G: GraphLike, result, attribute_name: str) -> None:
    """Store a ``LabelingResult`` on the graph in place.

    Node labels go to ``attribute_name`` (e.g. ``expected_ground_truth``) and
    to ``label``; edge labels go to the ``label`` attribute of the referenced
    edge (``(u, v)`` or ``(u, v, key)``); graph labels go to ``G.graph``.
    """
    for node, label in result.node_labels.items():
        if node in G:
            G.nodes[node][attribute_name] = label
            G.nodes[node]["label"] = label
    for ref, label in result.edge_labels.items():
        u, v, k = unpack_ref(ref)
        if has_edge(G, u, v, k):
            edge_attrs(G, u, v, k)["label"] = label
    for key, value in result.graph_labels.items():
        G.graph[key] = value
