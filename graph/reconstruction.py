from typing import Any, Dict, Optional

from interfaces import GraphLike
from utils.kg_utils import add_edge, has_edge, remove_edge


def _coerce_key(G: GraphLike, u: Any, v: Any, key: Any) -> Any:
    """Match a stored edge key against the keys actually present in ``G``.

    Keys reloaded from JSON/GraphML may have changed type (``1`` vs ``"1"``).
    """
    if key is None or not G.is_multigraph() or not G.has_edge(u, v):
        return key
    existing = G[u][v]
    if key in existing:
        return key
    for candidate in (str(key),):
        if candidate in existing:
            return candidate
    try:
        as_int = int(key)
        if as_int in existing:
            return as_int
    except (TypeError, ValueError):
        pass
    return key


def reconstruct_original(perturbed: GraphLike, changes: Dict[str, Any]) -> GraphLike:
    """Invert a perturbation to recover the original graph from a perturbed variant.

    Expects the reversible ``changes`` dict produced by the perturbations in
    ``graph.perturbations`` / ``graph.kg_perturbations``:

        {
            "removed_nodes":     [{"id", "attrs", "edges": [{"u", "v"[, "key"], "attrs"}]}],
            "removed_edges":     [{"u", "v"[, "key"], "attrs"}],
            "added_edges":       [{"u", "v"[, "key"][, "attrs"]}],
            "corrupted_triples": [{"u", "v"[, "key"], "attrs",
                                   "new_u", "new_v"[, "new_key"], "new_attrs"}],
        }

    ``key`` entries are only present for multigraphs. Works on any NetworkX
    graph class (direction and parallel edges are preserved).

    Note: attributes reloaded from GraphML arrive as strings; this function
    does not reinterpret their types — it round-trips whatever the variant
    file stores. See :func:`normalize_changes` for node-id coercion.
    """
    G = perturbed.copy()

    for e in changes.get("added_edges", []):
        key = _coerce_key(G, e["u"], e["v"], e.get("key"))
        if has_edge(G, e["u"], e["v"], key):
            remove_edge(G, e["u"], e["v"], key)

    for t in changes.get("corrupted_triples", []):
        key = _coerce_key(G, t["new_u"], t["new_v"], t.get("new_key"))
        if has_edge(G, t["new_u"], t["new_v"], key):
            remove_edge(G, t["new_u"], t["new_v"], key)
        add_edge(G, t["u"], t["v"], key=t.get("key"), **t.get("attrs", {}))

    for e in changes.get("removed_edges", []):
        add_edge(G, e["u"], e["v"], key=e.get("key"), **e.get("attrs", {}))

    for n in changes.get("removed_nodes", []):
        G.add_node(n["id"], **n.get("attrs", {}))
        for e in n.get("edges", []):
            add_edge(G, e["u"], e["v"], key=e.get("key"), **e.get("attrs", {}))

    return G


def normalize_changes(changes: Dict[str, Any], graph: Optional[GraphLike] = None) -> Dict[str, Any]:
    """Align a ``changes`` dict loaded from ``metadata.json`` with a reloaded graph.

    GraphML stores node IDs as strings, while JSON keeps the original ints,
    so node references are coerced to ``str`` when the graph uses string
    ids (always, when ``graph`` is None). Edge keys and attributes are left
    untouched.
    """
    if graph is None or all(isinstance(n, str) for n in graph.nodes()):
        conv = str
    else:
        def conv(x):
            return x

    def edge(e: Dict[str, Any]) -> Dict[str, Any]:
        out = {"u": conv(e["u"]), "v": conv(e["v"])}
        if "key" in e:
            out["key"] = e["key"]
        if "attrs" in e:
            out["attrs"] = e.get("attrs", {})
        return out

    out: Dict[str, Any] = {}
    for n in changes.get("removed_nodes", []) or []:
        out.setdefault("removed_nodes", []).append({
            "id": conv(n["id"]),
            "attrs": n.get("attrs", {}),
            "edges": [edge(e) for e in n.get("edges", [])],
        })
    for e in changes.get("removed_edges", []) or []:
        out.setdefault("removed_edges", []).append(edge(e))
    for e in changes.get("added_edges", []) or []:
        out.setdefault("added_edges", []).append(edge(e))
    for t in changes.get("corrupted_triples", []) or []:
        rec = {
            "u": conv(t["u"]), "v": conv(t["v"]),
            "attrs": t.get("attrs", {}),
            "new_u": conv(t["new_u"]), "new_v": conv(t["new_v"]),
            "new_attrs": t.get("new_attrs", {}),
        }
        if "key" in t:
            rec["key"] = t["key"]
        if "new_key" in t:
            rec["new_key"] = t["new_key"]
        out.setdefault("corrupted_triples", []).append(rec)
    return out
