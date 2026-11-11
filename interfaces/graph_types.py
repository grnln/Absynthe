"""Graph type aliases shared by the framework interfaces.

Absynthe components accept any of the four NetworkX graph classes. Plain
synthetic graphs are undirected ``nx.Graph`` instances; knowledge graphs are
typically ``nx.MultiDiGraph`` instances whose nodes carry a ``type`` attribute
and whose edges carry a ``relation`` attribute (see ``utils.kg_utils``).
"""
from typing import Any, Tuple, Union

import networkx as nx

GraphLike = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


def ordered_pair(u: Any, v: Any) -> Tuple[Any, Any]:
    """Return ``(u, v)`` ordered canonically for undirected edges.

    Falls back to string comparison when node ids are not mutually
    comparable (e.g. a mix of ints and IRIs).
    """
    try:
        return (u, v) if u <= v else (v, u)
    except TypeError:
        return (u, v) if str(u) <= str(v) else (v, u)
