import logging
import os
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx

from interfaces import GraphGenerator, GraphLike
from interfaces.exceptions import GraphSourceExhausted
from utils.rng import get_rng
from utils.kg_utils import (
    EDGE_PREDICATE_ATTR,
    EDGE_RELATION_ATTR,
    KG_FLAG,
    MULTIGRAPH_FLAG,
    NODE_TYPE_ATTR,
    NODE_TYPES_ATTR,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".graphml", ".rdf", ".ttl", ".nt", ".n3", ".owl")

_RDF_FORMAT_BY_EXT = {
    ".rdf": "xml",
    ".owl": "xml",
    ".ttl": "turtle",
    ".nt": "nt",
    ".n3": "n3",
}

RDF_TYPE_IRI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def _make_graph(directed: bool, multigraph: bool) -> GraphLike:
    if directed:
        return nx.MultiDiGraph() if multigraph else nx.DiGraph()
    return nx.MultiGraph() if multigraph else nx.Graph()


def _rdf_to_nx(
    path: str,
    rdf_format: str,
    *,
    directed: bool = True,
    multigraph: bool = True,
    type_predicates: Iterable[str] = (RDF_TYPE_IRI,),
    short_names: bool = True,
) -> GraphLike:
    """Load an RDF file and project it onto a NetworkX knowledge graph.

    Mapping (see ``utils.kg_utils`` for the attribute conventions):

    * ``(s, p, o)`` with an IRI / blank-node object becomes a directed edge
      ``s -> o`` with ``relation`` (short predicate name, e.g. ``foaf:knows``)
      and ``predicate`` (full IRI). Parallel predicates between the same
      pair are kept as parallel edges (``multigraph=True``).
    * ``(s, rdf:type, C)`` sets the ``type`` node attribute of ``s`` to the
      short name of ``C`` (first in sorted order when several) and ``types``
      to all class names joined by ``"|"``. No edge is created.
    * ``(s, p, literal)`` becomes a node attribute of ``s`` keyed by the
      short predicate name.

    Triples are processed in sorted order so that node insertion order and
    edge keys are deterministic for a given file. The result is flagged with
    ``graph.graph["kg"] = True``.
    """
    import rdflib

    rdf_graph = rdflib.Graph()
    rdf_graph.parse(path, format=rdf_format)
    nsm = rdf_graph.namespace_manager
    type_predicates = {str(p) for p in type_predicates}

    short_cache: Dict[str, str] = {}
    used_short: Dict[str, str] = {}

    def short(iri: Any) -> str:
        s = str(iri)
        if not short_names:
            return s
        if s in short_cache:
            return short_cache[s]
        try:
            prefix, _, name = nsm.compute_qname(iri, generate=True)
            candidate = f"{prefix}:{name}" if prefix else name
        except Exception:  # noqa: BLE001 - rdflib raises plain Exception on unsplittable IRIs
            candidate = s
        # Two different IRIs must not collapse onto the same short name.
        if used_short.get(candidate, s) != s:
            candidate = s
        used_short[candidate] = s
        short_cache[s] = candidate
        return candidate

    g = _make_graph(directed, multigraph)
    g.graph[KG_FLAG] = True
    types: Dict[str, set] = {}

    for s, p, o in sorted(rdf_graph):
        s_id = str(s)
        p_str = str(p)
        if not g.has_node(s_id):
            g.add_node(s_id)
        if p_str in type_predicates and not isinstance(o, rdflib.Literal):
            types.setdefault(s_id, set()).add(short(o))
        elif isinstance(o, rdflib.Literal):
            g.nodes[s_id][short(p)] = str(o)
        else:
            o_id = str(o)
            if not g.has_node(o_id):
                g.add_node(o_id)
            g.add_edge(s_id, o_id, **{EDGE_RELATION_ATTR: short(p), EDGE_PREDICATE_ATTR: p_str})

    for node, classes in types.items():
        ordered = sorted(classes)
        g.nodes[node][NODE_TYPE_ATTR] = ordered[0]
        g.nodes[node][NODE_TYPES_ATTR] = "|".join(ordered)
    return g


def load_graph_file(
    path: str,
    *,
    as_undirected: bool = False,
    rdf_options: Optional[Dict[str, Any]] = None,
) -> GraphLike:
    """Load a graph file (GraphML or RDF) and return a NetworkX graph.

    GraphML files keep their directedness; files written by
    ``GraphDatasetGenerator`` from a multigraph are reloaded as multigraphs
    (via the ``multigraph`` graph attribute). RDF files become directed
    multi-relational knowledge graphs (see :func:`_rdf_to_nx`).

    Args:
        path: File to load.
        as_undirected: Collapse the result onto a simple undirected
            ``nx.Graph`` (the historical behaviour). Direction, parallel
            relations and edge keys are lost.
        rdf_options: Extra keyword arguments for :func:`_rdf_to_nx`
            (``directed``, ``multigraph``, ``type_predicates``, ``short_names``).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".graphml":
        graph = nx.read_graphml(path)
        if graph.graph.get(MULTIGRAPH_FLAG) and not graph.is_multigraph():
            graph = nx.read_graphml(path, force_multigraph=True)
    else:
        rdf_format = _RDF_FORMAT_BY_EXT.get(ext)
        if rdf_format is None:
            raise ValueError(f"Unsupported graph file extension: {ext}")
        options = dict(rdf_options or {})
        if as_undirected:
            options.update(directed=False, multigraph=False)
        graph = _rdf_to_nx(path, rdf_format, **options)

    if as_undirected and (graph.is_directed() or graph.is_multigraph()):
        graph = nx.Graph(graph)
    return graph


class IterationOrder(Enum):
    SEQUENTIAL = "sequential"
    RANDOM = "random"


class ExhaustionPolicy(Enum):
    STOP = "stop"
    RAISE = "raise"
    CYCLE = "cycle"


class FolderGraphGenerator(GraphGenerator):
    """GraphGenerator that loads pre-existing graphs from a folder.

    Supports GraphML files (plain graphs, directed graphs and multigraphs)
    and RDF files (``.rdf``, ``.owl``, ``.ttl``, ``.nt``, ``.n3``), which are
    loaded as directed multi-relational knowledge graphs.
    """

    def __init__(
        self,
        folder_path: str,
        iteration_order: IterationOrder = IterationOrder.SEQUENTIAL,
        exhaustion_policy: ExhaustionPolicy = ExhaustionPolicy.STOP,
        as_undirected: bool = False,
        rdf_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            folder_path: Directory containing the graph files (non-recursive).
            iteration_order: Sequential (alphabetical) or random order.
            exhaustion_policy: What to do when every file has been served.
            as_undirected: Collapse every loaded graph onto a simple
                undirected ``nx.Graph`` (historical behaviour).
            rdf_options: Options forwarded to the RDF loader (see
                :func:`load_graph_file`).
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        self.folder_path = folder_path
        self.iteration_order = iteration_order
        self.exhaustion_policy = exhaustion_policy
        self.as_undirected = as_undirected
        self.rdf_options = rdf_options

        # Collect supported graph files (non-recursive), sorted alphabetically
        all_files = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        )
        if not all_files:
            raise ValueError(
                f"No supported graph files found in: {folder_path} "
                f"(extensions: {', '.join(SUPPORTED_EXTENSIONS)})"
            )

        self._files: List[str] = [
            os.path.join(folder_path, f) for f in all_files
        ]
        self._total = len(self._files)
        self._order: List[int] = []
        self._index = 0
        self._graphs_served = 0

        self._reset_order()

        logger.info("Loaded %d graphs from folder %s", self._total, folder_path)

    def _reset_order(self) -> None:
        """Initialize or re-shuffle the iteration order."""
        if self.iteration_order == IterationOrder.RANDOM:
            self._order = list(range(self._total))
            get_rng().shuffle(self._order)
        else:
            self._order = list(range(self._total))
        self._index = 0

    def generate_graph(self, **kwargs) -> GraphLike:
        if self._index >= self._total:
            if self.exhaustion_policy == ExhaustionPolicy.CYCLE:
                self._reset_order()
            else:
                raise GraphSourceExhausted(
                    loaded=self._total,
                    requested=self._graphs_served + 1,
                )

        file_path = self._files[self._order[self._index]]
        self._index += 1
        self._graphs_served += 1

        return load_graph_file(
            file_path,
            as_undirected=self.as_undirected,
            rdf_options=self.rdf_options,
        )
