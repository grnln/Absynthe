"""Export formats for the graphs of a dataset.

``GraphDatasetGenerator`` writes every graph through a list of
``GraphExporter`` instances. The first one is the *primary* format: its file
is the ``graph_path`` of the variant in ``metadata.json`` and must be
readable by ``graph.folder_graph_generator.load_graph_file`` so variants can
be reloaded and reconstructed. Every other exporter that writes a file adds
``<name>_path`` / ``original_<name>_path`` to the metadata entry.

Built-in formats::

    "graphml"   GraphML (always applies; the default primary format)
    "triples"   subject/relation/object triples for knowledge graphs only:
                ``.nt`` (N-Triples) when node ids are IRIs, ``.tsv`` otherwise

Defining a new format is a small subclass registered by name::

    import networkx as nx
    from interfaces import GraphExporter
    from graph.exporters import register_exporter

    @register_exporter
    class EdgeListExporter(GraphExporter):
        name = "edgelist"

        def export(self, graph, base_path):
            path = base_path + ".edgelist"
            nx.write_edgelist(graph, path, data=False)
            return path

    GraphDatasetGenerator(..., formats=["graphml", "edgelist"])

``formats`` accepts registered names or exporter instances (useful for
parametrised exporters).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import networkx as nx

from interfaces import GraphExporter, GraphLike
from utils.kg_utils import is_kg, looks_like_iri, stamp_graph_kind, write_triples

# Registry of export formats by name (extend it with ``register_exporter``).
EXPORTERS: Dict[str, Type[GraphExporter]] = {}

DEFAULT_FORMATS: Tuple[str, ...] = ("graphml", "triples")

# ``graph`` would produce the key ``graph_path``, which belongs to the primary format.
_RESERVED_NAMES = {"graph"}

FormatSpec = Union[str, GraphExporter]


def register_exporter(cls: Type[GraphExporter]) -> Type[GraphExporter]:
    """Class decorator that adds ``cls`` to ``EXPORTERS`` under ``cls.name``."""
    name = getattr(cls, "name", "")
    if not name or not isinstance(name, str):
        raise ValueError(f"{cls.__name__} must define a non-empty string 'name'")
    if name in _RESERVED_NAMES:
        raise ValueError(f"Exporter name '{name}' is reserved")
    EXPORTERS[name] = cls
    return cls


@register_exporter
class GraphMLExporter(GraphExporter):
    """GraphML through NetworkX; multigraphs are stamped so they reload as such."""

    name = "graphml"

    def export(self, graph: GraphLike, base_path: str) -> Optional[str]:
        stamp_graph_kind(graph)
        path = base_path + ".graphml"
        nx.write_graphml(graph, path)
        return path


@register_exporter
class TriplesExporter(GraphExporter):
    """Triples of a knowledge graph: N-Triples for IRI node ids, TSV otherwise.

    Graphs not flagged as knowledge graphs (``graph.graph["kg"]``) are skipped.
    """

    name = "triples"

    def export(self, graph: GraphLike, base_path: str) -> Optional[str]:
        if not is_kg(graph):
            return None
        if graph.number_of_nodes() > 0 and all(looks_like_iri(n) for n in graph.nodes()):
            path = base_path + ".nt"
            write_triples(graph, path, fmt="nt")
        else:
            path = base_path + ".tsv"
            write_triples(graph, path, fmt="tsv")
        return path


def resolve_exporters(formats: Optional[Iterable[FormatSpec]] = None) -> List[GraphExporter]:
    """Turn ``formats`` (names and/or exporter instances) into exporter instances.

    ``None`` selects ``DEFAULT_FORMATS``. Raises ``ValueError`` for an unknown
    name, an empty list or duplicate format names.
    """
    specs = list(DEFAULT_FORMATS if formats is None else formats)
    if not specs:
        raise ValueError("At least one export format is required")

    exporters: List[GraphExporter] = []
    for spec in specs:
        if isinstance(spec, GraphExporter):
            exporters.append(spec)
        elif isinstance(spec, str):
            if spec not in EXPORTERS:
                raise ValueError(
                    f"Unknown export format '{spec}'. Registered formats: {sorted(EXPORTERS)}"
                )
            exporters.append(EXPORTERS[spec]())
        else:
            raise TypeError(f"Export format must be a name or a GraphExporter, got {type(spec).__name__}")

    names = [e.name for e in exporters]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate export formats: {names}")
    return exporters


def export_graph(graph: GraphLike, base_path: str, exporters: Sequence[GraphExporter]) -> Dict[str, str]:
    """Write ``graph`` with every exporter; return ``{name: path}`` for those that wrote.

    The first exporter is the primary format and must write a file.
    """
    files: Dict[str, str] = {}
    for exporter in exporters:
        path = exporter.export(graph, base_path)
        if path is not None:
            files[exporter.name] = path
    primary = exporters[0].name
    if primary not in files:
        raise ValueError(f"Primary export format '{primary}' wrote nothing for {base_path}")
    return files
