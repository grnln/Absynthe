import abc
from typing import Optional

from .graph_types import GraphLike


class GraphExporter(abc.ABC):
    """Abstract base class for writing a graph to disk in one file format.

    ``name`` identifies the format: it is the key in the exporter registry
    (``graph.exporters.EXPORTERS``), the value used in ``formats=[...]`` and
    the prefix of the ``<name>_path`` keys written to ``metadata.json``.

    Built-in exporters live in ``graph.exporters``; a new format is a
    subclass with a ``name`` and an ``export`` method, registered with
    ``graph.exporters.register_exporter``.
    """

    name: str = ""

    @abc.abstractmethod
    def export(self, graph: GraphLike, base_path: str) -> Optional[str]:
        """Write ``graph`` next to ``base_path`` and return the written path.

        Args:
            graph: The graph to write (any NetworkX graph class).
            base_path: Destination path WITHOUT extension. The exporter
                appends the extension of its format.

        Returns:
            The path of the file written, or ``None`` to skip graphs the
            format does not apply to (e.g. triples for graphs that are not
            knowledge graphs).
        """
        pass
