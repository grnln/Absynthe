import networkx as nx
from interfaces import MotifGenerator
from motifs.utils import assign_labels_to_motif


class GridMotifGenerator(MotifGenerator):
    """Generator for grid (lattice) motifs.

    A grid motif is a ``rows`` x ``cols`` rectangular lattice: every node is
    connected to its horizontal and vertical neighbours. Node ``(r, c)`` gets
    the index ``start + r * cols + c``, so the lattice is laid out row by row::

        0 -- 1 -- 2
        |    |    |
        3 -- 4 -- 5

    Total nodes = rows * cols; total edges = rows*(cols-1) + cols*(rows-1).
    """

    def generate_motif(self, start: int, rows: int, cols: int, id: int = 0) -> nx.Graph:
        """Generate a ``rows`` x ``cols`` grid motif starting at ``start``.

        Args:
            start (int): Starting index for the nodes.
            rows (int): Number of rows. Must be >= 2.
            cols (int): Number of columns. Must be >= 2.
            id (int): Identifier for the motif.

        Returns:
            nx.Graph: The grid graph.

        Raises:
            ValueError: If ``rows`` < 2 or ``cols`` < 2 (a 1 x n grid is a chain).
        """
        if rows < 2 or cols < 2:
            raise ValueError("rows and cols must both be at least 2 for a grid motif.")

        graph = nx.Graph()
        graph.add_nodes_from(range(start, start + rows * cols))

        def node_at(r: int, c: int) -> int:
            return start + r * cols + c

        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    graph.add_edge(node_at(r, c), node_at(r, c + 1))
                if r + 1 < rows:
                    graph.add_edge(node_at(r, c), node_at(r + 1, c))

        motif_name = f'grid_{rows}_{cols}'
        for node in graph.nodes():
            graph.nodes[node]['motif'] = motif_name
            graph.nodes[node]['motif_id'] = f'{motif_name}_{id}'

        return graph

    @staticmethod
    def assign_labels(graph, rows=3, cols=3):
        """
        Search for subgraphs shaped like a ``rows`` x ``cols`` grid and label their nodes.
        """
        motif = GridMotifGenerator().generate_motif(0, rows, cols)
        return assign_labels_to_motif(graph, motif, f'grid_{rows}_{cols}')
