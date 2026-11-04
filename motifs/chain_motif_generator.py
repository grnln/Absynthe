import networkx as nx
from interfaces import MotifGenerator
from motifs.utils import assign_labels_to_motif


class ChainMotifGenerator(MotifGenerator):
    """Generator for chain (path) motifs.

    Generates a simple path graph with `length` nodes, starting from index `start`.
    """

    def generate_motif(self, start: int, length: int, id: int = 0) -> nx.Graph:
        """Generate a chain motif (path) with `length` nodes starting from `start`.

        Args:
            start (int): Starting index for the nodes.
            length (int): Number of nodes in the chain. Must be >= 2.
            id (int): Identifier for the motif.

        Returns:
            nx.Graph: The chain graph.

        Raises:
            ValueError: If `length` < 2.
        """
        if length < 2:
            raise ValueError("length must be at least 2 for a chain motif.")

        graph = nx.Graph()
        graph.add_nodes_from(range(start, start + length))
        for i in range(length - 1):
            graph.add_edge(start + i, start + i + 1)

        # Add motif attributes to each node
        for node in graph.nodes():
            graph.nodes[node]['motif'] = f'chain_{length}'
            graph.nodes[node]['motif_id'] = f'chain_{length}_{id}'

        return graph

    @staticmethod
    def assign_labels(graph, length=3):
        """
        Search for subgraphs shaped like a "chain" (path) of `length` nodes and label their nodes.
        """
        motif = ChainMotifGenerator().generate_motif(0, length)
        return assign_labels_to_motif(graph, motif, f'chain_{length}')
