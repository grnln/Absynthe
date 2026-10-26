import networkx as nx

from typing import Any
from interfaces import MotifGenerator
from motifs.utils import assign_labels_to_motif

class CycleMotifGenerator(MotifGenerator):
    """Generator for cycle motifs.

    Generates a cycle graph with len_cycle nodes, starting from index start.
    """

    def generate_motif(self, start: int, len_cycle: int, id: int = 0) -> nx.Graph[Any]:
        """Generate a cycle motif with len_cycle nodes starting from index start.

        Args:
            start (int): Starting index for the nodes.
            len_cycle (int): Number of nodes in the cycle. Must be >= 3.
            id (int): Identifier for the motif.

        Returns:
            nx.Graph: The cycle graph.

        Raises:
            ValueError: If len_cycle < 3.
        """
        if len_cycle < 3:
            raise ValueError("len_cycle must be at least 3 for a cycle.")

        graph = nx.Graph()
        graph.add_nodes_from(range(start, start + len_cycle))
        for i in range(len_cycle - 1):
            graph.add_edges_from([(start + i, start + i + 1)])
        graph.add_edges_from([(start + len_cycle - 1, start)])
        # Add motif attribute to each node
        for node in graph.nodes():
            graph.nodes[node]['motif'] = f'cycle_{len_cycle}'
            graph.nodes[node]['motif_id'] = f'cycle_{len_cycle}_{id}'
        return graph

    @staticmethod
    def assign_labels(graph, len_cycle=3):
        """
        Search for subgraphs shaped like a “cycle” and label their nodes.
        len_cycle: Length of the cycle to search for and label (default: 3).
        """
        motif = CycleMotifGenerator().generate_motif(0, len_cycle)
        return assign_labels_to_motif(graph, motif, f'cycle_{len_cycle}')