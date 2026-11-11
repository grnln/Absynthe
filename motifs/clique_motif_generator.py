import itertools

import networkx as nx
from interfaces import MotifGenerator
from motifs.utils import assign_labels_to_motif


class CliqueMotifGenerator(MotifGenerator):
    """Generator for clique (complete graph) motifs.

    A clique motif is a complete graph K_n: every pair of its ``num_nodes``
    nodes is connected by an edge. Total edges = n * (n - 1) / 2.
    """

    def generate_motif(self, start: int, num_nodes: int, id: int = 0) -> nx.Graph:
        """Generate a clique motif with ``num_nodes`` nodes starting at ``start``.

        Args:
            start (int): Starting index for the nodes.
            num_nodes (int): Number of nodes in the clique. Must be >= 2.
            id (int): Identifier for the motif.

        Returns:
            nx.Graph: The complete graph.

        Raises:
            ValueError: If ``num_nodes`` < 2.
        """
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2 for a clique motif.")

        graph = nx.Graph()
        nodes = list(range(start, start + num_nodes))
        graph.add_nodes_from(nodes)
        graph.add_edges_from(itertools.combinations(nodes, 2))

        motif_name = f'clique_{num_nodes}'
        for node in graph.nodes():
            graph.nodes[node]['motif'] = motif_name
            graph.nodes[node]['motif_id'] = f'{motif_name}_{id}'

        return graph

    @staticmethod
    def assign_labels(graph, num_nodes=4):
        """
        Search for subgraphs shaped like a clique of ``num_nodes`` nodes and label their nodes.
        """
        motif = CliqueMotifGenerator().generate_motif(0, num_nodes)
        return assign_labels_to_motif(graph, motif, f'clique_{num_nodes}')
