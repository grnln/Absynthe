import networkx as nx
from interfaces import GraphGenerator
from motifs import motif_generators, motif_param_names
from typing import Any
from collections import defaultdict
from utils.graph_utils import add_vertices, add_random_edges
from utils.rng import get_rng
from graph.composition_engine import compose_motifs


class MotifComposite(GraphGenerator):
    """Generator for composite graphs built from a list of motif specifications.

    This is one concrete implementation of GraphGenerator. It constructs
    graphs by instantiating structural motifs (cycles, houses, chains, etc.),
    composing them according to a pattern, and optionally adding extra
    vertices and edges.
    """

    def __init__(self, motifs: list[list[Any]] | None = None):
        self.motifs = motifs or []
        self.generators = motif_generators

    def generate_graph(self, num_extra_vertices: int = 0, num_extra_edges: int = 0, **kwargs: Any) -> nx.Graph[Any]:
        """Generate a composite graph by combining motifs from the list.

        Args:
            num_extra_vertices (int): Number of additional vertices to add.
            num_extra_edges (int): Number of additional edges to add randomly.
            **kwargs: Additional arguments (start, composition, composition_params).

        Returns:
            nx.Graph: The combined graph with extra vertices and edges.
        """
        combined_graph = nx.Graph()
        current_start = kwargs.get("start", 0)
        motif_counts = defaultdict(int)
        motif_node_sets = []

        for motif_list in self.motifs:
            base = motif_list[0]
            params = motif_list[1:]

            if base not in self.generators:
                raise ValueError(f"Unknown motif type: {base}")

            generator_class = self.generators[base]
            generator = generator_class()
            current_id = motif_counts[base]

            # Generic parameter dispatch using the motif_param_names registry
            param_names = motif_param_names.get(base, [])
            motif_kwargs = dict(zip(param_names, params))
            motif_graph = generator.generate_motif(
                start=current_start, id=current_id, **motif_kwargs
            )

            combined_graph.add_nodes_from(motif_graph.nodes(data=True))
            combined_graph.add_edges_from(motif_graph.edges())
            motif_node_sets.append(set(motif_graph.nodes()))

            current_start += len(motif_graph.nodes())
            motif_counts[base] += 1

        # Resolve RNG
        rng = kwargs.get('rng') or get_rng()

        # Compose motifs according to composition pattern
        pattern = kwargs.get('composition', 'sequential')
        comp_params = kwargs.get('composition_params', {}) or {}

        motif_edges = compose_motifs(len(motif_node_sets), pattern, comp_params, rng=rng)
        for i, j in motif_edges:
            if motif_node_sets[i] and motif_node_sets[j]:
                node1 = rng.choice(list(motif_node_sets[i]))
                node2 = rng.choice(list(motif_node_sets[j]))
                combined_graph.add_edge(node1, node2)

        # Add extra vertices, each connected to a random existing node
        combined_graph = add_vertices(combined_graph, num_extra_vertices, rng=rng)

        # Add extra edges randomly
        if num_extra_edges:
            combined_graph = add_random_edges(combined_graph, num_edges=num_extra_edges, rng=rng)

        return combined_graph
