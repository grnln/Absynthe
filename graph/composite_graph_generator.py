import networkx as nx
from interfaces import GraphGenerator
from motifs import motif_generators, motif_param_names
from typing import List, Sequence, Tuple
from collections import defaultdict
from utils.graph_utils import add_vertices, add_random_edges
from utils.distributions import CountSpec, resolve_count
from utils.rng import get_rng
from graph.composition_engine import compose_motifs


# One entry of ``MotifComposite(motifs=...)``, normalized: the motif
# ``[type, *params]`` and how many instances to build (int or distribution).
MotifSpec = Tuple[List, CountSpec]


def normalize_motif_entry(entry) -> MotifSpec:
    """Normalize one ``motifs`` entry to ``(motif_list, count_spec)``.

    Accepted forms:

    * ``["house"]`` / ``["cycle", 4]`` — one instance
    * ``(["house"], 3)`` — a fixed number of instances
    * ``(["house"], IntDistribution(...))`` — a number re-sampled per graph

    Lists and tuples are interchangeable (JSON round-trips turn tuples into
    lists), so ``[["house"], 3]`` is the second form as well.
    """
    if not isinstance(entry, (list, tuple)) or not entry:
        raise ValueError(f"Invalid motif entry: {entry!r}")
    first = entry[0]
    if isinstance(first, str):
        return list(entry), 1
    if isinstance(first, (list, tuple)) and len(entry) == 2 and first and isinstance(first[0], str):
        return list(first), entry[1]
    raise ValueError(
        f"Invalid motif entry: {entry!r}. Expected [type, *params] or ([type, *params], count)."
    )


class MotifComposite(GraphGenerator):
    """Generator for composite graphs built from a list of motif specifications.

    This is one concrete implementation of GraphGenerator. It constructs
    graphs by instantiating structural motifs (cycles, houses, chains, etc.),
    composing them according to a pattern, and optionally adding extra
    vertices and edges.

    Each entry of ``motifs`` is either a motif ``[type, *params]`` (one
    instance) or a pair ``(motif, count)`` where ``count`` is an int or an
    ``IntDistribution`` sampled on every call, so each graph can carry a
    different number of instances (see ``normalize_motif_entry``).
    """

    def __init__(self, motifs: Sequence = None):
        # Kept as given for provenance (config.json records it verbatim)
        self.motifs = list(motifs or [])
        self._specs: List[MotifSpec] = [normalize_motif_entry(m) for m in self.motifs]
        self.generators = motif_generators

    def expand_motifs(self, rng=None) -> List[List]:
        """The concrete motif list for one graph: fixed counts repeated and
        distribution counts sampled now (clamped to ``>= 1``)."""
        rng = rng or get_rng()
        expanded: List[List] = []
        for motif_list, count_spec in self._specs:
            expanded.extend([motif_list] * resolve_count(count_spec, rng, min_value=1))
        return expanded

    def generate_graph(
        self,
        num_extra_vertices: CountSpec = 0,
        num_extra_edges: CountSpec = 0,
        **kwargs,
    ) -> nx.Graph:
        """Generate a composite graph by combining motifs from the list.

        Args:
            num_extra_vertices: Number of additional vertices to add. Either a
                fixed int or an ``IntDistribution`` sampled on every call
                (clamped to ``>= 0``), so each graph gets its own count.
            num_extra_edges: Number of additional edges to add randomly; same
                int-or-distribution semantics as ``num_extra_vertices``.
            **kwargs: Additional arguments (start, composition, composition_params).

        Returns:
            nx.Graph: The combined graph with extra vertices and edges.
        """
        # Resolve RNG first: motif counts given as distributions are sampled here
        rng = kwargs.get('rng') or get_rng()

        combined_graph = nx.Graph()
        current_start = kwargs.get("start", 0)
        motif_counts = defaultdict(int)
        motif_node_sets = []

        for motif_list in self.expand_motifs(rng):
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

        # Compose motifs according to composition pattern
        pattern = kwargs.get('composition', 'sequential')
        comp_params = kwargs.get('composition_params', {}) or {}

        motif_edges = compose_motifs(len(motif_node_sets), pattern, comp_params, rng=rng)
        for i, j in motif_edges:
            if motif_node_sets[i] and motif_node_sets[j]:
                node1 = rng.choice(list(motif_node_sets[i]))
                node2 = rng.choice(list(motif_node_sets[j]))
                combined_graph.add_edge(node1, node2)

        # Extras: fixed ints pass through, distributions are sampled per graph
        n_vertices = resolve_count(num_extra_vertices, rng, min_value=0)
        n_edges = resolve_count(num_extra_edges, rng, min_value=0)

        # Add extra vertices, each connected to a random existing node
        combined_graph = add_vertices(combined_graph, n_vertices, rng=rng)

        # Add extra edges randomly
        if n_edges:
            combined_graph = add_random_edges(combined_graph, num_edges=n_edges, rng=rng)

        return combined_graph
