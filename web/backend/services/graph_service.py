"""Graph generation service."""
import web.backend.services  # noqa: F401 - side-effect import

from typing import Tuple

from web.backend.models.graph_models import GraphGenerateRequest, GraphStats
from web.backend.services import graph_store, serialization
from web.backend.services.registry import build_graph_kwargs


def generate(request: GraphGenerateRequest) -> Tuple[str, list, GraphStats]:
    """Generate a composite graph and store it. Returns (graph_id, elements, stats)."""
    from graph.composite_graph_generator import MotifComposite
    from utils.rng import set_seed, reset_rng

    if request.seed is not None:
        set_seed(request.seed)

    try:
        from utils.distributions import resolve_count, to_distribution
        from utils.rng import get_rng

        rng = get_rng()
        motif_lists = []
        for m in request.motifs:
            dist = to_distribution(m.count_distribution)
            n = resolve_count(dist if dist is not None else m.count, rng)
            motif_lists.extend([m.to_list()] * n)
        generator = MotifComposite(motifs=motif_lists)
        # Extras given as distributions are sampled inside generate_graph
        graph = generator.generate_graph(**build_graph_kwargs(request))
    finally:
        if request.seed is not None:
            reset_rng()

    stats = serialization.graph_stats(graph)

    graph_id = graph_store.store(graph)
    elements = serialization.graph_to_elements(graph)
    return graph_id, elements, stats
