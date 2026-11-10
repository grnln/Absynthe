"""Per-graph random counts for perturbations and extra vertices / edges.

Mirrors the motif ``count_distribution``: a perturbation's desired count and
the ``num_extra_vertices`` / ``num_extra_edges`` of the motif generators may
be an ``IntDistribution`` that is re-sampled for every base graph.

Run with ``pytest tests/`` or ``python tests/test_count_distributions.py``.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest  # noqa: E402

from graph.composite_graph_generator import MotifComposite  # noqa: E402
from graph.dataset_generator import GraphDatasetGenerator  # noqa: E402
from graph.dataset_metadata import load_dataset_config, load_dataset_metadata  # noqa: E402
from graph.labeling_functions import MotifLabelingFunction  # noqa: E402
from graph.perturbation_engine import PerturbationPipeline  # noqa: E402
from graph.composite_graph_generator import normalize_motif_entry  # noqa: E402
from graph.perturbations import RemoveEdgesPerturbation, RemoveNodesPerturbation  # noqa: E402
from utils.distributions import IntDistribution, resolve_count, to_distribution  # noqa: E402
from utils.rng import get_rng, reset_rng, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# utils.distributions helpers
# ---------------------------------------------------------------------------

def test_resolve_count_passes_ints_and_samples_distributions():
    rng = get_rng()
    assert resolve_count(3, rng) == 3
    assert resolve_count(0, rng, min_value=0) == 0
    assert resolve_count(0, rng, min_value=1) == 1  # clamped like sample_int

    dist = IntDistribution("uniform", {"min": 2, "max": 4})
    samples = {resolve_count(dist, rng) for _ in range(200)}
    assert samples <= {2, 3, 4} and len(samples) > 1

    # A plain {"type", "params"} mapping is accepted too
    assert 2 <= resolve_count({"type": "uniform", "params": {"min": 2, "max": 4}}, rng) <= 4


def test_to_distribution_accepts_models_dicts_and_none():
    from web.backend.models.distribution_models import IntDistributionModel

    assert to_distribution(None) is None
    d = IntDistribution("poisson", {"lambda": 2})
    assert to_distribution(d) is d
    assert to_distribution({"type": "normal", "params": {"mean": 1, "std": 1}}) == IntDistribution(
        "normal", {"mean": 1, "std": 1}
    )
    model = IntDistributionModel(type="uniform", params={"min": 1, "max": 3})
    assert to_distribution(model) == IntDistribution("uniform", {"min": 1, "max": 3})


# ---------------------------------------------------------------------------
# Extra vertices / edges sampled per graph
# ---------------------------------------------------------------------------

def _count_extra_vertices(graph):
    return sum(1 for _, data in graph.nodes(data=True) if "motif" not in data)


def test_motif_composite_samples_extras_per_graph():
    set_seed(7)
    try:
        gen = MotifComposite(motifs=[["house"], ["cycle", 4]])
        base_edges = gen.generate_graph(num_extra_vertices=0, num_extra_edges=0).number_of_edges()
        vdist = IntDistribution("uniform", {"min": 0, "max": 3})
        edist = IntDistribution("uniform", {"min": 0, "max": 2})

        extra_v, extra_e = set(), set()
        for _ in range(60):
            g = gen.generate_graph(num_extra_vertices=vdist, num_extra_edges=edist)
            nv = _count_extra_vertices(g)
            extra_v.add(nv)
            # extra vertices add one edge each; the remainder are extra edges
            extra_e.add(g.number_of_edges() - base_edges - nv)

        assert extra_v <= {0, 1, 2, 3} and len(extra_v) > 1
        assert extra_e <= {0, 1, 2} and len(extra_e) > 1

        # Fixed ints still behave as before (and are not re-sampled)
        g = gen.generate_graph(num_extra_vertices=2, num_extra_edges=1)
        assert _count_extra_vertices(g) == 2
        assert g.number_of_edges() == base_edges + 2 + 1
    finally:
        reset_rng()


# ---------------------------------------------------------------------------
# Motif counts: MotifComposite accepts plain motifs and (motif, count) pairs
# ---------------------------------------------------------------------------

def test_normalize_motif_entry_accepts_both_forms():
    dist = IntDistribution("uniform", {"min": 1, "max": 2})
    assert normalize_motif_entry(["house"]) == (["house"], 1)
    assert normalize_motif_entry(["cycle", 4]) == (["cycle", 4], 1)
    assert normalize_motif_entry((["house"], 3)) == (["house"], 3)
    assert normalize_motif_entry((["cycle", 4], dist)) == (["cycle", 4], dist)
    assert normalize_motif_entry([["cycle", 4], 3]) == (["cycle", 4], 3)  # JSON round-trip form
    for bad in ([], [4], (["house"],), (["house"], 1, None), ([], 2)):
        with pytest.raises(ValueError):
            normalize_motif_entry(bad)


def _count_motifs(graph, name):
    return len({data["motif_id"] for _, data in graph.nodes(data=True) if data.get("motif") == name})


def test_motif_composite_fixed_pairs_match_expanded_lists():
    set_seed(2)
    try:
        a = MotifComposite(motifs=[(["house"], 2), ["cycle", 4]]).generate_graph()
    finally:
        reset_rng()
    set_seed(2)
    try:
        b = MotifComposite(motifs=[["house"], ["house"], ["cycle", 4]]).generate_graph()
    finally:
        reset_rng()
    assert sorted(a.nodes(data=True)) == sorted(b.nodes(data=True))
    assert sorted(map(sorted, a.edges())) == sorted(map(sorted, b.edges()))


def test_motif_composite_samples_motif_counts_per_graph():
    set_seed(1)
    try:
        gen = MotifComposite(motifs=[
            (["house"], IntDistribution("uniform", {"min": 1, "max": 3})),
            ["cycle", 4],
        ])
        houses, cycles, extras = set(), set(), set()
        for _ in range(40):
            g = gen.generate_graph(num_extra_vertices=IntDistribution("uniform", {"min": 1, "max": 3}))
            houses.add(_count_motifs(g, "house"))
            cycles.add(_count_motifs(g, "cycle_4"))
            extras.add(_count_extra_vertices(g))
        assert houses <= {1, 2, 3} and len(houses) > 1
        assert cycles == {1}
        assert extras <= {1, 2, 3} and len(extras) > 1
    finally:
        reset_rng()


# ---------------------------------------------------------------------------
# Perturbation desired_count sampled per base graph
# ---------------------------------------------------------------------------

def test_pipeline_samples_desired_count_per_call():
    set_seed(11)
    try:
        graph = MotifComposite(motifs=[["house"], ["house"], ["house"], ["cycle", 4]]).generate_graph()
        labeling = [MotifLabelingFunction(motif_order=["house", "cycle_4"])]
        dist = IntDistribution("uniform", {"min": 1, "max": 3})
        pipeline = PerturbationPipeline(
            perturbations=[(RemoveNodesPerturbation(num_nodes=1, strategy="motif"), dist)],
            labeling_functions=labeling,
            max_iterations=50,
        )
        targets = set()
        for _ in range(30):
            results = pipeline.apply_and_check(graph)
            assert results, "removing a motif node always changes labels"
            desired = {r["desired_count"] for r in results}
            assert len(desired) == 1
            targets |= desired
            assert len(results) == desired.pop()
        assert targets <= {1, 2, 3} and len(targets) > 1

        # An int count is reported unchanged
        pipeline = PerturbationPipeline(
            perturbations=[(RemoveNodesPerturbation(num_nodes=1, strategy="motif"), 2)],
            labeling_functions=labeling,
            max_iterations=50,
        )
        results = pipeline.apply_and_check(graph)
        assert len(results) == 2 and all(r["desired_count"] == 2 for r in results)
    finally:
        reset_rng()


def test_dataset_generator_records_distribution_counts_in_config():
    with tempfile.TemporaryDirectory() as tmp:
        set_seed(3)
        try:
            dist = IntDistribution("uniform", {"min": 1, "max": 2})
            gen = GraphDatasetGenerator(
                graph_generator=MotifComposite(motifs=[["house"], ["house"], ["cycle", 4]]),
                labeling_functions=[MotifLabelingFunction(motif_order=["house", "cycle_4"])],
                perturbations=[
                    (RemoveNodesPerturbation(num_nodes=1, strategy="motif"), dist),
                    (RemoveEdgesPerturbation(p_remove=0.3), 1),
                ],
                output_dir=tmp,
                max_perturbation_iterations=20,
            )
            variants = gen.generate_dataset(
                num_graphs=4,
                num_extra_vertices=IntDistribution("uniform", {"min": 0, "max": 2}),
                num_extra_edges=0,
            )
        finally:
            reset_rng()

        assert load_dataset_metadata(tmp) == json.loads(json.dumps(variants, default=str))
        cfg = load_dataset_config(tmp)
        json.dumps(cfg)  # fully serialisable
        perts = cfg["components"]["perturbations"]
        assert perts[0]["count"] == {"type": "uniform", "params": {"min": 1, "max": 2}}
        assert perts[1]["count"] == 1
        assert cfg["generation"]["graph_kwargs"]["num_extra_vertices"] == {
            "type": "uniform",
            "params": {"min": 0, "max": 2},
        }
        assert cfg["generation"]["graph_kwargs"]["num_extra_edges"] == 0

        per_graph = {}
        for entry in variants:
            if entry["perturbation_name"] == "remove_nodes":
                per_graph[entry["base_graph_id"]] = per_graph.get(entry["base_graph_id"], 0) + 1
        assert set(per_graph.values()) <= {1, 2}


# ---------------------------------------------------------------------------
# Web models / registry / JSON suite
# ---------------------------------------------------------------------------

def test_request_models_accept_distribution_fields():
    from web.backend.models.dataset_models import DatasetGenerateRequest
    from web.backend.models.graph_models import GraphGenerateRequest
    from web.backend.services.registry import build_graph_generator, build_graph_kwargs, build_perturbations

    req = DatasetGenerateRequest(
        num_graphs=2,
        motifs=[{"type": "house"}],
        num_extra_vertices=5,
        num_extra_vertices_distribution={"type": "poisson", "params": {"lambda": 1.5}},
        num_extra_edges=1,
        perturbations=[
            {"type": "remove_nodes", "count": 1, "count_distribution": {"type": "uniform", "params": {"min": 1, "max": 3}}},
            {"type": "remove_edges", "count": 2, "folder_name": "edges"},
        ],
    )
    kwargs = build_graph_kwargs(req)
    assert kwargs["num_extra_vertices"] == IntDistribution("poisson", {"lambda": 1.5})  # distribution wins
    assert kwargs["num_extra_edges"] == 1
    assert kwargs["composition"] == "sequential" and kwargs["composition_params"] == {}

    pairs = build_perturbations(req.perturbations)
    assert pairs[0][1] == IntDistribution("uniform", {"min": 1, "max": 3})
    assert pairs[1][1] == 2

    # Motif configs map onto (motif, count) pairs of a single MotifComposite
    gen = build_graph_generator(
        DatasetGenerateRequest(
            num_graphs=1,
            motifs=[
                {"type": "house", "count": 2},
                {"type": "cycle", "params": [5], "count_distribution": {"type": "poisson", "params": {"lambda": 2}}},
            ],
        )
    )
    assert isinstance(gen, MotifComposite)
    assert gen.motifs == [(["house"], 2), (["cycle", 5], IntDistribution("poisson", {"lambda": 2}))]

    # Same fields on the single-graph request
    greq = GraphGenerateRequest(
        motifs=[{"type": "house"}],
        num_extra_edges_distribution={"type": "uniform", "params": {"min": 0, "max": 2}},
    )
    assert build_graph_kwargs(greq)["num_extra_edges"] == IntDistribution("uniform", {"min": 0, "max": 2})
    assert build_graph_kwargs(greq)["num_extra_vertices"] == 0

    # Distribution params are validated like count_distribution
    with pytest.raises(ValueError):
        DatasetGenerateRequest(
            num_graphs=1,
            motifs=[{"type": "house"}],
            perturbations=[{"type": "remove_nodes", "count_distribution": {"type": "uniform", "params": {"min": 1}}}],
        )


def test_graph_service_samples_extras_from_request():
    from web.backend.models.graph_models import GraphGenerateRequest
    from web.backend.services import graph_service, graph_store

    seen = set()
    for seed in range(12):
        req = GraphGenerateRequest(
            motifs=[{"type": "house", "count": 1}],
            num_extra_vertices_distribution={"type": "uniform", "params": {"min": 0, "max": 3}},
            seed=seed,
        )
        graph_id, _elements, stats = graph_service.generate(req)
        seen.add(stats.num_nodes - 5)  # a house has 5 nodes
        graph_store.get(graph_id)
    assert seen <= {0, 1, 2, 3} and len(seen) > 1


def test_json_suite_with_distributions_round_trips():
    import gen_dataset

    with tempfile.TemporaryDirectory() as tmp:
        root = tmp.replace("\\", "/")
        suite = {
            "seed": 5,
            "clean_output": False,
            "datasets_root": root,
            "shared": {
                "num_graphs": 3,
                "max_perturbation_iterations": 20,
                "num_extra_vertices": 0,
                "num_extra_vertices_distribution": {"type": "uniform", "params": {"min": 0, "max": 2}},
                "num_extra_edges_distribution": {"type": "poisson", "params": {"lambda": 1}},
                "perturbations": [
                    {
                        "type": "remove_nodes",
                        "count_distribution": {"type": "uniform", "params": {"min": 1, "max": 2}},
                        "folder_name": "remove_nodes",
                        "params": {"num_nodes": 1, "strategy": "motif"},
                    }
                ],
            },
            "datasets": [
                {
                    "name": "houses",
                    "motifs": [{"type": "house", "count_distribution": {"type": "uniform", "params": {"min": 2, "max": 3}}}],
                    "labeling_functions": [{"type": "motif_labeling", "params": {"motif_order": ["house"]}}],
                }
            ],
        }
        cfg_path = os.path.join(tmp, "suite.json")
        with open(cfg_path, "w") as f:
            json.dump(suite, f)
        gen_dataset.main([cfg_path])

        cfg = load_dataset_config(os.path.join(tmp, "houses"))
        # The declarative request keeps the distribution fields verbatim...
        assert cfg["config"]["num_extra_vertices_distribution"] == {"type": "uniform", "params": {"min": 0, "max": 2}}
        assert cfg["config"]["perturbations"][0]["count_distribution"] == {"type": "uniform", "params": {"min": 1, "max": 2}}
        # ...and the runtime objects show what was actually sampled from
        assert cfg["generation"]["graph_kwargs"]["num_extra_vertices"] == {"type": "uniform", "params": {"min": 0, "max": 2}}
        assert cfg["generation"]["graph_kwargs"]["num_extra_edges"] == {"type": "poisson", "params": {"lambda": 1}}
        assert cfg["components"]["perturbations"][0]["count"] == {"type": "uniform", "params": {"min": 1, "max": 2}}
        assert load_dataset_metadata(os.path.join(tmp, "houses"))


if __name__ == "__main__":
    test_resolve_count_passes_ints_and_samples_distributions()
    test_to_distribution_accepts_models_dicts_and_none()
    test_motif_composite_samples_extras_per_graph()
    test_normalize_motif_entry_accepts_both_forms()
    test_motif_composite_fixed_pairs_match_expanded_lists()
    test_motif_composite_samples_motif_counts_per_graph()
    test_pipeline_samples_desired_count_per_call()
    test_dataset_generator_records_distribution_counts_in_config()
    test_request_models_accept_distribution_fields()
    test_graph_service_samples_extras_from_request()
    test_json_suite_with_distributions_round_trips()
    print("[ok] count distributions")
