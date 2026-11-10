"""Provenance: ``config.json`` stores seed, config, components and generation
arguments next to ``metadata.json`` (the variants list).

Run with ``pytest tests/`` or ``python tests/test_metadata.py``.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import networkx as nx  # noqa: E402

from graph.composite_graph_generator import MotifComposite  # noqa: E402
from graph.dataset_generator import GraphDatasetGenerator  # noqa: E402
from graph.dataset_metadata import (  # noqa: E402
    describe_component,
    find_variant,
    load_dataset_config,
    load_dataset_metadata,
)
from graph.labeling_functions import MotifLabelingFunction  # noqa: E402
from graph.perturbations import EdgePerturbation, RemoveNodesPerturbation  # noqa: E402
from graph.reconstruction import normalize_changes, reconstruct_original  # noqa: E402
from utils.distributions import IntDistribution  # noqa: E402
from utils.rng import reset_rng, set_seed  # noqa: E402


def _generate(output_dir, seed=None, config=None, provenance=None):
    gen = GraphDatasetGenerator(
        graph_generator=MotifComposite(motifs=[["cycle", 4], ["house"]]),
        labeling_functions=[MotifLabelingFunction(motif_order=["cycle_4", "house"])],
        perturbations=[
            (RemoveNodesPerturbation(num_nodes=1, strategy="motif"), 1),
            (EdgePerturbation(p_remove=0.2, folder_name="edges"), 1),
        ],
        output_dir=output_dir,
        max_perturbation_iterations=10,
        seed=seed,
        config=config,
        provenance=provenance,
    )
    variants = gen.generate_dataset(num_graphs=2, num_extra_vertices=1)
    return gen, variants


def test_config_json_records_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        set_seed(5)
        gen, variants = _generate(tmp, config={"motifs": ["cycle_4", "house"]}, provenance={"note": "x"})
        reset_rng()

        # metadata.json is the plain variants list, as before
        assert load_dataset_metadata(tmp) == json.loads(json.dumps(variants, default=str))

        cfg = load_dataset_config(tmp)
        assert cfg == gen.last_config
        assert set(cfg) == {"seed", "config", "components", "generation", "provenance"}
        assert cfg["seed"] == 5  # picked up from utils.rng.set_seed
        assert cfg["config"] == {"motifs": ["cycle_4", "house"]}
        assert cfg["provenance"] == {"note": "x"}
        assert cfg["generation"]["num_graphs"] == 2
        assert cfg["generation"]["graph_kwargs"] == {"num_extra_vertices": 1}

        comps = cfg["components"]
        assert comps["graph_generator"]["class"] == "MotifComposite"
        assert comps["graph_generator"]["params"]["motifs"] == [["cycle", 4], ["house"]]
        assert comps["labeling_functions"][0]["params"]["motif_order"] == ["cycle_4", "house"]
        perts = comps["perturbations"]
        assert perts[0]["class"] == "RemoveNodesPerturbation"
        assert perts[0]["params"]["num_nodes"] == 1 and perts[0]["params"]["strategy"] == "motif"
        assert perts[0]["count"] == 1
        assert perts[1]["params"]["p_remove"] == 0.2 and perts[1]["folder_name"] == "edges"
        assert "rng" not in perts[0]["params"]
    print("[ok] config.json records provenance")


def test_explicit_seed_wins_over_global_seed():
    with tempfile.TemporaryDirectory() as tmp:
        set_seed(5)
        _generate(tmp, seed=99)
        reset_rng()
        assert load_dataset_config(tmp)["seed"] == 99

    with tempfile.TemporaryDirectory() as tmp:
        reset_rng()
        _generate(tmp)
        assert load_dataset_config(tmp)["seed"] is None
        assert "provenance" not in load_dataset_config(tmp)
    print("[ok] seed precedence")


def test_variants_reconstruct_from_metadata_file_alone():
    with tempfile.TemporaryDirectory() as tmp:
        set_seed(3)
        _generate(tmp)
        reset_rng()
        variants = load_dataset_metadata(tmp)
        assert variants
        for entry in variants:
            perturbed = nx.read_graphml(entry["graph_path"])
            original = nx.read_graphml(entry["original_graph_path"])
            changes = normalize_changes(entry["perturbation_info"]["changes"], perturbed)
            rebuilt = reconstruct_original(perturbed, changes)
            assert set(rebuilt.nodes()) == set(original.nodes())
            assert {frozenset(e) for e in rebuilt.edges()} == {frozenset(e) for e in original.edges()}
        assert find_variant(variants, variants[0]["graph_id"]) is variants[0]
    print("[ok] variants reconstruct from metadata.json")


def test_dataset_without_config_json():
    with tempfile.TemporaryDirectory() as tmp:
        legacy = [{"graph_id": 0, "graph_path": "x.graphml", "perturbation_info": None}]
        with open(os.path.join(tmp, "metadata.json"), "w") as f:
            json.dump(legacy, f)
        assert load_dataset_metadata(tmp) == legacy
        assert load_dataset_config(tmp) is None
    print("[ok] datasets without config.json are still readable")


def test_describe_component_serialises_distributions():
    gen = MotifComposite(motifs=[
        (["house"], IntDistribution("normal", {"mean": 3, "std": 1})),
        (["cycle", 5], 2),
        ["star", 3],
    ])
    desc = describe_component(gen)
    assert desc["class"] == "MotifComposite"
    # The motifs are recorded as given: distributions become {type, params}
    assert desc["params"]["motifs"] == [
        [["house"], {"type": "normal", "params": {"mean": 3, "std": 1}}],
        [["cycle", 5], 2],
        ["star", 3],
    ]
    json.dumps(desc)
    print("[ok] component description is JSON serialisable")


def test_json_mode_stores_config_and_suite_provenance():
    import gen_dataset

    with tempfile.TemporaryDirectory() as tmp:
        root = tmp.replace("\\", "/")
        cfg = json.load(open(REPO_ROOT / "configs" / "default.json"))
        cfg["shared"]["num_graphs"] = 2
        cfg["datasets"] = cfg["datasets"][:2]
        cfg["datasets"][1]["seed"] = 123  # dataset-level seed
        cfg["datasets_root"] = root
        cfg_path = os.path.join(tmp, "cfg.json")
        json.dump(cfg, open(cfg_path, "w"))
        gen_dataset.main([cfg_path])

        first = load_dataset_config(os.path.join(tmp, cfg["datasets"][0]["name"]))
        second = load_dataset_config(os.path.join(tmp, cfg["datasets"][1]["name"]))

        assert first["seed"] == 42 and first["provenance"]["seed_scope"] == "suite"
        assert first["provenance"]["suite_index"] == 0 and first["provenance"]["suite_size"] == 2
        assert first["provenance"]["config_path"].endswith("cfg.json")
        assert first["config"]["num_graphs"] == 2
        assert first["config"]["motifs"][0]["type"] == "house"
        assert [p["type"] for p in first["config"]["perturbations"]] == [
            "remove_nodes", "edge_perturbation", "remove_edges",
        ]

        assert second["seed"] == 123 and second["provenance"]["seed_scope"] == "dataset"
        assert second["provenance"]["suite_seed"] == 42

        # A dataset with its own seed is reproducible from its folder alone:
        # replaying its stored config (not the suite) yields identical variants.
        replay_cfg = {
            "seed": None,
            "clean_output": False,
            "datasets_root": root + "/replay",
            "datasets": [{"name": "again", **second["config"], "output_dir": root + "/replay/again"}],
        }
        replay_path = os.path.join(tmp, "replay.json")
        json.dump(replay_cfg, open(replay_path, "w"))
        gen_dataset.main([replay_path])

        def strip_paths(entries):
            out = []
            for e in entries:
                e = dict(e)
                for k in ("graph_path", "original_graph_path", "triples_path", "original_triples_path"):
                    e.pop(k, None)
                out.append(e)
            return out

        replayed = load_dataset_metadata(root + "/replay/again")
        original = load_dataset_metadata(os.path.join(tmp, cfg["datasets"][1]["name"]))
        assert strip_paths(replayed) == strip_paths(original)
    print("[ok] JSON mode stores config and suite provenance")


if __name__ == "__main__":
    test_config_json_records_provenance()
    test_explicit_seed_wins_over_global_seed()
    test_variants_reconstruct_from_metadata_file_alone()
    test_dataset_without_config_json()
    test_describe_component_serialises_distributions()
    test_json_mode_stores_config_and_suite_provenance()
