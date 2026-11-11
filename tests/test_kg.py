"""Tests for the knowledge-graph (multi-relational, directed) support.

Run with ``pytest tests/`` or ``python tests/test_kg.py``.

The plain-graph regression test compares a reduced run of
``configs/default.json`` against a baseline generated before the KG changes.
Set ``ABSYNTHE_BASELINE=<dir>`` (the ``datasets_root`` of that baseline run,
produced with ``num_graphs=20``) to enable it; it is skipped otherwise.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.rng import set_seed, reset_rng  # noqa: E402
from interfaces import PerturbationHint
from graph.folder_graph_generator import FolderGraphGenerator, load_graph_file
from graph.schema_kg_generator import SchemaKGGenerator
from graph.kg_labeling_functions import EntityTypeLabelingFunction
from graph.kg_perturbations import CorruptTriplesPerturbation
from utils.kg_utils import edge_key, is_kg, iter_edges, graph_to_triples
from graph.labeling_functions import MotifLabelingFunction
from graph.perturbations import (
    AddEdgesPerturbation,
    EdgePerturbation,
    RemoveEdgesPerturbation,
    RemoveNodesPerturbation,
    _edge_in_zone,
)
from graph.perturbation_engine import PerturbationPipeline
from graph.reconstruction import normalize_changes, reconstruct_original
from graph.dataset_generator import GraphDatasetGenerator
from graph.dataset_metadata import load_dataset_metadata
from motifs.utils import assign_labels_to_motif

SAMPLE_TTL = REPO_ROOT / "examples" / "rdf" / "sample.ttl"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.json"
EX = "http://example.org/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_set(G):
    """Edges with keys and attributes, comparable across copies."""
    return {
        (u, v, k, tuple(sorted((a, str(b)) for a, b in d.items())))
        for u, v, k, d in iter_edges(G)
    }


def _schema_kg(seed=3):
    set_seed(seed)
    gen = SchemaKGGenerator(
        entity_types={"Person": 10, "Company": 3, "Paper": 6},
        relations=[
            {"name": "works_at", "domain": "Person", "range": "Company", "p": 0.35},
            {"name": "wrote", "domain": "Person", "range": "Paper", "count": 8},
            {"name": "cites", "domain": "Paper", "range": "Paper", "p": 0.2},
            {"name": "knows", "domain": "Person", "range": "Person", "p": 0.15},
        ],
    )
    return gen.generate_graph()


def _patterns_labeler():
    return EntityTypeLabelingFunction(patterns={
        "employee": {"relation": "works_at", "direction": "out"},
        "author": {"relation": "wrote", "direction": "out"},
    })


# ---------------------------------------------------------------------------
# 1. RDF loading
# ---------------------------------------------------------------------------

def test_rdf_round_trip():
    g = load_graph_file(str(SAMPLE_TTL))
    assert isinstance(g, nx.MultiDiGraph)
    assert is_kg(g)

    alice, bob, carol = EX + "alice", EX + "bob", EX + "carol"
    assert g.nodes[alice]["type"] == "ex:Author"
    assert g.nodes[alice]["types"] == "ex:Author|ex:Person"
    assert g.nodes[bob]["type"] == "ex:Person"
    assert g.nodes[alice]["foaf:name"] == "Alice"

    # two parallel relations alice -> bob, direction preserved
    rels = sorted(d["relation"] for d in g[alice][bob].values())
    assert rels == ["ex:collaboratesWith", "foaf:knows"]
    assert not g.has_edge(bob, alice)
    assert g.has_edge(carol, alice)

    # rdf:type is not an edge and classes are not nodes
    assert not any(d["relation"] == "rdf:type" for _, _, d in g.edges(data=True))
    assert EX + "Person" not in g
    assert all("predicate" in d for _, _, d in g.edges(data=True))

    # legacy mode still collapses onto a simple undirected graph
    plain = load_graph_file(str(SAMPLE_TTL), as_undirected=True)
    assert isinstance(plain, nx.Graph) and not plain.is_directed()
    print("[ok] RDF round trip")


def test_folder_generator_serves_kgs():
    gen = FolderGraphGenerator(str(SAMPLE_TTL.parent))
    g = gen.generate_graph()
    assert g.is_directed() and g.is_multigraph() and is_kg(g)
    print("[ok] FolderGraphGenerator yields knowledge graphs")


# ---------------------------------------------------------------------------
# 2. GraphML round trip
# ---------------------------------------------------------------------------

def test_graphml_multigraph_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        gen = GraphDatasetGenerator(SchemaKGGenerator({"A": 2}, []), output_dir=tmp)
        # with parallel edges
        g1 = nx.MultiDiGraph(kg=True)
        g1.add_edge("a", "b", relation="r1")
        g1.add_edge("a", "b", relation="r2")
        # without parallel edges (would reload as DiGraph without the stamp)
        g2 = nx.MultiDiGraph(kg=True)
        g2.add_edge("a", "b", relation="r1")
        g2.add_edge("b", "c", relation="r2")
        for i, g in enumerate((g1, g2)):
            files = gen.save_graph(g, os.path.join(tmp, f"g{i}"))
            path, triples = files["graphml"], files["triples"]
            back = load_graph_file(path)
            assert back.is_multigraph() and back.is_directed(), type(back)
            assert _edge_set(back) == _edge_set(g)
            assert triples and triples.endswith(".tsv") and os.path.exists(triples)
            assert len(open(triples).read().splitlines()) == g.number_of_edges()
    print("[ok] GraphML multigraph round trip")


def test_triples_export_nt_for_iri_nodes():
    g = load_graph_file(str(SAMPLE_TTL))
    with tempfile.TemporaryDirectory() as tmp:
        gen = GraphDatasetGenerator(SchemaKGGenerator({"A": 1}, []), output_dir=tmp)
        triples = gen.save_graph(g, os.path.join(tmp, "g"))["triples"]
        assert triples.endswith(".nt")
        import rdflib
        rg = rdflib.Graph()
        rg.parse(triples, format="nt")
        assert len(rg) == g.number_of_edges()
    assert len(graph_to_triples(g)) == g.number_of_edges()
    print("[ok] N-Triples export")


# ---------------------------------------------------------------------------
# 3. Perturbations + reconstruction on MultiDiGraph
# ---------------------------------------------------------------------------

def test_perturb_and_reconstruct_multidigraph():
    G = _schema_kg()
    perturbations = [
        RemoveNodesPerturbation(2, strategy="random"),
        RemoveNodesPerturbation(1, strategy="by_attribute", params={"attr": "type", "value": "Company"}),
        RemoveEdgesPerturbation(0.5, relations=["works_at"]),
        AddEdgesPerturbation(add_num=4),
        EdgePerturbation(0.3, 0.05),
        CorruptTriplesPerturbation(3, mode="relation"),
        CorruptTriplesPerturbation(3, mode="head"),
        CorruptTriplesPerturbation(3, mode="tail"),
        CorruptTriplesPerturbation(3, mode="any"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        saver = GraphDatasetGenerator(SchemaKGGenerator({"A": 1}, []), output_dir=tmp)
        for i, p in enumerate(perturbations):
            H, changes = p.apply(G)
            assert any(changes.values()), f"{type(p).__name__} changed nothing"
            R = reconstruct_original(H, changes)
            assert _edge_set(R) == _edge_set(G), type(p).__name__
            assert dict(R.nodes(data=True)) == dict(G.nodes(data=True))

            # removed edges keep their relation; added ones get one
            for rec in changes.get("removed_edges", []):
                assert "key" in rec and "relation" in rec["attrs"]
            for rec in changes.get("added_edges", []):
                assert "key" in rec and rec["attrs"]["relation"] in ("works_at", "wrote", "cites", "knows")

            # same after a GraphML + JSON round trip
            path = saver.save_graph(H, os.path.join(tmp, f"v{i}"))["graphml"]
            loaded = load_graph_file(path)
            ch = normalize_changes(json.loads(json.dumps(changes, default=str)), loaded)
            R2 = reconstruct_original(loaded, ch)
            assert {(u, v, k) for u, v, k, _ in iter_edges(R2)} == {(u, v, k) for u, v, k, _ in iter_edges(G)}, type(p).__name__
            assert sorted(d["relation"] for *_, d in R2.edges(data=True)) == sorted(d["relation"] for *_, d in G.edges(data=True))
    print("[ok] perturb + reconstruct on MultiDiGraph")


def test_corrupt_triples_never_duplicates():
    G = _schema_kg()
    existing = {(u, v, d["relation"]) for u, v, d in G.edges(data=True)}
    for mode in ("relation", "head", "tail"):
        H, ch = CorruptTriplesPerturbation(5, mode=mode).apply(G)
        triples = [(u, v, d["relation"]) for u, v, d in H.edges(data=True)]
        assert len(triples) == len(set(triples)), mode
        for rec in ch["corrupted_triples"]:
            new = (rec["new_u"], rec["new_v"], rec["new_attrs"]["relation"])
            assert new not in existing, (mode, new)
    print("[ok] corrupt_triples avoids duplicates")


# ---------------------------------------------------------------------------
# 4. Hints
# ---------------------------------------------------------------------------

def test_hint_respects_direction_and_keys():
    G = _schema_kg()
    res = _patterns_labeler().label(G)
    assert res.hint is not None and res.hint.edges
    ref = next(iter(res.hint.edges))
    assert len(ref) == 3  # (u, v, key)
    u, v, k = ref
    assert _edge_in_zone(u, v, PerturbationHint(edges={ref}), k, directed=True)
    assert not _edge_in_zone(v, u, PerturbationHint(edges={ref}), k, directed=True)
    # keyless query on a keyed hint matches the pair
    assert PerturbationHint(edges={ref}).contains_edge(u, v, directed=True)
    # legacy undirected call keeps sorting endpoints
    assert PerturbationHint.normalize_edge(5, 2) == (2, 5)
    print("[ok] hints honour direction and keys")


# ---------------------------------------------------------------------------
# 5. Motif matching on knowledge graphs
# ---------------------------------------------------------------------------

def test_motif_matching_on_kg():
    G = _schema_kg()
    res = MotifLabelingFunction(["cycle_3"]).label(G)
    found = {frozenset(i["nodes"]) for i in res.metadata.get("instances", [])}
    res_und = MotifLabelingFunction(["cycle_3"]).label(nx.Graph(G))
    found_und = {frozenset(i["nodes"]) for i in res_und.metadata.get("instances", [])}
    assert found == found_und
    if found:
        inst = res.metadata["instances"][0]
        assert all(len(e) == 3 for e in inst["edges"])  # keyed edge refs

    # relation-aware directed motif: Person -works_at-> *
    m = nx.DiGraph()
    m.add_edge(0, 1, relation="works_at")
    m.nodes[0]["type"] = "Person"
    inst = assign_labels_to_motif(G.copy(), m, "wa")
    n_works_at = sum(1 for *_, d in G.edges(data=True) if d["relation"] == "works_at")
    assert len(inst) == n_works_at
    # wrong type / unknown relation -> no match
    m.nodes[0]["type"] = "Paper"
    assert assign_labels_to_motif(G.copy(), m, "x") == []
    m2 = nx.DiGraph()
    m2.add_edge(0, 1, relation="nope")
    assert assign_labels_to_motif(G.copy(), m2, "x") == []
    print("[ok] motif matching on KG")


# ---------------------------------------------------------------------------
# 6. Pipeline accepts edge-label changes
# ---------------------------------------------------------------------------

def test_pipeline_accepts_edge_label_changes():
    G = _schema_kg()
    labeler = EntityTypeLabelingFunction()  # no patterns: node labels never change
    pipe = PerturbationPipeline([(CorruptTriplesPerturbation(2, "relation"), 2)], [labeler], max_iterations=10)
    results = pipe.apply_and_check(G)
    assert len(results) == 2
    assert all(not r["changed_nodes"] and r["changed_edges"] for r in results)

    pipe = PerturbationPipeline([(RemoveEdgesPerturbation(0.5, relations=["works_at"]), 1)], [_patterns_labeler()], max_iterations=10)
    results = pipe.apply_and_check(G)
    assert results and results[0]["changed_nodes"]
    print("[ok] pipeline accepts edge-label changes")


# ---------------------------------------------------------------------------
# 7. End-to-end KG dataset
# ---------------------------------------------------------------------------

def test_end_to_end_kg_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        set_seed(11)
        gen = GraphDatasetGenerator(
            graph_generator=SchemaKGGenerator(
                {"Person": 12, "Company": 4, "Paper": 8},
                [
                    {"name": "works_at", "domain": "Person", "range": "Company", "p": 0.25},
                    {"name": "wrote", "domain": "Person", "range": "Paper", "count": 10},
                    {"name": "cites", "domain": "Paper", "range": "Paper", "p": 0.15},
                ],
            ),
            labeling_functions=[_patterns_labeler()],
            perturbations=[
                (RemoveEdgesPerturbation(0.3, relations=["works_at"], folder_name="remove_works_at"), 1),
                (CorruptTriplesPerturbation(2, "relation"), 1),
                (RemoveNodesPerturbation(1, "by_attribute", {"attr": "type", "value": "Company"}), 1),
            ],
            output_dir=tmp,
            max_perturbation_iterations=15,
        )
        meta = gen.generate_dataset(num_graphs=4)
        reset_rng()
        assert meta
        json.dumps(meta)  # serialisable
        assert os.path.exists(os.path.join(tmp, "originals", "graph_0.graphml"))
        assert os.path.exists(os.path.join(tmp, "originals", "graph_0.tsv"))
        for e in meta:
            assert e["graph_kind"] == {"directed": True, "multigraph": True, "kg": True}
            assert os.path.exists(e["triples_path"])
            P = load_graph_file(e["graph_path"])
            O = load_graph_file(e["original_graph_path"])
            R = reconstruct_original(P, normalize_changes(json.loads(json.dumps(e["perturbation_info"]["changes"], default=str)), P))
            assert {(u, v, k) for u, v, k, _ in iter_edges(R)} == {(u, v, k) for u, v, k, _ in iter_edges(O)}
            assert set(R.nodes()) == set(O.nodes())
            # observed labels were stored on nodes and edges
            assert all("observed_ground_truth" in d for _, d in P.nodes(data=True))
            assert all("label" in d for *_, d in P.edges(data=True))
    print("[ok] end-to-end KG dataset")


# ---------------------------------------------------------------------------
# 8. Plain-graph regression
# ---------------------------------------------------------------------------

def _norm_entry(entry, root):
    e = json.loads(json.dumps(entry).replace(root, "ROOT").replace(root.replace("/", "\\\\"), "ROOT"))
    ch = e["perturbation_info"]["changes"]
    # A removed edge that is re-added by the same perturbation used to be
    # cancelled out by the old set-based diff; the new records list both.
    if "removed_edges" in ch and "added_edges" in ch:
        rem = {(r["u"], r["v"]) for r in ch["removed_edges"]}
        add = {(r["u"], r["v"]) for r in ch["added_edges"]}
        both = rem & add
        ch["removed_edges"] = [r for r in ch["removed_edges"] if (r["u"], r["v"]) not in both]
        ch["added_edges"] = [r for r in ch["added_edges"] if (r["u"], r["v"]) not in both]
    for k in ("removed_edges", "added_edges"):
        if k in ch:
            ch[k] = sorted(ch[k], key=lambda r: (r["u"], r["v"]))
    return e


def test_plain_regression():
    baseline = os.environ.get("ABSYNTHE_BASELINE")
    if not baseline:
        print("[skip] plain regression (set ABSYNTHE_BASELINE)")
        return
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import gen_dataset

    with tempfile.TemporaryDirectory() as tmp:
        cfg = json.load(open(DEFAULT_CONFIG))
        cfg["shared"]["num_graphs"] = 20
        cfg["datasets_root"] = tmp.replace("\\", "/")
        cfg_path = os.path.join(tmp, "cfg.json")
        json.dump(cfg, open(cfg_path, "w"))
        gen_dataset.main([cfg_path])

        for ds in [d["name"] for d in cfg["datasets"]]:
            A = load_dataset_metadata(os.path.join(baseline, ds))
            B = load_dataset_metadata(os.path.join(tmp, ds))
            assert len(A) == len(B), ds
            for a, b in zip(A, B):
                assert _norm_entry(a, baseline.replace("\\", "/")) == _norm_entry(b, tmp.replace("\\", "/")), (ds, a["graph_id"])
                assert nx.utils.graphs_equal(nx.read_graphml(a["graph_path"]), nx.read_graphml(b["graph_path"]))
                assert "graph_kind" not in b and "changed_edges" not in b["perturbation_info"]
            for f in os.listdir(os.path.join(baseline, ds, "originals")):
                assert nx.utils.graphs_equal(
                    nx.read_graphml(os.path.join(baseline, ds, "originals", f)),
                    nx.read_graphml(os.path.join(tmp, ds, "originals", f)),
                )
    print("[ok] plain regression against baseline")


if __name__ == "__main__":
    test_rdf_round_trip()
    test_folder_generator_serves_kgs()
    test_graphml_multigraph_round_trip()
    test_triples_export_nt_for_iri_nodes()
    test_perturb_and_reconstruct_multidigraph()
    test_corrupt_triples_never_duplicates()
    test_hint_respects_direction_and_keys()
    test_motif_matching_on_kg()
    test_pipeline_accepts_edge_label_changes()
    test_end_to_end_kg_dataset()
    test_plain_regression()
    print("\nAll KG tests passed.")
