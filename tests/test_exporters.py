"""Tests for the pluggable export formats (``graph.exporters``).

Run with ``pytest tests/`` or ``python tests/test_exporters.py``.
"""
import os
import sys
import tempfile
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interfaces import GraphExporter  # noqa: E402
from graph.composite_graph_generator import MotifComposite  # noqa: E402
from graph.dataset_generator import GraphDatasetGenerator  # noqa: E402
from graph.dataset_metadata import load_dataset_config, load_dataset_metadata  # noqa: E402
from graph.exporters import (  # noqa: E402
    DEFAULT_FORMATS,
    EXPORTERS,
    GraphMLExporter,
    TriplesExporter,
    export_graph,
    register_exporter,
    resolve_exporters,
)
from graph.folder_graph_generator import load_graph_file  # noqa: E402
from graph.labeling_functions import MotifLabelingFunction  # noqa: E402
from graph.perturbations import RemoveNodesPerturbation  # noqa: E402
from graph.schema_kg_generator import SchemaKGGenerator  # noqa: E402
from utils.rng import reset_rng, set_seed  # noqa: E402


class EdgeListExporter(GraphExporter):
    """Minimal custom format used by the tests (not registered globally)."""

    name = "edgelist"

    def export(self, graph, base_path):
        path = base_path + ".edgelist"
        nx.write_edgelist(graph, path, data=False)
        return path


class SkippingExporter(GraphExporter):
    name = "skip"

    def export(self, graph, base_path):
        return None


def _plain_dataset(output_dir, formats=None):
    return GraphDatasetGenerator(
        graph_generator=MotifComposite(motifs=[["cycle", 4], ["house"]]),
        labeling_functions=[MotifLabelingFunction(motif_order=["cycle_4", "house"])],
        perturbations=[(RemoveNodesPerturbation(num_nodes=1, strategy="motif"), 1)],
        output_dir=output_dir,
        formats=formats,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_defaults():
    assert DEFAULT_FORMATS == ("graphml", "triples")
    assert EXPORTERS["graphml"] is GraphMLExporter and EXPORTERS["triples"] is TriplesExporter

    exporters = resolve_exporters()
    assert [e.name for e in exporters] == ["graphml", "triples"]

    # instances pass through, names are instantiated
    custom = EdgeListExporter()
    assert resolve_exporters(["graphml", custom])[1] is custom

    for bad in (["nope"], [], ["graphml", "graphml"]):
        try:
            resolve_exporters(bad)
        except ValueError as exc:
            if bad == ["nope"]:
                assert "nope" in str(exc) and "graphml" in str(exc)
        else:
            raise AssertionError(f"resolve_exporters({bad}) should fail")

    for bad_cls in (type("NoName", (GraphExporter,), {"export": lambda self, g, p: None}),
                    type("Reserved", (GraphExporter,), {"name": "graph", "export": lambda self, g, p: None})):
        try:
            register_exporter(bad_cls)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad_cls.__name__} should not be registrable")
    print("[ok] exporter registry")


# ---------------------------------------------------------------------------
# export_graph
# ---------------------------------------------------------------------------

def test_export_graph_plain_skips_triples():
    g = nx.Graph([(0, 1), (1, 2)])
    with tempfile.TemporaryDirectory() as tmp:
        files = export_graph(g, os.path.join(tmp, "g"), resolve_exporters())
        assert set(files) == {"graphml"}
        assert files["graphml"].endswith("g.graphml")
        assert set(os.listdir(tmp)) == {"g.graphml"}
        back = load_graph_file(files["graphml"])
        assert set(back.edges()) == {("0", "1"), ("1", "2")}
    print("[ok] plain graphs get GraphML only")


def test_export_graph_kg_writes_triples():
    g = nx.MultiDiGraph(kg=True)
    g.add_edge("a", "b", relation="r1")
    g.add_edge("a", "b", relation="r2")
    with tempfile.TemporaryDirectory() as tmp:
        files = export_graph(g, os.path.join(tmp, "g"), resolve_exporters())
        assert set(files) == {"graphml", "triples"}
        assert files["triples"].endswith(".tsv")
        assert open(files["triples"], encoding="utf-8").read().splitlines() == ["a\tr1\tb", "a\tr2\tb"]
        assert load_graph_file(files["graphml"]).is_multigraph()
    print("[ok] knowledge graphs get GraphML + triples")


def test_primary_must_write():
    g = nx.Graph([(0, 1)])
    with tempfile.TemporaryDirectory() as tmp:
        try:
            export_graph(g, os.path.join(tmp, "g"), [SkippingExporter(), GraphMLExporter()])
        except ValueError as exc:
            assert "skip" in str(exc)
        else:
            raise AssertionError("a primary exporter that writes nothing must fail")
        # the same exporter is fine when it is not the primary one
        files = export_graph(g, os.path.join(tmp, "g"), [GraphMLExporter(), SkippingExporter()])
        assert set(files) == {"graphml"}
    print("[ok] primary format must write a file")


# ---------------------------------------------------------------------------
# GraphDatasetGenerator integration
# ---------------------------------------------------------------------------

def test_custom_exporter_end_to_end():
    register_exporter(EdgeListExporter)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            set_seed(1)
            gen = _plain_dataset(tmp, formats=["graphml", "edgelist"])
            variants = gen.generate_dataset(num_graphs=2, num_extra_vertices=1)
            reset_rng()
            assert gen.formats == ["graphml", "edgelist"]
            assert variants
            for entry in variants:
                assert entry["graph_path"].endswith(".graphml")
                assert entry["original_graph_path"].endswith(".graphml")
                assert entry["edgelist_path"].endswith(".edgelist") and os.path.exists(entry["edgelist_path"])
                assert entry["original_edgelist_path"].endswith(".edgelist") and os.path.exists(entry["original_edgelist_path"])
                assert os.path.splitext(entry["edgelist_path"])[0] == os.path.splitext(entry["graph_path"])[0]
                assert "triples_path" not in entry
            assert load_dataset_metadata(tmp)[0]["edgelist_path"] == variants[0]["edgelist_path"]
            assert load_dataset_config(tmp)["generation"]["formats"] == ["graphml", "edgelist"]
            assert "export_triples" not in load_dataset_config(tmp)["generation"]
    finally:
        EXPORTERS.pop("edgelist", None)

    # an exporter instance works without registering it
    with tempfile.TemporaryDirectory() as tmp:
        set_seed(1)
        variants = _plain_dataset(tmp, formats=["graphml", EdgeListExporter()]).generate_dataset(num_graphs=1)
        reset_rng()
        assert all(os.path.exists(e["edgelist_path"]) for e in variants)
    print("[ok] custom exporter end to end")


def test_plain_dataset_entry_keys_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        set_seed(2)
        variants = _plain_dataset(tmp).generate_dataset(num_graphs=2, num_extra_vertices=1)
        reset_rng()
        assert variants
        for entry in variants:
            path_keys = {k for k in entry if k.endswith("_path")}
            assert path_keys == {"graph_path", "original_graph_path"}, path_keys
            assert "graph_kind" not in entry
        assert sorted(os.listdir(os.path.join(tmp, "originals"))) == ["graph_0.graphml", "graph_1.graphml"]
    print("[ok] plain dataset metadata unchanged")


def test_graphml_only_on_kg():
    gen_cls = lambda tmp, formats: GraphDatasetGenerator(  # noqa: E731
        SchemaKGGenerator({"A": 3, "B": 2}, [{"name": "r", "domain": "A", "range": "B", "p": 1.0}]),
        output_dir=tmp,
        formats=formats,
    )
    with tempfile.TemporaryDirectory() as tmp:
        gen_cls(tmp, ["graphml"]).generate_dataset(num_graphs=1)
        assert os.listdir(os.path.join(tmp, "originals")) == ["graph_0.graphml"]
    with tempfile.TemporaryDirectory() as tmp:
        gen_cls(tmp, None).generate_dataset(num_graphs=1)
        assert sorted(os.listdir(os.path.join(tmp, "originals"))) == ["graph_0.graphml", "graph_0.tsv"]
    print("[ok] formats=['graphml'] skips triples on knowledge graphs")


if __name__ == "__main__":
    test_registry_defaults()
    test_export_graph_plain_skips_triples()
    test_export_graph_kg_writes_triples()
    test_primary_must_write()
    test_custom_exporter_end_to_end()
    test_plain_dataset_entry_keys_unchanged()
    test_graphml_only_on_kg()
