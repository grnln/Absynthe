"""Tests for the grid and clique motif generators and their use as labels."""
import networkx as nx
import pytest

from graph.composite_graph_generator import MotifComposite
from graph.labeling_functions import MotifLabelingFunction
from motifs import (
    ChainMotifGenerator,
    CliqueMotifGenerator,
    GateMotifGenerator,
    GridMotifGenerator,
    StarMotifGenerator,
    motif_generators,
    motif_param_names,
)


# ── Grid ────────────────────────────────────────────────────────────────────

def test_grid_structure_and_attributes():
    g = GridMotifGenerator().generate_motif(start=10, rows=2, cols=3, id=7)

    assert sorted(g.nodes()) == list(range(10, 16))
    assert g.number_of_edges() == 2 * 2 + 3 * 1  # rows*(cols-1) + cols*(rows-1)
    assert nx.is_isomorphic(g, nx.grid_2d_graph(2, 3))
    # Row-major layout: (r, c) -> start + r*cols + c
    assert g.has_edge(10, 11) and g.has_edge(10, 13) and not g.has_edge(12, 13)
    for _, data in g.nodes(data=True):
        assert data["motif"] == "grid_2_3"
        assert data["motif_id"] == "grid_2_3_7"


@pytest.mark.parametrize("rows, cols", [(1, 3), (3, 1), (0, 2)])
def test_grid_rejects_degenerate_sizes(rows, cols):
    with pytest.raises(ValueError):
        GridMotifGenerator().generate_motif(0, rows, cols)


# ── Clique ──────────────────────────────────────────────────────────────────

def test_clique_structure_and_attributes():
    g = CliqueMotifGenerator().generate_motif(start=5, num_nodes=4, id=2)

    assert sorted(g.nodes()) == [5, 6, 7, 8]
    assert g.number_of_edges() == 6
    assert nx.is_isomorphic(g, nx.complete_graph(4))
    for _, data in g.nodes(data=True):
        assert data["motif"] == "clique_4"
        assert data["motif_id"] == "clique_4_2"


def test_clique_rejects_single_node():
    with pytest.raises(ValueError):
        CliqueMotifGenerator().generate_motif(0, 1)


# ── Registry and composite generation ───────────────────────────────────────

def test_registry_exposes_new_motifs():
    assert motif_generators["grid"] is GridMotifGenerator
    assert motif_generators["clique"] is CliqueMotifGenerator
    assert motif_param_names["grid"] == ["rows", "cols"]
    assert motif_param_names["clique"] == ["num_nodes"]


def test_composite_builds_grid_and_clique():
    graph = MotifComposite(motifs=[["grid", 2, 3], ["clique", 4]]).generate_graph()

    assert graph.number_of_nodes() == 6 + 4
    motifs = {data["motif"] for _, data in graph.nodes(data=True)}
    assert motifs == {"grid_2_3", "clique_4"}
    # 7 grid edges + 6 clique edges + 1 sequential composition edge
    assert graph.number_of_edges() == 7 + 6 + 1


# ── Labeling ────────────────────────────────────────────────────────────────

def test_motif_labeling_with_grid_and_clique():
    graph = MotifComposite(motifs=[["grid", 2, 3], ["clique", 4]]).generate_graph()
    result = MotifLabelingFunction(motif_order=["grid_2_3", "clique_4"]).label(graph)

    for node, data in graph.nodes(data=True):
        assert result.node_labels[node] == data["motif"]
    assert "unknown" not in result.node_labels.values()

    instances = result.metadata["instances"]
    assert {inst["motif_name"] for inst in instances} == {"grid_2_3", "clique_4"}
    assert len(instances) == 2


def test_star_and_gate_labels_carry_their_size():
    # Disjoint union (no composition edge) so no node can accidentally join
    # a second star_4 instance through the star's center.
    star = StarMotifGenerator().generate_motif(0, 4)
    gate = GateMotifGenerator().generate_motif(star.number_of_nodes(), 2)
    graph = nx.union(star, gate)
    result = MotifLabelingFunction(motif_order=["star_4", "gate_2"]).label(graph)

    for node, data in graph.nodes(data=True):
        assert result.node_labels[node] == data["motif"]
    assert {inst["motif_name"] for inst in result.metadata["instances"]} == {"star_4", "gate_2"}


def test_chain_label_carries_its_length():
    chain = ChainMotifGenerator().generate_motif(0, 5)
    instances = ChainMotifGenerator.assign_labels(chain, 5)

    assert len(instances) == 1
    assert all(d["label"] == "chain_5" for _, d in chain.nodes(data=True))
    # Default length is 3: a 5-node path contains three distinct 3-node paths.
    fresh = ChainMotifGenerator().generate_motif(0, 5)
    assert len(ChainMotifGenerator.assign_labels(fresh)) == 3
    assert all(d["label"] == "chain_3" for _, d in fresh.nodes(data=True))


def test_assign_labels_default_parameters():
    grid = GridMotifGenerator().generate_motif(0, 3, 3)
    assert len(GridMotifGenerator.assign_labels(grid)) == 1
    assert all(d["label"] == "grid_3_3" for _, d in grid.nodes(data=True))

    clique = CliqueMotifGenerator().generate_motif(0, 4)
    assert len(CliqueMotifGenerator.assign_labels(clique)) == 1
    assert all(d["label"] == "clique_4" for _, d in clique.nodes(data=True))
