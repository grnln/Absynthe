"""Tests for node-, edge- and graph-level labels in ``LabelingResult``.

Run with ``pytest tests/`` or ``python tests/test_edge_labels.py``.
"""
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interfaces import LabelingFunction  # noqa: E402
from interfaces.labeling_result import LabelingResult  # noqa: E402
from graph.labeling_functions import MotifLabelingFunction  # noqa: E402


class OnlyNodeLabels(LabelingFunction):
    """Labeling function que solo etiqueta nodos."""

    def label(self, graph: nx.Graph) -> LabelingResult:
        node_labels = {node: f"node_{node}" for node in graph.nodes()}
        return LabelingResult(node_labels=node_labels)


class OnlyEdgeLabels(LabelingFunction):
    """Labeling function que solo etiqueta aristas."""

    def label(self, graph: nx.Graph) -> LabelingResult:
        edge_labels = {}
        for u, v in graph.edges():
            avg_degree = (graph.degree(u) + graph.degree(v)) / 2
            edge_labels[(u, v)] = f"edge_avg_degree_{avg_degree:.1f}"
        return LabelingResult(node_labels={}, edge_labels=edge_labels)


class OnlyGraphLabels(LabelingFunction):
    """Labeling function que solo etiqueta el grafo."""

    def label(self, graph: nx.Graph) -> LabelingResult:
        return LabelingResult(
            node_labels={},
            graph_labels={
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "density": nx.density(graph),
            }
        )


class AllLabels(LabelingFunction):
    """Labeling function que etiqueta nodos, aristas y grafo."""

    def label(self, graph: nx.Graph) -> LabelingResult:
        node_labels = {node: "node" for node in graph.nodes()}
        edge_labels = {(u, v): f"edge_{u}_{v}" for u, v in graph.edges()}
        graph_labels = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        }
        return LabelingResult(
            node_labels=node_labels,
            edge_labels=edge_labels,
            graph_labels=graph_labels,
        )


def test_motif_labeling_still_works():
    """Verificar que MotifLabelingFunction sigue funcionando."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

    labeler = MotifLabelingFunction()
    result = labeler.label(graph)

    assert result.node_labels, "Should have node labels"
    assert isinstance(result.edge_labels, dict), "Should have edge_labels dict"
    assert len(result.edge_labels) == 0, "MotifLabelingFunction should have empty edge_labels"
    assert isinstance(result.graph_labels, dict), "Should have graph_labels dict"
    assert len(result.graph_labels) == 0, "MotifLabelingFunction should have empty graph_labels"
    print("✓ MotifLabelingFunction works with label() method")


def test_only_node_labels():
    """Verificar que se puede etiqueter solo nodos."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])

    labeler = OnlyNodeLabels()
    result = labeler.label(graph)

    assert len(result.node_labels) == 3, "Should have 3 node labels"
    assert len(result.edge_labels) == 0, "Should have no edge labels"
    assert len(result.graph_labels) == 0, "Should have no graph labels"
    print("✓ Only node labels works")


def test_only_edge_labels():
    """Verificar que se puede etiquetar solo aristas."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

    labeler = OnlyEdgeLabels()
    result = labeler.label(graph)

    assert len(result.node_labels) == 0, "Should have no node labels"
    assert len(result.edge_labels) == 3, "Should have 3 edge labels"
    assert len(result.graph_labels) == 0, "Should have no graph labels"
    print("✓ Only edge labels works")


def test_only_graph_labels():
    """Verificar que se puede etiquetar solo el grafo."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])

    labeler = OnlyGraphLabels()
    result = labeler.label(graph)

    assert len(result.node_labels) == 0, "Should have no node labels"
    assert len(result.edge_labels) == 0, "Should have no edge labels"
    assert len(result.graph_labels) == 3, "Should have 3 graph labels"
    assert result.graph_labels["num_nodes"] == 3
    assert result.graph_labels["num_edges"] == 2
    print("✓ Only graph labels works")


def test_all_labels():
    """Verificar que se pueden etiquetar nodos, aristas y grafo a la vez."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])

    labeler = AllLabels()
    result = labeler.label(graph)

    assert len(result.node_labels) == 3, "Should have 3 node labels"
    assert len(result.edge_labels) == 2, "Should have 2 edge labels"
    assert len(result.graph_labels) == 2, "Should have 2 graph labels"
    print("✓ All labels (nodes, edges, graph) works")


def test_edge_labels_in_graph():
    """Verificar que edge_labels se escriben en el grafo."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

    labeler = AllLabels()
    result = labeler.label(graph)

    # Mismo volcado que usa _compute_and_store_labels en dataset_generator.py
    from utils.kg_utils import apply_labeling_result

    apply_labeling_result(graph, result, "expected_ground_truth")

    # Verificar que las etiquetas están en el grafo
    for u, v in graph.edges():
        assert "label" in graph.edges[u, v], f"Edge ({u}, {v}) should have 'label' attribute"
        assert graph.edges[u, v]["label"] == f"edge_{u}_{v}"

    print("✓ Edge labels are stored in graph attributes")


if __name__ == "__main__":
    test_motif_labeling_still_works()
    test_only_node_labels()
    test_only_edge_labels()
    test_only_graph_labels()
    test_all_labels()
    test_edge_labels_in_graph()
    print("\n✅ All tests passed!")
