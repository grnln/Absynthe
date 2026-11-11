"""Walkthrough of the building blocks on a single graph.

Composes three motifs, labels the nodes by motif membership, applies one
node-removal perturbation through the pipeline, re-labels the perturbed graph
and renders it to ``pictures/graph.png``.

    python examples/single_graph_walkthrough.py
"""
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.composite_graph_generator import MotifComposite  # noqa: E402
from graph.labeling_functions import MotifLabelingFunction  # noqa: E402
from graph.perturbations import RemoveNodesPerturbation  # noqa: E402
from graph.perturbation_engine import PerturbationPipeline  # noqa: E402
from utils.kg_utils import apply_labeling_result  # noqa: E402
from utils.rng import set_seed  # noqa: E402
from utils.visualize import visualize_graph  # noqa: E402

# Global seed: set once here to control all randomness in the framework
set_seed(42)

gen = MotifComposite(motifs=[["cycle", 4], ["house"], ["cycle", 3]])
graph = gen.generate_graph(num_extra_vertices=2, num_extra_edges=3, start=0)

# Define labeling function with motif order
motif_order = ["house", "cycle_4", "cycle_3"]
labeling = MotifLabelingFunction(motif_order=motif_order)

# Assign expected ground truth before perturbation
apply_labeling_result(graph, labeling.label(graph), "expected_ground_truth")

for node in graph.nodes():
    print(f"Node {node}: {graph.nodes[node]}")

# Apply perturbation (remove 2 nodes) checking for label changes
# Each result is an independent perturbed variant of the original graph
perturbations = [(RemoveNodesPerturbation(num_nodes=2, strategy="motif"), 1)]
pipeline = PerturbationPipeline(
    perturbations=perturbations,
    labeling_functions=[labeling],
    max_iterations=10,
)
results = pipeline.apply_and_check(graph)

if results:
    # Take the first successful perturbation
    result = results[0]
    perturbed_graph = result["perturbed_graph"]
    print("\nPerturbation changes:", result["changes"])
    print("Changed nodes:", result["changed_nodes"])

    # Store observed ground truth after perturbation
    apply_labeling_result(perturbed_graph, labeling.label(perturbed_graph), "observed_ground_truth")

    print("\nNodes:", list(perturbed_graph.nodes()))
    print("Edges:", list(perturbed_graph.edges()))
    print("Node attributes:")
    for node in perturbed_graph.nodes():
        print(f"Node {node}: {perturbed_graph.nodes[node]}")
    print("Connected components:", list(nx.connected_components(perturbed_graph)))

    # Visualize the perturbed graph
    visualize_graph(perturbed_graph, title="Perturbed Graph")
else:
    print("No perturbation caused label changes.")
