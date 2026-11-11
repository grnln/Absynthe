"""Smoke test for ``GraphDatasetGenerator`` on plain motif graphs.

Run with ``pytest tests/`` or ``python tests/test_dataset.py``. Output goes
to ``test_output/`` in the repository root (git-ignored).
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from graph.dataset_generator import GraphDatasetGenerator  # noqa: E402
from graph.dataset_metadata import load_dataset_config, load_dataset_metadata  # noqa: E402
from graph.composite_graph_generator import MotifComposite  # noqa: E402
from graph.labeling_functions import MotifLabelingFunction  # noqa: E402
from graph.perturbations import RemoveNodesPerturbation, EdgePerturbation  # noqa: E402


def test_graph_dataset_generator():
    """
    Test the GraphDatasetGenerator with the new pluggable architecture.
    """
    output_dir = str(REPO_ROOT / "test_output")

    # Clean up previous test output if it exists
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(output_dir)

    # 1. Create pluggable components
    graph_gen = MotifComposite(motifs=[["cycle", 4], ["house"], ["cycle", 3]])

    labeling = [MotifLabelingFunction(motif_order=["cycle_4", "house", "cycle_3"])]

    perturbations = [
        (RemoveNodesPerturbation(num_nodes=2, strategy="motif"), 2),
        (EdgePerturbation(p_remove=0.1), 1),
    ]

    # 2. Create orchestrator with all components
    generator = GraphDatasetGenerator(
        graph_generator=graph_gen,
        labeling_functions=labeling,
        perturbations=perturbations,
        output_dir=output_dir,
        max_perturbation_iterations=5,
    )

    # 3. Generate the dataset
    metadata = generator.generate_dataset(num_graphs=3)

    # 4. Assertions
    assert os.path.exists(output_dir), "Output directory was not created."
    assert os.path.exists(os.path.join(output_dir, "originals")), "Originals directory was not created."
    for perturbation, _ in perturbations:
        assert os.path.exists(os.path.join(output_dir, perturbation.folder_name)), (
            f"Perturbation directory '{perturbation.folder_name}' was not created."
        )
    assert os.path.exists(os.path.join(output_dir, "metadata.json")), "Metadata file was not created."

    # metadata.json holds the variants; config.json the provenance of the run
    assert [e["graph_id"] for e in load_dataset_metadata(output_dir)] == [e["graph_id"] for e in metadata]
    cfg = load_dataset_config(output_dir)
    assert cfg == generator.last_config
    assert cfg["components"]["graph_generator"]["class"] == "MotifComposite"
    assert [p["count"] for p in cfg["components"]["perturbations"]] == [2, 1]
    assert cfg["generation"]["num_graphs"] == 3

    # Print metadata for verification
    print("Generated Metadata:")
    for entry in metadata:
        print(entry)

    # Every entry is a perturbed variant referencing its original
    originals = {m["original_graph_path"] for m in metadata}
    print(f"\nOriginal graphs referenced: {len(originals)}")
    print(f"Perturbed variants: {len(metadata)}")


def test_progress_callback_reports_every_base_graph(tmp_path):
    """``progress_callback`` fires once per base graph, with and without perturbations."""
    graph_gen = MotifComposite(motifs=[["cycle", 4], ["house"]])
    labeling = [MotifLabelingFunction(motif_order=["cycle_4", "house"])]

    calls = []
    generator = GraphDatasetGenerator(
        graph_generator=graph_gen,
        labeling_functions=labeling,
        perturbations=[(EdgePerturbation(p_remove=0.2), 1)],
        output_dir=str(tmp_path / "with_perturbations"),
        max_perturbation_iterations=3,
    )
    generator.generate_dataset(num_graphs=3, progress_callback=lambda done, total: calls.append((done, total)))
    assert calls == [(1, 3), (2, 3), (3, 3)]

    # Without perturbations the loop short-circuits; progress must still be reported.
    calls.clear()
    generator = GraphDatasetGenerator(graph_generator=graph_gen, output_dir=str(tmp_path / "plain"))
    generator.generate_dataset(num_graphs=2, progress_callback=lambda done, total: calls.append((done, total)))
    assert calls == [(1, 2), (2, 2)]


if __name__ == "__main__":
    test_graph_dataset_generator()
