"""Load a perturbed variant from a dataset, reconstruct its original, and visualize both.

    python scripts/reconstruct_variant.py <dataset_dir> <graph_id>
    python scripts/reconstruct_variant.py datasets/house 0
    python scripts/reconstruct_variant.py datasets_kg/synthetic_kg 3

Works for plain datasets and for knowledge-graph datasets (directed
multigraphs are restored from GraphML before reconstruction). The two PNGs
are written under ``pictures/reconstruction/``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.dataset_metadata import find_variant, load_dataset_config, load_dataset_metadata  # noqa: E402
from graph.folder_graph_generator import load_graph_file  # noqa: E402
from graph.reconstruction import normalize_changes, reconstruct_original  # noqa: E402
from utils.visualize import visualize_graph  # noqa: E402


def main(dataset_dir: str = "datasets/house", graph_id: int = 0) -> None:
    ds = Path(dataset_dir)
    entry = find_variant(load_dataset_metadata(ds), graph_id)
    config = load_dataset_config(ds)

    # load_graph_file restores directed / multigraph variants (knowledge graphs)
    perturbed = load_graph_file(entry["graph_path"])
    changes = normalize_changes(entry["perturbation_info"]["changes"], perturbed)
    reconstructed = reconstruct_original(perturbed, changes)

    print(f"Dataset:      {ds}")
    if config is not None:
        print(f"Seed:         {config.get('seed')}")
    print(f"Variant id:   {graph_id}  (base_graph_id={entry['base_graph_id']})")
    print(f"Graph kind:   {entry.get('graph_kind', 'plain undirected')}")
    print(f"Perturbed:    {len(perturbed.nodes())} nodes, {len(perturbed.edges())} edges")
    print(f"Reconstructed:{len(reconstructed.nodes())} nodes, {len(reconstructed.edges())} edges")
    print(f"Changes summary: "
          f"removed_nodes={len(changes.get('removed_nodes', []))}, "
          f"removed_edges={len(changes.get('removed_edges', []))}, "
          f"added_edges={len(changes.get('added_edges', []))}, "
          f"corrupted_triples={len(changes.get('corrupted_triples', []))}")

    for node in reconstructed.nodes():
        if "expected_ground_truth" in reconstructed.nodes[node]:
            reconstructed.nodes[node]["label"] = reconstructed.nodes[node]["expected_ground_truth"]

    out_dir = Path("pictures/reconstruction")
    visualize_graph(
        perturbed,
        title=f"Perturbed graph_{graph_id}",
        filename=str(out_dir / f"graph_{graph_id}_perturbed.png"),
    )
    visualize_graph(
        reconstructed,
        title=f"Reconstructed original from graph_{graph_id}",
        filename=str(out_dir / f"graph_{graph_id}_reconstructed.png"),
    )
    print(f"Saved visualizations under {out_dir}/")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "datasets/house"
    gid = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(ds, gid)
