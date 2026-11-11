"""Render one GraphML file of a dataset to PNG, colouring nodes by label.

    python scripts/visualize_graph_dataset.py <graph.graphml> [output.png]
    python scripts/visualize_graph_dataset.py datasets/house/originals/graph_0.graphml

Defaults to ``test_output/originals/graph_0.graphml`` (the file written by
``tests/test_dataset.py``) and ``pictures/graph_dataset.png``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.folder_graph_generator import load_graph_file  # noqa: E402
from utils.visualize import visualize_graph  # noqa: E402


def main(graph_path: str, output: str) -> None:
    graph = load_graph_file(graph_path)

    print("Node attributes:")
    for node in graph.nodes(data=True):
        print(node)

    visualize_graph(graph, title=Path(graph_path).stem, filename=output)
    print(f"Saved {output}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_output/originals/graph_0.graphml"
    out = sys.argv[2] if len(sys.argv) > 2 else "pictures/graph_dataset.png"
    main(path, out)
