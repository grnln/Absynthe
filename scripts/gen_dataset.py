"""Generate the canonical Absynthe datasets.

Two configuration sources are supported:

  - **Python mode** (default when no CLI argument is given): edit the
    `DATASET_CONFIGS` block below, just like before.
  - **JSON mode**: pass a config file path as a positional argument, or use
    `--json` to load `configs/default.json`. The JSON format mirrors the
    `DatasetGenerateRequest` model used by the web backend, so an experiment
    can be reproduced from a single file.

Run from the repository root (relative paths such as `datasets/` and
`configs/` resolve against the current directory):

    python scripts/gen_dataset.py                    # Python mode (DATASET_CONFIGS)
    python scripts/gen_dataset.py configs/default.json
    python scripts/gen_dataset.py --json             # equivalent to the previous line
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.rng import set_seed, reset_rng  # noqa: E402
from graph.random_motif_composite import RandomMotifComposite  # noqa: E402
from graph.dataset_generator import GraphDatasetGenerator  # noqa: E402
from graph.labeling_functions import MotifLabelingFunction  # noqa: E402
from graph.perturbations import (  # noqa: E402
    RemoveNodesPerturbation,
    RemoveEdgesPerturbation,
    EdgePerturbation,
)
from utils.distributions import IntDistribution  # noqa: E402


# ---------------------------------------------------------------------------
# Python-mode configuration (used when no JSON file is provided)
# ---------------------------------------------------------------------------

SEED = 42
NUM_GRAPHS = 300
DATASETS_ROOT = "datasets"
CLEAN_OUTPUT = True

# Normal distribution centered at 3 with std=1 -> mostly 2-4 motifs per graph
count_dist = IntDistribution("normal", {"mean": 3, "std": 1})

DATASET_CONFIGS = [
    # Single-motif datasets
    {
        "name": "house",
        "motifs": [
            (["house"], 1, count_dist),
        ],
        "motif_order": ["house"],
    },
    {
        "name": "cycle_5",
        "motifs": [
            (["cycle", 5], 1, count_dist),
        ],
        "motif_order": ["cycle_5"],
    },
    {
        "name": "star_5",
        "motifs": [
            (["star", 5], 1, count_dist),
        ],
        "motif_order": ["star_5"],
    },
    # Multi-motif dataset
    {
        "name": "house_cycle5_star5",
        "motifs": [
            (["house"], 1, IntDistribution("normal", {"mean": 2, "std": 1})),
            (["cycle", 5], 1, IntDistribution("normal", {"mean": 2, "std": 1})),
            (["star", 5], 1, IntDistribution("normal", {"mean": 2, "std": 1})),
        ],
        "motif_order": ["house", "cycle_5", "star_5"],
    },
]


# ---------------------------------------------------------------------------
# Python-mode adapter: turn DATASET_CONFIGS into the runtime form expected by
# `run_pipeline`.
# ---------------------------------------------------------------------------

def _datasets_from_python_configs() -> List[Dict[str, Any]]:
    datasets: List[Dict[str, Any]] = []
    for config in DATASET_CONFIGS:
        name = config["name"]
        graph_gen = RandomMotifComposite(motif_configs=config["motifs"])
        labeling = [MotifLabelingFunction(motif_order=config["motif_order"])]
        perturbations = [
            (RemoveNodesPerturbation(num_nodes=1, strategy="motif", folder_name="remove_nodes"), 1),
            (EdgePerturbation(p_remove=0.1, p_add=0.05, folder_name="edge_perturbation"), 1),
            (RemoveEdgesPerturbation(p_remove=0.15, folder_name="remove_edges"), 1),
        ]
        datasets.append({
            "name": name,
            "graph_generator": graph_gen,
            "labeling": labeling,
            "perturbations": perturbations,
            "output_dir": f"{DATASETS_ROOT}/{name}",
            "max_perturbation_iterations": 15,
            "gen_kwargs": {
                "num_graphs": NUM_GRAPHS,
                "num_extra_vertices": 0,
                "num_extra_edges": 0,
                "composition": "sequential",
            },
        })
    return datasets


# ---------------------------------------------------------------------------
# JSON-mode adapter: load a JSON suite file and translate each entry to the
# same runtime form via the shared web/backend builders.
# ---------------------------------------------------------------------------

def _datasets_from_json(path: str | Path) -> tuple[List[Dict[str, Any]], int | None, bool]:
    from utils.dataset_config import load_dataset_config
    from web.backend.services.registry import (
        build_graph_generator,
        build_labeling_functions,
        build_perturbations,
        normalize_composition_params,
    )

    suite = load_dataset_config(path)
    config_path = Path(path).resolve().as_posix()
    datasets: List[Dict[str, Any]] = []
    for name, request in zip(suite.names, suite.datasets):
        datasets.append({
            "name": name,
            "graph_generator": build_graph_generator(request),
            "labeling": build_labeling_functions(request.labeling_functions),
            "perturbations": build_perturbations(request.perturbations),
            "output_dir": request.output_dir,
            "max_perturbation_iterations": request.max_perturbation_iterations,
            "formats": request.formats,
            # Recorded in metadata.json so the dataset can be replayed from
            # its own folder. A per-dataset `seed` (set in the JSON entry or
            # in `shared`) re-seeds the RNG right before that dataset.
            "config": request.model_dump(mode="json"),
            "seed": request.seed,
            "config_path": config_path,
            "gen_kwargs": {
                "num_graphs": request.num_graphs,
                "num_extra_vertices": request.num_extra_vertices,
                "num_extra_edges": request.num_extra_edges,
                "composition": request.composition,
                "composition_params": normalize_composition_params(request.composition_params),
            },
        })
    return datasets, suite.seed, suite.clean_output


# ---------------------------------------------------------------------------
# Generation pipeline (shared by both modes).
# ---------------------------------------------------------------------------

def run_pipeline(
    datasets: List[Dict[str, Any]],
    *,
    seed: int | None,
    clean_output: bool,
    datasets_root: str,
) -> None:
    if seed is not None:
        set_seed(seed)

    root = Path(datasets_root)
    if clean_output and root.exists():
        shutil.rmtree(root)
        print(f"Cleared existing {root}/ folder.")

    for index, entry in enumerate(datasets):
        name = entry["name"]
        print(f"Generating dataset: {name}")

        # A dataset-level seed makes this dataset reproducible on its own;
        # otherwise it inherits the suite seed and the RNG state left by the
        # datasets generated before it.
        dataset_seed = entry.get("seed")
        if dataset_seed is not None:
            set_seed(dataset_seed)
            seed_scope = "dataset"
        else:
            seed_scope = "suite" if seed is not None else None

        provenance: Dict[str, Any] = {
            "dataset_name": name,
            "suite_index": index,
            "suite_size": len(datasets),
            "suite_seed": seed,
            "seed_scope": seed_scope,
        }
        if entry.get("config_path"):
            provenance["config_path"] = entry["config_path"]

        generator = GraphDatasetGenerator(
            graph_generator=entry["graph_generator"],
            labeling_functions=entry["labeling"],
            perturbations=entry["perturbations"],
            output_dir=entry["output_dir"],
            max_perturbation_iterations=entry["max_perturbation_iterations"],
            formats=entry.get("formats"),
            seed=dataset_seed if dataset_seed is not None else seed,
            config=entry.get("config"),
            provenance=provenance,
        )

        metadata = generator.generate_dataset(**entry["gen_kwargs"])
        num_graphs = entry["gen_kwargs"]["num_graphs"]
        print(f"  -> {len(metadata)} perturbed variants from {num_graphs} base graphs")

        # To render every graph, use scripts/reconstruct_variant.py on a single
        # variant or utils.visualize.visualize_graph over the metadata entries.

    if seed is not None:
        reset_rng()
    print(f"Done. Datasets saved under {datasets_root}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_JSON_PATH = str(REPO_ROOT / "configs" / "default.json")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Optional path to a JSON config file. If omitted, uses the in-script DATASET_CONFIGS.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=f"Use JSON mode with the default config at {DEFAULT_JSON_PATH}.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    config_path: str | None = args.config
    if config_path is None and args.json:
        config_path = DEFAULT_JSON_PATH

    if config_path is None:
        datasets = _datasets_from_python_configs()
        run_pipeline(
            datasets,
            seed=SEED,
            clean_output=CLEAN_OUTPUT,
            datasets_root=DATASETS_ROOT,
        )
    else:
        datasets, seed, clean_output = _datasets_from_json(config_path)
        # The JSON's per-dataset output_dirs are absolute or relative paths set
        # by the loader; pick the common parent as the cleanup root if every
        # dataset shares one, otherwise fall back to "datasets".
        roots = {Path(d["output_dir"]).parent.as_posix() for d in datasets}
        datasets_root = roots.pop() if len(roots) == 1 else "datasets"
        run_pipeline(
            datasets,
            seed=seed,
            clean_output=clean_output,
            datasets_root=datasets_root,
        )


if __name__ == "__main__":
    main()
