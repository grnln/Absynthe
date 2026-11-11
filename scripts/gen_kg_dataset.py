"""Generate example knowledge-graph (KG) datasets with Absynthe.

This is the KG counterpart of `gen_dataset.py`. The pipeline is the same
(`GraphDatasetGenerator` + labeling functions + reversible perturbations);
what changes is the kind of graph the `GraphGenerator` produces: directed
multi-relational graphs (`nx.MultiDiGraph`) whose nodes carry an entity
`type` and whose edges carry a `relation`.

Two graph sources are shown:

  - `SchemaKGGenerator`: synthetic KGs sampled from an entity/relation schema.
  - `FolderGraphGenerator`: real KGs loaded from RDF files in a folder
    (predicates -> `relation`, `rdf:type` -> node `type`, literals -> node
    attributes).

Two configuration sources are supported, like in `gen_dataset.py`:

  - **Python mode** (default): edit the `DATASET_CONFIGS` block below.
  - **JSON mode**: pass a config file path, or `--json` to load
    `configs/kg_example.json`.

Run from the repository root (`datasets_kg/` and `examples/rdf` resolve
against the current directory):

    python scripts/gen_kg_dataset.py                     # Python mode (DATASET_CONFIGS)
    python scripts/gen_kg_dataset.py configs/kg_example.json
    python scripts/gen_kg_dataset.py --json              # equivalent to the previous line

Each dataset ends up under `datasets_kg/<name>/` with:

    originals/graph_<i>.graphml   base KG (+ graph_<i>.tsv or .nt with its triples)
    <perturbation>/graph_<n>.graphml + triples   perturbed variants
    metadata.json                 per variant: reversible changes (with edge
                                  keys), changed node/edge labels, graph_kind,
                                  triples paths
    config.json                   seed, configuration and components of the run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen_dataset import _datasets_from_json, run_pipeline  # noqa: E402
from graph.folder_graph_generator import (  # noqa: E402
    ExhaustionPolicy,
    FolderGraphGenerator,
    IterationOrder,
)
from graph.kg_labeling_functions import EntityTypeLabelingFunction  # noqa: E402
from graph.kg_perturbations import CorruptTriplesPerturbation  # noqa: E402
from graph.perturbations import RemoveEdgesPerturbation, RemoveNodesPerturbation  # noqa: E402
from graph.schema_kg_generator import SchemaKGGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# Python-mode configuration (used when no JSON file is provided)
# ---------------------------------------------------------------------------

SEED = 7
NUM_GRAPHS = 50
DATASETS_ROOT = "datasets_kg"
CLEAN_OUTPUT = True

# Schema of the synthetic KG: entity types with their cardinality, and
# relation types with domain, range and either a probability per candidate
# pair (`p`) or an exact number of triples (`count`).
ACADEMIC_SCHEMA = {
    "entity_types": {"Person": 12, "Company": 4, "Paper": 8},
    "relations": [
        {"name": "works_at", "domain": "Person", "range": "Company", "p": 0.25},
        {"name": "wrote",    "domain": "Person", "range": "Paper",   "count": 10},
        {"name": "cites",    "domain": "Paper",  "range": "Paper",   "p": 0.15},
        {"name": "knows",    "domain": "Person", "range": "Person",  "p": 0.10},
    ],
}

DATASET_CONFIGS = [
    # Synthetic KG sampled from a schema.
    {
        "name": "synthetic_kg",
        "source": {"type": "schema", **ACADEMIC_SCHEMA},
        "num_graphs": NUM_GRAPHS,
        # Node labels derived from the relations a node takes part in
        # (later patterns win); edge labels are the relation types.
        "patterns": {
            "author":   {"relation": "wrote",    "direction": "out"},
            "employee": {"relation": "works_at", "direction": "out"},
        },
        "relation_to_remove": "works_at",
        "node_type_to_remove": "Company",
    },
    # Real KG(s) read from RDF files. examples/rdf/ contains one small
    # Turtle file; with exhaustion_policy=CYCLE it is reused for every base
    # graph (different perturbations each time).
    {
        "name": "rdf_folder_kg",
        "source": {"type": "folder", "folder_path": "examples/rdf"},
        "num_graphs": 3,
        "patterns": {
            "author":   {"relation": "ex:wrote",   "direction": "out"},
            "employee": {"relation": "ex:worksAt", "direction": "out"},
        },
        "relation_to_remove": "ex:worksAt",
        "node_type_to_remove": "ex:Company",
    },
]


# ---------------------------------------------------------------------------
# Python-mode adapter: turn DATASET_CONFIGS into the runtime form expected by
# `run_pipeline` (shared with gen_dataset.py).
# ---------------------------------------------------------------------------

def _build_graph_generator(source: Dict[str, Any]):
    if source["type"] == "schema":
        return SchemaKGGenerator(
            entity_types=source["entity_types"],
            relations=source["relations"],
        )
    if source["type"] == "folder":
        return FolderGraphGenerator(
            folder_path=source["folder_path"],
            iteration_order=IterationOrder.SEQUENTIAL,
            exhaustion_policy=ExhaustionPolicy.CYCLE,
        )
    raise ValueError(f"Unknown source type: {source['type']}")


def _datasets_from_python_configs() -> List[Dict[str, Any]]:
    datasets: List[Dict[str, Any]] = []
    for config in DATASET_CONFIGS:
        name = config["name"]
        graph_gen = _build_graph_generator(config["source"])
        labeling = [EntityTypeLabelingFunction(patterns=config["patterns"])]
        perturbations = [
            # Remove only edges of one relation type (heads lose the
            # "employee" label -> accepted by the pipeline).
            (RemoveEdgesPerturbation(
                p_remove=0.3,
                relations=[config["relation_to_remove"]],
                folder_name="remove_" + config["relation_to_remove"].replace(":", "_"),
            ), 1),
            # Negative-sampling style corruption: swap the relation type of
            # two triples (changes edge labels).
            (CorruptTriplesPerturbation(num_triples=2, mode="relation", folder_name="corrupt_relation"), 1),
            # Swap the tail entity of one triple (same entity type).
            (CorruptTriplesPerturbation(num_triples=1, mode="tail", folder_name="corrupt_tail"), 1),
            # Remove one entity of a given type with all its triples.
            (RemoveNodesPerturbation(
                num_nodes=1,
                strategy="by_attribute",
                params={"attr": "type", "value": config["node_type_to_remove"]},
                folder_name="remove_nodes",
            ), 1),
        ]
        datasets.append({
            "name": name,
            "graph_generator": graph_gen,
            "labeling": labeling,
            "perturbations": perturbations,
            "output_dir": f"{DATASETS_ROOT}/{name}",
            "max_perturbation_iterations": 15,
            # KG generators ignore the motif-specific kwargs, so only
            # num_graphs is needed here.
            "gen_kwargs": {"num_graphs": config["num_graphs"]},
        })
    return datasets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_JSON_PATH = str(REPO_ROOT / "configs" / "kg_example.json")


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
        roots = {Path(d["output_dir"]).parent.as_posix() for d in datasets}
        datasets_root = roots.pop() if len(roots) == 1 else DATASETS_ROOT
        run_pipeline(
            datasets,
            seed=seed,
            clean_output=clean_output,
            datasets_root=datasets_root,
        )


if __name__ == "__main__":
    main()
