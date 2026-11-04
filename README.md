# Absynthe — Synthetic Graph Dataset Framework

Absynthe is a Python framework for generating synthetic graphs out of configurable **motifs**, composing them with several **composition patterns**, **labeling** the result via subgraph isomorphism, applying **reversible perturbations**, and producing reproducible **datasets** for graph-learning experiments. It can also build datasets from **existing graph collections** by ingesting graphs from a folder (GraphML or RDF).

The pipeline is agnostic to the kind of graph a `GraphGenerator` produces: plain undirected graphs (the motif composites) and **multi-relational knowledge graphs** (directed multigraphs with entity types on nodes and relation types on edges, loaded from RDF or sampled from a schema) go through the same labeling, perturbation, reconstruction and dataset machinery.

## Features
- Reusable motif generators — cycle, house, chain, star, gate, grid, clique (see [motifs/](motifs/) and the registry in [motifs/__init__.py](motifs/__init__.py)).
- Composition patterns — sequential, ER, BA, SBM, star, hierarchical — via [graph/composition_engine.py](graph/composition_engine.py).
- Random per-graph motif counts using `IntDistribution` (uniform / normal / poisson) in [utils/distributions.py](utils/distributions.py).
- Flexible labeling: per-node, per-edge, per-graph, or custom combinations via pluggable `LabelingFunction` in [graph/labeling_functions.py](graph/labeling_functions.py) and [interfaces/labeling_function.py](interfaces/labeling_function.py).
- Reversible perturbations — node removal with six strategies (random, motif, degree, centrality, role, by attribute), edge removal, edge addition, edge rewiring — in [graph/perturbations.py](graph/perturbations.py) and [graph/perturbation_strategies.py](graph/perturbation_strategies.py), plus reconstruction in [graph/reconstruction.py](graph/reconstruction.py).
- **Knowledge graphs**: any NetworkX graph class is supported end to end (`nx.Graph`, `nx.DiGraph`, `nx.MultiGraph`, `nx.MultiDiGraph`). Conventions and helpers live in [utils/kg_utils.py](utils/kg_utils.py); a schema-driven synthetic KG generator in [graph/schema_kg_generator.py](graph/schema_kg_generator.py); relation-aware labeling in [graph/kg_labeling_functions.py](graph/kg_labeling_functions.py); triple corruption (relation / head / tail) and relation-filtered edge removal in [graph/kg_perturbations.py](graph/kg_perturbations.py); motif matching honours direction, entity types and relation types. KG datasets are exported as GraphML **and** triples (`.tsv` / `.nt`).
- Pluggable export formats via [graph/exporters.py](graph/exporters.py): GraphML and triples are built in, and a new format is a small `GraphExporter` subclass registered by name and selected with `formats=[...]` (see the "Export formats" section of [Absynthe_GUIDE.md](Absynthe_GUIDE.md)).
- Folder / RDF graph ingestion via [graph/folder_graph_generator.py](graph/folder_graph_generator.py) — supports `.graphml`, `.rdf`, `.ttl`, `.nt`, `.n3`, `.owl`. RDF files become directed multi-relational graphs: predicates are kept as `relation` edge attributes (parallel predicates preserved), `rdf:type` becomes the node `type`, literals become node attributes.
- Optional FastAPI + React web interface for interactive exploration (see [web/](web/)).

## Requirements
- Python 3.8 or newer.
- A virtual environment is recommended (example below).
- Core dependencies are listed in [requirements.txt](requirements.txt) (`networkx`, `matplotlib`, `lxml`, `rdflib`).
- Web-interface dependencies are listed in [requirements-web.txt](requirements-web.txt).

## Installation

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
# macOS / Linux (bash)
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start — generating a dataset

The canonical end-to-end driver is [scripts/gen_dataset.py](scripts/gen_dataset.py). After activating the venv and installing `requirements.txt`:

```bash
python scripts/gen_dataset.py
```

This produces four bundled datasets under `datasets/`:

- `datasets/house/`
- `datasets/cycle_5/`
- `datasets/star_5/`
- `datasets/house_cycle5_star5/`

Each dataset contains the perturbed graph variants (organized by perturbation type) along with a `metadata.json` describing the motifs, labels and the reversible perturbation trace of every variant.

Reproducibility is controlled by a single global seed set through `utils.rng.set_seed(...)` (see [utils/rng.py](utils/rng.py)) at the top of [scripts/gen_dataset.py](scripts/gen_dataset.py). Change that value to obtain different random instances.

### Provenance stored with each dataset

Next to `metadata.json`, every dataset gets a `config.json` with the seed, the configuration that produced it and a description of the live components, so a dataset can be audited or regenerated from its own folder given the framework version (see the "Metadata Format" section of [Absynthe_GUIDE.md](Absynthe_GUIDE.md)):

```json
{
  "seed": 42,
  "config": { "...the JSON dataset entry / web request that produced it..." },
  "components": {"graph_generator": {"class": "RandomMotifComposite", "params": {"…": "…"}}, "labeling_functions": [], "perturbations": []},
  "generation": {"num_graphs": 300, "graph_kwargs": {"…": "…"}, "max_perturbation_iterations": 15, "formats": ["graphml", "triples"], "output_dir": "datasets/house"},
  "provenance": {"config_path": "…/configs/default.json", "suite_index": 0, "suite_size": 4, "suite_seed": 42, "seed_scope": "suite"}
}
```

Seeding has two scopes. The top-level `seed` of a JSON suite is set once, so a dataset that is not the first of its suite also depends on the RNG state left by the datasets before it (`seed_scope: "suite"`). Give a dataset entry (or the `shared` block) its own `seed` to re-seed right before it (`seed_scope: "dataset"`); such a dataset can then be regenerated on its own by replaying the `config` stored in its `config.json`.

### Reproducible runs via JSON config

The same configuration can be expressed in a JSON file and passed to the script, so an experiment can be reproduced from a single artifact:

```bash
python scripts/gen_dataset.py configs/default.json   # explicit path
python scripts/gen_dataset.py --json                 # shorthand for configs/default.json
python scripts/gen_dataset.py                        # no args → in-script DATASET_CONFIGS
```

The JSON describes a suite of datasets sharing a `shared` block of defaults (seed, perturbations, labeling, generation parameters) which each entry in `datasets[]` can selectively override. The schema mirrors the `DatasetGenerateRequest` model used by the web backend ([web/backend/models/dataset_models.py](web/backend/models/dataset_models.py)), so a config produced from the GUI is interchangeable with one generated by hand. See [configs/default.json](configs/default.json) for a working template that reproduces the four bundled datasets above.

### Knowledge-graph datasets

Each dataset entry names exactly one graph source: `motifs` (synthetic motif composites), `folder_source` (GraphML / RDF files) or `kg_schema` (synthetic knowledge graph sampled from entity types and relation specs). [configs/kg_example.json](configs/kg_example.json) builds two KG datasets, one from a schema and one from the RDF file in [examples/rdf/](examples/rdf/):

```bash
python scripts/gen_kg_dataset.py                        # Python mode: DATASET_CONFIGS in the script
python scripts/gen_kg_dataset.py configs/kg_example.json   # JSON mode → datasets_kg/synthetic_kg, datasets_kg/rdf_folder_kg
```

[scripts/gen_kg_dataset.py](scripts/gen_kg_dataset.py) is the KG counterpart of `gen_dataset.py`: same pipeline and CLI, but the graph sources are `SchemaKGGenerator` and `FolderGraphGenerator` over RDF files, and the labeling / perturbations are the relation-aware ones.

It uses `entity_type_labeling` (node labels from entity types and relation patterns, edge labels from relation types), relation-filtered edge removal (`remove_edges` with `relations`), `corrupt_triples` and attribute-based node removal. Every variant is saved as GraphML plus a triples file, and `metadata.json` records the graph kind and the reversible changes (with edge keys). The programmatic API is documented in the "Knowledge graphs" section of [Absynthe_GUIDE.md](Absynthe_GUIDE.md); [tests/test_kg.py](tests/test_kg.py) exercises it end to end.

For short narrative examples that walk through the building blocks on a single graph (composition, labeling, one perturbation, reconstruction, visualization), see [examples/single_graph_walkthrough.py](examples/single_graph_walkthrough.py) for motif graphs and [examples/kg_walkthrough.py](examples/kg_walkthrough.py) for knowledge graphs.

For the full API walkthrough — motif tables, composition parameters, custom labelers and custom motifs — see the companion guide [Absynthe_GUIDE.md](Absynthe_GUIDE.md).

## Tests and tools

```bash
pip install pytest
pytest tests/                     # KG support, labeling results, dataset smoke test
```

Each test module also runs standalone (`python tests/test_kg.py`). The plain-graph regression test in `tests/test_kg.py` is skipped unless `ABSYNTHE_BASELINE` points to a reference run of `configs/default.json` with `num_graphs=20`.

[scripts/reconstruct_variant.py](scripts/reconstruct_variant.py) loads one perturbed variant of a dataset, inverts its reversible changes and renders both graphs:

```bash
python scripts/reconstruct_variant.py datasets/house 0
python scripts/reconstruct_variant.py datasets_kg/synthetic_kg 3
```

## Project structure
- [graph/](graph/) — composite generators, composition engine, dataset orchestrator, export formats ([graph/exporters.py](graph/exporters.py)), labeling functions, perturbations and strategies, reconstruction, folder/RDF loader; knowledge-graph components in [graph/schema_kg_generator.py](graph/schema_kg_generator.py), [graph/kg_labeling_functions.py](graph/kg_labeling_functions.py) and [graph/kg_perturbations.py](graph/kg_perturbations.py).
- [motifs/](motifs/) — built-in motif generators and the registry in [motifs/__init__.py](motifs/__init__.py).
- [interfaces/](interfaces/) — abstract base classes (`GraphGenerator`, `MotifGenerator`, `LabelingFunction`, `Perturbation`, `GraphExporter`), data models (`LabelingResult`, `PerturbationHint`) and the `GraphLike` alias ([interfaces/graph_types.py](interfaces/graph_types.py)).
- [utils/](utils/) — global RNG ([utils/rng.py](utils/rng.py)), `IntDistribution` ([utils/distributions.py](utils/distributions.py)), graph helpers, the knowledge-graph conventions and graph-class-agnostic edge helpers in [utils/kg_utils.py](utils/kg_utils.py), and the matplotlib renderer in [utils/visualize.py](utils/visualize.py).
- [configs/](configs/) — JSON suites: [configs/default.json](configs/default.json) (motif datasets), [configs/kg_example.json](configs/kg_example.json) (knowledge graphs). [examples/rdf/](examples/rdf/) holds a small Turtle knowledge graph.
- [web/](web/) — FastAPI backend + React/TypeScript frontend.
- [tests/](tests/) — pytest suite: [tests/test_kg.py](tests/test_kg.py), [tests/test_exporters.py](tests/test_exporters.py), [tests/test_metadata.py](tests/test_metadata.py), [tests/test_edge_labels.py](tests/test_edge_labels.py), [tests/test_dataset.py](tests/test_dataset.py).
- [examples/](examples/) — [examples/single_graph_walkthrough.py](examples/single_graph_walkthrough.py), [examples/kg_walkthrough.py](examples/kg_walkthrough.py) and the sample RDF knowledge graph in [examples/rdf/](examples/rdf/).
- [scripts/](scripts/) — command-line entry points: [scripts/gen_dataset.py](scripts/gen_dataset.py), [scripts/gen_kg_dataset.py](scripts/gen_kg_dataset.py), [scripts/reconstruct_variant.py](scripts/reconstruct_variant.py), [scripts/visualize_graph_dataset.py](scripts/visualize_graph_dataset.py). Run them from the repository root.
- Companion docs: [Absynthe_GUIDE.md](Absynthe_GUIDE.md).

## Web interface

### Development (two terminals)
**Terminal 1 — backend (from project root):**
```bash
pip install -r requirements-web.txt
uvicorn web.backend.app:app --reload --port 8000
```

**Terminal 2 — frontend:**
```bash
cd web/frontend
npm install
npm run dev          # → http://localhost:5173
```

### Production (single command)
```bash
pip install -r requirements-web.txt
python -m web        # builds the frontend (if needed) and serves the SPA + API at :8000
```

The `python -m web` entry point is implemented in [web/__main__.py](web/__main__.py) and accepts `--host`, `--port`, `--reload` and `--build` flags.

### Docker
```bash
docker compose up    # → http://localhost:8000
```

See [docker-compose.yml](docker-compose.yml) and [Dockerfile](Dockerfile) for the build configuration.

## License
This repository is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [GPL-3.0 license](GPL-3.0%20license) file in the project root for the full text.
