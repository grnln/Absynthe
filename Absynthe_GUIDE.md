# Absynthe Framework Guide

Absynthe is a Python framework for generating synthetic graph datasets with controllable structural motifs, labeling, and perturbations. It is designed for benchmarking graph-level machine learning tasks where ground-truth labels and controlled perturbations are needed.

## Installation

Absynthe requires Python 3.10+ and the following dependencies:

```bash
pip install -r requirements.txt
```

Core dependency: `networkx`. Optional: `matplotlib` (for visualization).

---

## Core Concepts

- **Motif**: A small structural subgraph pattern (cycle, house, chain, star, gate, grid, clique).
- **Composite graph**: A graph built by composing multiple motif instances connected via a composition pattern.
- **Labeling**: Assigning labels to nodes based on which motif they belong to.
- **Perturbation**: A controlled modification to a graph (removing nodes, adding/removing edges).
- **Dataset**: A collection of perturbed graph variants together with reversible perturbation metadata — each variant references its base graph (saved under `originals/`) and can also be reconstructed from the metadata alone.
- **Knowledge graph**: A directed multigraph whose nodes carry an entity `type` and whose edges carry a `relation`. Every component below the `GraphGenerator` (labeling, perturbations, reconstruction, datasets, web API) accepts any NetworkX graph class, so knowledge graphs loaded from RDF or sampled from a schema go through the same pipeline as motif composites. See §12.

---

## 1. Generating a Single Graph

Use `MotifComposite` to build a graph from a list of motif specifications.

```python
from graph.composite_graph_generator import MotifComposite

gen = MotifComposite(motifs=[
    ["cycle", 4],   # cycle with 4 nodes
    ["house"],      # house motif (5 nodes, no params)
    ["cycle", 3],   # cycle with 3 nodes
])

graph = gen.generate_graph(
    num_extra_vertices=2,   # random vertices connected to existing nodes
    num_extra_edges=3,      # random edges between existing nodes
    composition="sequential",  # how motifs are connected
)
```

Each node in the generated graph has attributes:
- `motif`: the motif type (e.g. `"cycle_4"`, `"house"`)
- `motif_id`: unique instance identifier (e.g. `"cycle_4_0"`)

### Available Motifs

| Type    | Parameter     | Default | Min | Description                          |
|---------|---------------|---------|-----|--------------------------------------|
| `cycle` | `len_cycle`   | 4       | 3   | Ring of n nodes                      |
| `house` | *(none)*      | —       | —   | 5-node house (square + triangle)     |
| `chain` | `length`      | 3       | 2   | Linear path of n nodes               |
| `star`  | `num_leaves`  | 3       | 1   | Central hub with n leaf nodes        |
| `gate`  | `arm_length`  | 1       | 1   | Entry/exit with two parallel arms    |
| `grid`  | `rows`, `cols`| 3, 3    | 2, 2| Rectangular lattice of rows x cols   |
| `clique`| `num_nodes`   | 4       | 2   | Complete graph K_n                   |

Parameters are passed positionally after the type name:

```python
["cycle", 5]      # len_cycle=5
["star", 4]       # num_leaves=4
["gate", 2]       # arm_length=2
["chain", 6]      # length=6
["grid", 2, 3]    # rows=2, cols=3
["clique", 5]     # num_nodes=5
["house"]         # no parameters
```

The same motifs can be used as labels in `MotifLabelingFunction` by name, with the parameters joined by `_` (e.g. `"cycle_4"`, `"grid_2_3"`, `"clique_5"`).

### Composition Patterns

The `composition` parameter controls how motifs are connected to each other:

| Pattern        | Description                            | Parameters                                  |
|----------------|----------------------------------------|---------------------------------------------|
| `sequential`   | Linear chain: motif 0—1—2—...         | *(none)*                                    |
| `er`           | Erdos-Renyi random edges               | `p` (float): edge probability               |
| `ba`           | Barabasi-Albert preferential attachment | `m` (int): edges per new node               |
| `sbm`          | Stochastic Block Model                 | `p_in`, `p_out` (float), `blocks` (list)    |
| `star`         | One central motif connected to all     | `center` (int): center motif index           |
| `hierarchical` | Motifs grouped into clusters           | `groups` (int): number of groups             |

```python
graph = gen.generate_graph(
    composition="er",
    composition_params={"p": 0.5},
)
```

---

## 2. Motif Count (Fixed and Random)

Each motif entry can specify how many instances to include.

### Fixed Count

```python
gen = MotifComposite(motifs=[
    ["house"],       # 1 instance (default)
    ["house"],       # another instance — or use count below
])

# Equivalent: repeat programmatically
motifs = [["cycle", 4]] * 3 + [["house"]] * 2
gen = MotifComposite(motifs=motifs)
```

### Random Count with Distributions

Use `RandomMotifComposite` when you want a different number of motif instances per graph (useful for dataset generation):

```python
from graph.random_motif_composite import RandomMotifComposite
from utils.distributions import IntDistribution

gen = RandomMotifComposite(motif_configs=[
    (["cycle", 4], 1, None),  # fixed: always 1 cycle
    (["house"], 1, IntDistribution("normal", {"mean": 4, "std": 0.5})),  # ~4 houses per graph
])

# Each call produces a graph with a different number of houses
graph1 = gen.generate_graph(composition="sequential")
graph2 = gen.generate_graph(composition="sequential")
```

Each tuple in `motif_configs` is `(motif_spec, fixed_count, distribution)`:
- When `distribution` is `None`, `fixed_count` is used.
- When `distribution` is provided, it overrides `fixed_count` and is sampled on each call.

### Supported Distributions

| Type      | Parameters              | Description                          |
|-----------|-------------------------|--------------------------------------|
| `uniform` | `min`, `max`            | Equally likely integer in [min, max] |
| `normal`  | `mean`, `std`           | Gaussian, rounded to nearest integer |
| `poisson` | `lambda`                | Poisson-distributed count            |

All distributions clamp the result to `>= 1` by default. You can change the lower bound:

```python
from utils.distributions import IntDistribution, sample_int
from utils.rng import get_rng

dist = IntDistribution("uniform", {"min": 0, "max": 10})
value = sample_int(dist, get_rng(), min_value=0)  # allows 0
```

---

## 3. Reproducibility (Seed)

Use the global RNG module to make all framework randomness deterministic:

```python
from utils.rng import set_seed, reset_rng

set_seed(42)

# All subsequent graph generation, perturbations, etc. are reproducible
graph = gen.generate_graph()

reset_rng()  # revert to default unseeded randomness
```

`get_seed()` returns the seed currently in force (`None` when unseeded). `GraphDatasetGenerator` records it in `config.json` together with the configuration, so a dataset documents how to regenerate itself given the framework version (see §7, "Metadata Format").

---

## 4. Labeling

`MotifLabelingFunction` assigns labels to nodes based on motif subgraph membership. It detects motif instances via subgraph monomorphism and labels each node with the motif it belongs to.

```python
from graph.labeling_functions import MotifLabelingFunction

labeling = MotifLabelingFunction(
    motif_order=["cycle_4", "house", "cycle_3"]
)

result = labeling.label(graph)
```

Motif matching is purely structural on plain graphs. On knowledge graphs it honours direction (an undirected motif is matched on the undirected view of a directed host, a directed motif with direction) and, when the motif carries `type` node attributes or `relation` edge attributes, entity types and relation types as well (see §12).

### motif_order

The `motif_order` list determines label priority: motifs later in the list overwrite labels from earlier ones. This matters when nodes belong to multiple motifs (e.g., a node shared between a cycle and a house).

If `motif_order` is `None`, all registered motif types are used in their default order.

### LabelingResult

Call `labeling.label(graph)` to get a `LabelingResult` with:

```python
result.node_labels   # Dict[Any, Any]  — node_id -> label ("house", "cycle_4", "unknown")
result.edge_labels   # Dict[Tuple, Any] — edge reference -> label
result.graph_labels  # Dict[str, Any]  — graph-level labels
result.details       # Dict[Any, Dict] — per-node extra info
result.metadata      # Dict[str, Any]  — contains "instances" list (motif-specific)
result.hint          # Optional PerturbationHint — zone of interest for perturbations
```

Edge references are `(u, v)` on simple graphs and `(u, v, key)` on multigraphs; build them with `utils.kg_utils.edge_key(graph, u, v, key)` so that labels, hints and metadata agree on every graph class.

### Custom Labeling Functions

Subclass `LabelingFunction` and implement the `label()` method. You decide which fields of `LabelingResult` to populate — nodes, edges, graph-level, or any combination:

```python
from interfaces import LabelingFunction
from interfaces.labeling_result import LabelingResult
from typing import Dict, Tuple, Any
import networkx as nx

class CustomLabelingFunction(LabelingFunction):
    def label(self, graph: nx.Graph) -> LabelingResult:
        # Example: only label nodes
        node_labels = {node: f"node_{node}" for node in graph.nodes()}
        return LabelingResult(node_labels=node_labels)


class EdgeOnlyLabeler(LabelingFunction):
    def label(self, graph: nx.Graph) -> LabelingResult:
        # Example: only label edges (skip node labeling)
        edge_labels = {}
        for u, v in graph.edges():
            edge_labels[(u, v)] = f"edge_{u}_{v}"
        return LabelingResult(node_labels={}, edge_labels=edge_labels)


class CompleteLabeler(LabelingFunction):
    def label(self, graph: nx.Graph) -> LabelingResult:
        # Example: label nodes, edges, and graph
        node_labels = {node: "node" for node in graph.nodes()}
        edge_labels = {(u, v): "edge" for u, v in graph.edges()}
        graph_labels = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        }
        return LabelingResult(
            node_labels=node_labels,
            edge_labels=edge_labels,
            graph_labels=graph_labels,
        )
```

### Storing Labels on the Graph

Labels are automatically stored by `GraphDatasetGenerator`, but if you need to store them manually use the shared helper (it handles simple graphs and multigraphs alike):

```python
from utils.kg_utils import apply_labeling_result

result = labeling.label(graph)
apply_labeling_result(graph, result, "expected_ground_truth")
# node labels  -> graph.nodes[n]["expected_ground_truth"] and graph.nodes[n]["label"]
# edge labels  -> graph.edges[u, v(, key)]["label"]
# graph labels -> graph.graph[name]
```

---

## 5. Perturbations

Perturbations are controlled modifications that produce variants of a graph.

### Available Perturbations

```python
from graph.perturbations import (
    RemoveNodesPerturbation,
    RemoveEdgesPerturbation,
    AddEdgesPerturbation,
    EdgePerturbation,
)
```

#### RemoveNodesPerturbation

Removes nodes using a selection strategy.

```python
perturb = RemoveNodesPerturbation(
    num_nodes=2,
    strategy="random",   # see strategies below
    params=None,         # strategy-specific parameters
)
perturbed_graph, changes = perturb.apply(graph)
# changes is reversible — it captures node attributes and incident edges:
# changes = {
#     "removed_nodes": [
#         {"id": 3, "attrs": {"motif": "house", ...}, "edges": [{"u": 3, "v": 1, "attrs": {}}, ...]},
#         {"id": 7, "attrs": {...}, "edges": [...]},
#     ]
# }
```

**Strategies:**

| Strategy       | Description                        | Params                              |
|----------------|------------------------------------|-------------------------------------|
| `random`       | Random selection                   | *(none)*                            |
| `motif`        | Nodes from the same motif instance | *(none)*                            |
| `degree`       | Highest or lowest degree nodes     | `{"mode": "high"}` or `"low"`       |
| `centrality`   | Highest betweenness centrality     | *(none)*                            |
| `role`         | Nodes matching a role attribute    | `{"role": "some_role"}`             |
| `by_attribute` | Nodes matching an attribute value  | `{"attr": "motif", "value": "house"}` |

#### RemoveEdgesPerturbation

```python
perturb = RemoveEdgesPerturbation(p_remove=0.1)  # 10% chance per edge
perturbed_graph, changes = perturb.apply(graph)
# changes = {"removed_edges": [{"u": 0, "v": 1, "attrs": {}}, {"u": 3, "v": 4, "attrs": {}}]}

# Knowledge graphs: restrict removal to some relation types
perturb = RemoveEdgesPerturbation(p_remove=0.3, relations=["works_at"])
```

#### AddEdgesPerturbation

```python
perturb = AddEdgesPerturbation(
    p_add=0.05,     # probability per non-edge
    add_num=None,    # or explicit count (overrides p_add)
)
```

#### EdgePerturbation

Combines removal and addition in one step:

```python
perturb = EdgePerturbation(p_remove=0.1, p_add=0.05)
```

#### EdgePerturbation changes schema

```python
changes = {
    "removed_edges": [{"u": u, "v": v, "attrs": {...}}, ...],
    "added_edges":   [{"u": u, "v": v}, ...],
}
```

On plain graphs added edges do not carry attributes — inverting them only requires their endpoints. On multigraphs every edge record also carries `"key"` (the edge key, so parallel edges are told apart), and on knowledge graphs added edges get a `relation` (random from the graph's vocabulary, or `add_relation=`) recorded under `"attrs"`. An edge removed and re-added by the same perturbation appears in both lists.

### Important

All perturbations return a **copy** of the graph; the original is never mutated.

### Reversibility

Every `changes` dict produced by the built-in perturbations is **reversible**: it contains enough information to rebuild the original graph from the perturbed variant. Use `graph.reconstruction.reconstruct_original`:

```python
import networkx as nx
from graph.reconstruction import reconstruct_original

perturbed = nx.read_graphml("datasets/my_dataset/graphs/graph_0.graphml")
original  = reconstruct_original(perturbed, entry["perturbation_info"]["changes"])
```

Caveat: attributes reloaded from GraphML come back as strings, so the reconstructed graph matches the string-typed view of the variant file, not the in-memory Python types from generation time.

---

## 6. Perturbation Pipeline

`PerturbationPipeline` applies perturbations and only accepts those that cause label changes. Each perturbation is applied to the **original** graph independently (not chained).

```python
from graph.perturbation_engine import PerturbationPipeline

perturbations = [
    (RemoveNodesPerturbation(num_nodes=2, strategy="motif"), 2),  # want 2 successful variants
    (EdgePerturbation(p_remove=0.1), 1),                         # want 1 successful variant
]

pipeline = PerturbationPipeline(
    perturbations=perturbations,
    labeling_functions=[labeling],
    max_iterations=10,   # max attempts per perturbation to find label-changing ones
)

results = pipeline.apply_and_check(graph)

for r in results:
    print(r["changes"])        # what was removed/added
    print(r["changed_nodes"])  # {node_id: (old_label, new_label)}
    perturbed = r["perturbed_graph"]
```

---

## 7. Dataset Generation

`GraphDatasetGenerator` orchestrates the full pipeline: generate base graphs, label them, perturb them, and save the perturbed variants to disk. Base graphs themselves are **not** written to disk — the reversible `perturbation_info` in each variant's metadata entry is enough to reconstruct them with `reconstruct_original`.

```python
from graph.dataset_generator import GraphDatasetGenerator
from graph.composite_graph_generator import MotifComposite
from graph.labeling_functions import MotifLabelingFunction
from graph.perturbations import RemoveNodesPerturbation, EdgePerturbation
from utils.rng import set_seed, reset_rng

set_seed(42)

# 1. Graph generator
graph_gen = MotifComposite(motifs=[["cycle", 4], ["house"], ["cycle", 3]])

# 2. Labeling
labeling = [MotifLabelingFunction(motif_order=["cycle_4", "house", "cycle_3"])]

# 3. Perturbations: (perturbation_instance, desired_count)
#    desired_count is an int, or an IntDistribution re-sampled per base graph (see §8)
perturbations = [
    (RemoveNodesPerturbation(num_nodes=2, strategy="motif"), 2),
    (EdgePerturbation(p_remove=0.1), 1),
]

# 4. Orchestrator
generator = GraphDatasetGenerator(
    graph_generator=graph_gen,
    labeling_functions=labeling,
    perturbations=perturbations,
    output_dir="datasets/my_dataset",
    max_perturbation_iterations=10,
)

# 5. Generate
metadata = generator.generate_dataset(
    num_graphs=100,
    num_extra_vertices=2,
    num_extra_edges=3,
    composition="sequential",
)

reset_rng()
```

### Output Structure

```
datasets/my_dataset/
  originals/
    graph_0.graphml         # base graph 0 (unperturbed)
    graph_1.graphml
    ...
  remove_nodes/             # one folder per perturbation (its folder_name)
    graph_0.graphml         # perturbed variant of base graph 0
    graph_3.graphml
  edge_perturbation/
    graph_1.graphml
    ...
  metadata.json             # one entry per variant (reversible trace)
  config.json               # seed, configuration and components of the run
```

Variant ids are global (`graph_<n>` across all perturbation folders); `metadata.json` maps each variant to its base graph. For knowledge graphs every `.graphml` is accompanied by a triples file (`graph_<n>.tsv`, or `.nt` when node ids are IRIs). Which files are written is decided by the export formats (next section): the default is `formats=["graphml", "triples"]`; pass `formats=["graphml"]` to skip the triples.

### Export Formats

Graphs are written through the exporters in `graph.exporters`. Each format is a `GraphExporter` (from `interfaces`) with a `name` and one method, `export(graph, base_path)`, that receives the destination path **without** extension, writes the file and returns its path (or `None` to skip graphs the format does not apply to). The built-in formats are registered under their names:

| Name | Files | Applies to |
|---|---|---|
| `graphml` | `graph_<n>.graphml` (multigraphs are stamped so they reload as such) | every graph |
| `triples` | `graph_<n>.tsv` (`subject<TAB>relation<TAB>object`), or `graph_<n>.nt` when node ids are IRIs | knowledge graphs only |

`GraphDatasetGenerator(formats=[...])` takes registered names and/or exporter instances (handy for parametrised exporters). The **first** format is the primary one: its file is the `graph_path` of every variant, so it must be readable by `load_graph_file` (GraphML or RDF) for reconstruction and the web browser to reload variants. Every other format that writes a file adds `<name>_path` / `original_<name>_path` to the metadata entry (`triples_path` for knowledge graphs). The names used are recorded under `generation.formats` in `config.json`, and JSON suites / web requests accept the same list as `"formats"`.

Defining a new format is a small subclass registered by name:

```python
import networkx as nx
from interfaces import GraphExporter
from graph.exporters import register_exporter

@register_exporter
class EdgeListExporter(GraphExporter):
    name = "edgelist"

    def export(self, graph, base_path):
        path = base_path + ".edgelist"
        nx.write_edgelist(graph, path, data=False)
        return path

generator = GraphDatasetGenerator(..., formats=["graphml", "edgelist"])
# every metadata entry now also carries edgelist_path / original_edgelist_path
```

Two exporters must not share a name (or an extension, or they overwrite each other's file); the name `graph` is reserved. `graph.exporters.EXPORTERS` is the registry dict, `resolve_exporters(formats)` turns a list of names/instances into exporters and `export_graph(graph, base_path, exporters)` runs them all.

### Metadata Format

`config.json` records the provenance of the run:

```json
{
    "seed": 42,
    "config": {"num_graphs": 100, "motifs": ["…"], "labeling_functions": ["…"], "perturbations": ["…"], "…": "…"},
    "components": {
        "graph_generator": {"class": "MotifComposite", "module": "graph.composite_graph_generator", "params": {"motifs": [["cycle", 4], ["house"]]}},
        "labeling_functions": [{"class": "MotifLabelingFunction", "module": "graph.labeling_functions", "params": {"motif_order": ["cycle_4", "house"]}}],
        "perturbations": [{"class": "RemoveNodesPerturbation", "module": "graph.perturbations", "params": {"num_nodes": 1, "strategy": "motif", "params": null, "folder_name": "remove_nodes"}, "folder_name": "remove_nodes", "count": 1}]
    },
    "generation": {"num_graphs": 100, "graph_kwargs": {"num_extra_vertices": 2, "num_extra_edges": 3, "composition": "sequential"}, "max_perturbation_iterations": 10, "formats": ["graphml", "triples"], "output_dir": "datasets/my_dataset"},
    "provenance": {"config_path": "…", "suite_index": 0, "suite_size": 4, "suite_seed": 42, "seed_scope": "suite"}
}
```

- `seed` is the seed passed to `GraphDatasetGenerator(seed=...)`, or else the one last given to `utils.rng.set_seed` (`None` when unseeded). It is recorded, not applied: seed the RNG yourself as in §3.
- `config` is the declarative configuration handed to `GraphDatasetGenerator(config=...)`; the scripts store the resolved JSON dataset entry and the web backend stores the request. It is `null` for datasets assembled programmatically, in which case `components` is the record to rely on.
- `components` is always derived from the live objects: class, module and constructor arguments (`rng` excluded). A component may override this by implementing `describe() -> dict`. A perturbation `count` given as a distribution is recorded as its `{"type", "params"}` dict.
- `generation` keeps the `generate_dataset(...)` call and the export `formats` used (primary first). `graph_kwargs.num_extra_vertices` / `num_extra_edges` are likewise the distribution dict when they were sampled per graph. `provenance` is free-form (`GraphDatasetGenerator(provenance=...)`); the scripts use it to note the config file and the position of the dataset in its suite.
- `generate_dataset` returns the variants list written to `metadata.json`; the `config.json` document is available as `generator.last_config`. `graph.dataset_metadata` offers `load_dataset_metadata(path)`, `load_dataset_config(path)` (both accept the dataset directory) and `find_variant(variants, graph_id)`.

`metadata.json` is a list with one entry per perturbed variant:

```json
{
    "graph_id": 0,
    "base_graph_id": 0,
    "graph_path": "datasets/my_dataset/graphs/graph_0.graphml",
    "perturbation_info": {
        "changes": {
            "removed_nodes": [
                {"id": 3, "attrs": {"motif": "house", "label": "house"}, "edges": [{"u": 3, "v": 1, "attrs": {}}]}
            ]
        },
        "changed_nodes": {"1": ["house", "unknown"]}
    },
    "graph_labels": {},
    "motif_instances": [
        {"motif_name": "cycle_4", "nodes": [0, 1, 2, 3], "edges": [[0,1], [1,2], [2,3], [0,3]]}
    ]
}
```

- `perturbation_info.changes` is the reversible diff described in §5 — feed it to `reconstruct_original` to recover the base graph.
- `perturbation_info.changed_nodes` maps node IDs to `(old_label, new_label)` pairs for nodes whose label changed as a result of the perturbation.
- `perturbation_info.changed_edges` (only when present) lists `{"u", "v"[, "key"], "old", "new"}` for edges whose label changed; `changed_graph_labels` does the same for graph-level labels. A perturbation is accepted when any of the three kinds of label changes.
- `graph_labels` and `motif_instances` describe the **base** graph (copied into every variant that derived from it), not the perturbed variant. Motif instance edges are `[u, v]` lists, or `[u, v, key]` on multigraphs.
- `graph_path` / `original_graph_path` point at the primary export format (GraphML by default). Every other format that wrote a file adds `<name>_path` / `original_<name>_path`: `triples_path` and `original_triples_path` for knowledge graphs, `edgelist_path` for a custom `edgelist` exporter, and so on.
- `graph_kind` (`{"directed", "multigraph", "kg"}`) and the triples paths appear only for directed / multigraph / knowledge-graph datasets, so the metadata of plain datasets is unchanged.

---

## 8. Dataset Generation with Random Counts

Every count that shapes a dataset can be a fixed int or an `IntDistribution` that is re-sampled for each base graph: the number of motif instances, the number of label-changing variants each perturbation must produce, and the extra vertices / edges added after composition.

Use `RandomMotifComposite` as the graph generator for datasets where each graph should have a different number of motif instances:

```python
from graph.random_motif_composite import RandomMotifComposite
from utils.distributions import IntDistribution

graph_gen = RandomMotifComposite(motif_configs=[
    (["cycle", 4], 1, None),                                              # always 1 cycle
    (["house"], 1, IntDistribution("uniform", {"min": 2, "max": 5})),     # 2-5 houses
    (["star", 3], 1, IntDistribution("poisson", {"lambda": 2})),          # ~2 stars
])

generator = GraphDatasetGenerator(
    graph_generator=graph_gen,
    labeling_functions=[MotifLabelingFunction()],
    perturbations=[
        # desired_count sampled per base graph: 1-3 node-removal variants
        (RemoveNodesPerturbation(num_nodes=1), IntDistribution("uniform", {"min": 1, "max": 3})),
        (EdgePerturbation(p_remove=0.1), 1),                               # always 1 variant
    ],
    output_dir="datasets/variable_motifs",
)

metadata = generator.generate_dataset(
    num_graphs=50,
    composition="er",
    composition_params={"p": 0.3},
    # extras sampled per graph too (clamped to >= 0); plain ints still work
    num_extra_vertices=IntDistribution("uniform", {"min": 0, "max": 4}),
    num_extra_edges=IntDistribution("poisson", {"lambda": 1}),
)
```

`MotifComposite.generate_graph` accepts the same int-or-distribution values for `num_extra_vertices` / `num_extra_edges`, and `PerturbationPipeline` samples a distribution `desired_count` on every `apply_and_check` call (the value used is reported as `desired_count` in each result). `utils.distributions.resolve_count(spec, rng, min_value)` is the helper behind both: it returns ints untouched and samples distributions.

In JSON suites and web requests the distribution lives in a sibling field that overrides the fixed one when present, exactly like `count_distribution` for motifs:

| Field | Fixed | Per-graph distribution |
|---|---|---|
| motif instances | `motifs[].count` | `motifs[].count_distribution` |
| variants per perturbation | `perturbations[].count` | `perturbations[].count_distribution` |
| extra vertices | `num_extra_vertices` | `num_extra_vertices_distribution` |
| extra edges | `num_extra_edges` | `num_extra_edges_distribution` |

```json
{
  "num_extra_vertices_distribution": {"type": "uniform", "params": {"min": 0, "max": 4}},
  "num_extra_edges_distribution": {"type": "poisson", "params": {"lambda": 1}},
  "perturbations": [
    {"type": "remove_nodes", "folder_name": "remove_nodes",
     "count_distribution": {"type": "uniform", "params": {"min": 1, "max": 3}},
     "params": {"num_nodes": 1, "strategy": "motif"}}
  ]
}
```

In the web GUI the dice button next to each count (Graph Builder extras, Perturbation Lab and Dataset Studio) switches between a fixed value and a distribution.

---

## 9. Loading Graphs from Files

`FolderGraphGenerator` loads pre-existing graph files from a folder: `.graphml` (plain, directed or multigraph) and RDF (`.rdf`, `.owl`, `.ttl`, `.nt`, `.n3`), the latter as knowledge graphs (see §12 for the RDF mapping):

```python
from graph.folder_graph_generator import FolderGraphGenerator, IterationOrder, ExhaustionPolicy

graph_gen = FolderGraphGenerator(
    folder_path="path/to/graph/files",
    iteration_order=IterationOrder.RANDOM,
    exhaustion_policy=ExhaustionPolicy.CYCLE,  # restart from beginning when exhausted
    as_undirected=False,   # True collapses everything onto simple undirected graphs (legacy)
    rdf_options=None,      # e.g. {"multigraph": False} or {"type_predicates": [...]}
)

# Use with GraphDatasetGenerator as usual
generator = GraphDatasetGenerator(
    graph_generator=graph_gen,
    labeling_functions=[MotifLabelingFunction()],
    output_dir="datasets/from_files",
)
metadata = generator.generate_dataset(num_graphs=20)
```

**Exhaustion policies:**
- `STOP`: stop generating (returns fewer graphs than requested)
- `CYCLE`: restart from the beginning
- `RAISE`: raise `GraphSourceExhausted` exception

---

## 10. Visualization

```python
from utils.visualize import visualize_graph

visualize_graph(graph, title="My Graph", filename="output/my_graph.png")
```

Nodes are colored by their `label` attribute: red (house), blue (cycle_3), green (cycle_4), gray (unknown/other).

---

## 11. Extending the Framework

### Custom Motif Generator

```python
from interfaces import MotifGenerator
import networkx as nx

class TriangleMotifGenerator(MotifGenerator):
    def generate_motif(self, start: int, id: int = 0, **kwargs) -> nx.Graph:
        g = nx.Graph()
        nodes = [start, start + 1, start + 2]
        g.add_nodes_from(nodes)
        g.add_edges_from([(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])])
        for n in nodes:
            g.nodes[n]["motif"] = "triangle"
            g.nodes[n]["motif_id"] = f"triangle_{id}"
        return g
```

Register it to make it available to `MotifComposite`:

```python
from motifs import motif_generators, motif_param_names

motif_generators["triangle"] = TriangleMotifGenerator
motif_param_names["triangle"] = []  # no extra params
```

### Custom Labeling Function

```python
from interfaces import LabelingFunction
from interfaces.labeling_result import LabelingResult

class DegreeLabelingFunction(LabelingFunction):
    def label(self, graph):
        node_labels = {}
        for node in graph.nodes():
            degree = graph.degree(node)
            node_labels[node] = "hub" if degree > 3 else "leaf" if degree == 1 else "mid"
        return LabelingResult(node_labels=node_labels)
```

`graph.degree` works on every NetworkX class, so this labeler is already graph-kind agnostic. When you need edge access, use the helpers in `utils.kg_utils` (`iter_edges`, `edge_key`, `has_edge`, `edge_attrs`) instead of `graph.edges[u, v]`, which does not exist on multigraphs.

### Custom Perturbation

```python
from interfaces import Perturbation
from utils.kg_utils import iter_edges, edge_record, remove_edge, add_edge

class SwapEdgesPerturbation(Perturbation):
    folder_name = "swap_edges"

    def __init__(self, num_swaps=1):
        self.num_swaps = num_swaps

    def apply(self, graph, hint=None):
        g = graph.copy()                     # keeps the graph class
        edges = list(iter_edges(g))          # (u, v, key, attrs) on any class
        # ... swap logic using remove_edge(g, u, v, key) / add_edge(g, u, v, **attrs) ...
        return g, {"removed_edges": [...], "added_edges": [...]}
```

Reuse the record format of §5 (`{"u", "v"[, "key"], "attrs"}`, built with `edge_record`) whenever possible so that `reconstruct_original` can invert your perturbation without extra code.

---

## 12. Knowledge Graphs

Absynthe treats a knowledge graph (KG) as a NetworkX graph, typically an `nx.MultiDiGraph`, that follows a few attribute conventions (module `utils.kg_utils`):

| Where            | Name         | Meaning                                                                 |
|------------------|--------------|-------------------------------------------------------------------------|
| node attribute   | `type`       | entity type (string)                                                    |
| node attribute   | `types`      | all entity types joined with `\|` (RDF loader, when a node has several) |
| edge attribute   | `relation`   | relation type (short name such as `foaf:knows` or `works_at`)           |
| edge attribute   | `predicate`  | full predicate IRI (RDF loader only)                                    |
| `graph.graph`    | `kg`         | `True` when the graph is a knowledge graph (enables triple export)      |
| `graph.graph`    | `multigraph` | stamped on save so a GraphML file reloads as a multigraph               |

Nothing in the plain motif pipeline sets these attributes, so existing datasets are unaffected. Direction and parallel relations are preserved everywhere: edges are referenced as `(u, v, key)` (see `edge_key`), reversible change records carry `"key"`, and `reconstruct_original` restores exact triples.

### Sources of knowledge graphs

Any `GraphGenerator` may return a KG. Two are provided:

**1. RDF / GraphML files** via `FolderGraphGenerator` (§9). An RDF file is mapped as follows:

- `(s, p, o)` with an IRI or blank-node object → directed edge `s → o` with `relation` (short predicate name) and `predicate` (full IRI). Several predicates between the same pair become parallel edges.
- `(s, rdf:type, C)` → node attribute `type` (first class in sorted order) and `types`; no edge and no class node.
- `(s, p, literal)` → node attribute of `s` keyed by the short predicate name.

```python
from graph.folder_graph_generator import load_graph_file

kg = load_graph_file("examples/rdf/sample.ttl")   # nx.MultiDiGraph, kg.graph["kg"] is True
kg.nodes["http://example.org/alice"]              # {'foaf:name': 'Alice', 'type': 'ex:Author', 'types': 'ex:Author|ex:Person'}
kg["http://example.org/alice"]["http://example.org/bob"]
# {0: {'relation': 'ex:collaboratesWith', 'predicate': ...}, 1: {'relation': 'foaf:knows', 'predicate': ...}}
```

**2. Synthetic KGs from a schema** via `SchemaKGGenerator`:

```python
from graph.schema_kg_generator import SchemaKGGenerator

graph_gen = SchemaKGGenerator(
    entity_types={"Person": 12, "Company": 4, "Paper": 8},
    relations=[
        {"name": "works_at", "domain": "Person", "range": "Company", "p": 0.25},   # Bernoulli per pair
        {"name": "wrote",    "domain": "Person", "range": "Paper",   "count": 10}, # exact number of triples
        {"name": "cites",    "domain": "Paper",  "range": "Paper",   "p": 0.15},
    ],
)
kg = graph_gen.generate_graph()   # nodes "Person_0", ..., edges with relation=...
```

### Labeling knowledge graphs

`EntityTypeLabelingFunction` labels nodes by entity type and edges by relation type. Optional `patterns` derive node labels from the relations a node takes part in, so that removing or corrupting a triple changes labels (which is what the perturbation pipeline requires to accept a perturbation):

```python
from graph.kg_labeling_functions import EntityTypeLabelingFunction

labeling = EntityTypeLabelingFunction(patterns={
    "author":   {"relation": "wrote",    "direction": "out"},                       # heads of `wrote`
    "employee": {"relation": "works_at", "direction": "out"},                       # later patterns win
    "employer": {"relation": "works_at", "direction": "in", "node_type": "Company"},
})
result = labeling.label(kg)
result.node_labels   # {"Person_0": "employee", "Company_1": "employer", "Paper_2": "Paper", ...}
result.edge_labels   # {("Person_0", "Company_1", 0): "works_at", ...}
result.hint          # nodes and edges matched by the patterns
```

`MotifLabelingFunction` also works on KGs. Undirected motifs (`house`, `cycle_5`, ...) are matched on the undirected view of the graph; a directed motif is matched with direction; a motif whose nodes carry `type` or whose edges carry `relation` matches only host nodes of that type and host edges containing those relations:

```python
import networkx as nx
from motifs.utils import assign_labels_to_motif

motif = nx.DiGraph()
motif.add_edge(0, 1, relation="works_at")
motif.nodes[0]["type"] = "Person"
instances = assign_labels_to_motif(kg, motif, "employment")   # one instance per Person -works_at-> x
```

### Perturbing knowledge graphs

All built-in perturbations work on any graph class. Two KG-specific tools are added:

```python
from graph.perturbations import RemoveEdgesPerturbation, EdgePerturbation
from graph.kg_perturbations import CorruptTriplesPerturbation

RemoveEdgesPerturbation(p_remove=0.3, relations=["works_at"])       # only these relation types
EdgePerturbation(p_remove=0.1, p_add=0.05, add_relation="knows")   # added edges get this relation

# Negative-sampling style corruption, reversible
CorruptTriplesPerturbation(num_triples=2, mode="relation")   # swap the relation type
CorruptTriplesPerturbation(num_triples=2, mode="head")       # swap the head (same entity type)
CorruptTriplesPerturbation(num_triples=2, mode="tail")       # swap the tail
CorruptTriplesPerturbation(num_triples=2, mode="any")
```

`CorruptTriplesPerturbation` never creates a triple that already exists. With `same_type=True` (the default) a relation swap prefers relations that already occur between the same pair of entity types, so the corrupted triple stays schema-consistent (`Person -works_at-> Company` becomes `Person -founded-> Company`, not `Person -cites-> Company`); the full vocabulary is used only when no such relation exists. Its change record is:

```python
changes = {
    "corrupted_triples": [
        {"u": "Person_1", "v": "Paper_1", "key": 0, "attrs": {"relation": "wrote"},
         "new_u": "Person_1", "new_v": "Paper_1", "new_key": 0, "new_attrs": {"relation": "cites"}},
    ]
}
```

### Datasets, triples and reconstruction

`GraphDatasetGenerator` needs no special configuration. With the default formats it writes, next to every `.graphml`, a triples file (`.tsv` with `subject<TAB>relation<TAB>object` lines, or N-Triples `.nt` when node ids are IRIs), and adds `graph_kind`, `triples_path` and `changed_edges` to the metadata (§7). Pass `formats=["graphml"]` to skip the triples, or add your own exporter (see "Export Formats" in §7).

```python
from graph.dataset_generator import GraphDatasetGenerator
from graph.perturbations import RemoveEdgesPerturbation, RemoveNodesPerturbation

generator = GraphDatasetGenerator(
    graph_generator=graph_gen,                 # SchemaKGGenerator or FolderGraphGenerator
    labeling_functions=[labeling],
    perturbations=[
        (RemoveEdgesPerturbation(0.3, relations=["works_at"], folder_name="remove_works_at"), 1),
        (CorruptTriplesPerturbation(2, mode="relation"), 1),
        (RemoveNodesPerturbation(1, strategy="by_attribute", params={"attr": "type", "value": "Company"}), 1),
    ],
    output_dir="datasets_kg/synthetic_kg",
)
metadata = generator.generate_dataset(num_graphs=50)
```

Reload variants with `load_graph_file` (it restores the multigraph class) and invert them with `reconstruct_original`; `normalize_changes(changes, graph)` aligns node ids read from JSON with those read from GraphML:

```python
from graph.folder_graph_generator import load_graph_file
from graph.reconstruction import normalize_changes, reconstruct_original

entry = metadata[0]
perturbed = load_graph_file(entry["graph_path"])
original = reconstruct_original(perturbed, normalize_changes(entry["perturbation_info"]["changes"], perturbed))
```

The same workflow is available from JSON configs (`kg_schema` or `folder_source` as the graph source, `entity_type_labeling`, `corrupt_triples`, `relations` on edge perturbations): see `configs/kg_example.json` and run `python scripts/gen_kg_dataset.py configs/kg_example.json` from the repository root.

### Helper API (`utils.kg_utils`)

| Function | Purpose |
|---|---|
| `iter_edges(G)` | yields `(u, v, key, attrs)` on any class, `key=None` on simple graphs |
| `edge_key(G, u, v, key)` / `unpack_ref(ref)` | canonical edge reference and its inverse |
| `has_edge`, `edge_attrs`, `remove_edge`, `add_edge` | key-aware edge access (`add_edge` returns the key used) |
| `incident_edges(G, n)` | edges touching `n` with their real endpoint order |
| `edge_record(G, u, v, key)` | reversible `{"u", "v"[, "key"], "attrs"}` record |
| `relation_vocab`, `relation_counts`, `entity_type_counts` | vocabulary and statistics |
| `graph_to_triples`, `write_triples(G, path, fmt="tsv"\|"nt")` | triple export (used by the `triples` exporter of `graph.exporters`) |
| `apply_labeling_result(G, result, attr)` | store a `LabelingResult` on the graph |
| `graph_kind(G)`, `is_kg(G)`, `stamp_graph_kind(G)` | graph-kind introspection |

---

## Complete Example

```python
from utils.rng import set_seed, reset_rng
from graph.composite_graph_generator import MotifComposite
from graph.dataset_generator import GraphDatasetGenerator
from graph.labeling_functions import MotifLabelingFunction
from graph.perturbations import RemoveNodesPerturbation, EdgePerturbation

set_seed(42)

# Build graphs with 1 cycle(4), 2 houses, and 1 star(3)
graph_gen = MotifComposite(motifs=[
    ["cycle", 4],
    ["house"],
    ["house"],
    ["star", 3],
])

# Label nodes by motif membership
labeling = [MotifLabelingFunction(motif_order=["cycle_4", "house", "star_3"])]

# Define perturbations
perturbations = [
    (RemoveNodesPerturbation(num_nodes=1, strategy="degree", params={"mode": "high"}), 2),
    (EdgePerturbation(p_remove=0.15, p_add=0.05), 1),
]

# Generate dataset
generator = GraphDatasetGenerator(
    graph_generator=graph_gen,
    labeling_functions=labeling,
    perturbations=perturbations,
    output_dir="datasets/example",
    max_perturbation_iterations=20,
)

metadata = generator.generate_dataset(
    num_graphs=50,
    num_extra_vertices=3,
    num_extra_edges=2,
    composition="er",
    composition_params={"p": 0.4},
)

print(f"Generated {len(metadata)} perturbed variants")

reset_rng()
```
