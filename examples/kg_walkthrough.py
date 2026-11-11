"""Walkthrough of the building blocks on a single knowledge graph.

Knowledge-graph counterpart of ``single_graph_walkthrough.py``. It:

1. samples a small multi-relational KG from an entity/relation schema
   (``SchemaKGGenerator``), and shows the same for an RDF file
   (``load_graph_file``);
2. labels nodes by entity type and relation patterns, and edges by relation
   type (``EntityTypeLabelingFunction``);
3. finds a relation-typed motif with the isomorphism matcher;
4. corrupts one triple through the perturbation pipeline
   (``CorruptTriplesPerturbation``) and inspects what changed;
5. inverts the perturbation with ``reconstruct_original`` and checks the
   triples are back;
6. renders the perturbed graph to ``pictures/kg_walkthrough.png``.

    python examples/kg_walkthrough.py
"""
import sys
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from graph.folder_graph_generator import load_graph_file  # noqa: E402
from graph.kg_labeling_functions import EntityTypeLabelingFunction  # noqa: E402
from graph.kg_perturbations import CorruptTriplesPerturbation  # noqa: E402
from graph.perturbation_engine import PerturbationPipeline  # noqa: E402
from graph.reconstruction import reconstruct_original  # noqa: E402
from graph.schema_kg_generator import SchemaKGGenerator  # noqa: E402
from motifs.utils import assign_labels_to_motif  # noqa: E402
from utils.kg_utils import apply_labeling_result, graph_to_triples, iter_edges  # noqa: E402
from utils.rng import set_seed  # noqa: E402
from utils.visualize import visualize_graph  # noqa: E402

set_seed(2)


def show_triples(graph, limit=8):
    triples = graph_to_triples(graph)
    for s, p, o in triples[:limit]:
        print(f"    ({s}, {p}, {o})")
    if len(triples) > limit:
        print(f"    ... {len(triples) - limit} more")


# ---------------------------------------------------------------------------
# 1. Two ways of obtaining a knowledge graph
# ---------------------------------------------------------------------------

print("== 1. Synthetic KG from a schema")
generator = SchemaKGGenerator(
    entity_types={"Person": 6, "Company": 2, "Paper": 4},
    relations=[
        {"name": "works_at", "domain": "Person", "range": "Company", "p": 0.25},
        {"name": "founded",  "domain": "Person", "range": "Company", "count": 2},
        {"name": "wrote",    "domain": "Person", "range": "Paper",   "count": 5},
        {"name": "cites",    "domain": "Paper",  "range": "Paper",   "p": 0.3},
    ],
)
kg = generator.generate_graph()
print(f"  {type(kg).__name__}: {kg.number_of_nodes()} entities, {kg.number_of_edges()} triples, "
      f"kg flag = {kg.graph.get('kg')}")
print(f"  node 'Person_0' -> {kg.nodes['Person_0']}")
show_triples(kg)

print("\n== 1b. The same from an RDF file")
rdf_kg = load_graph_file(str(REPO_ROOT / "examples" / "rdf" / "sample.ttl"))
alice, bob = "http://example.org/alice", "http://example.org/bob"
print(f"  {type(rdf_kg).__name__}: {rdf_kg.number_of_nodes()} entities, {rdf_kg.number_of_edges()} triples")
print(f"  alice -> {rdf_kg.nodes[alice]}")
print(f"  parallel relations alice -> bob: "
      f"{sorted(d['relation'] for d in rdf_kg[alice][bob].values())}")

# ---------------------------------------------------------------------------
# 2. Labeling: entity types, relation patterns and edge labels
# ---------------------------------------------------------------------------

print("\n== 2. Labeling")
labeling = EntityTypeLabelingFunction(patterns={
    "author":   {"relation": "wrote",    "direction": "out"},
    "employee": {"relation": "works_at", "direction": "out"},   # later patterns win
})
expected = labeling.label(kg)
apply_labeling_result(kg, expected, "expected_ground_truth")

for node in sorted(kg.nodes()):
    print(f"  {node:10s} type={kg.nodes[node]['type']:8s} label={expected.node_labels[node]}")
first_edge = next(iter(expected.edge_labels.items()))
print(f"  edge labels are keyed by (u, v, key): {first_edge}")
print(f"  hint covers {len(expected.hint.nodes)} nodes and {len(expected.hint.edges)} edges")

# ---------------------------------------------------------------------------
# 3. Relation-typed motif matching
# ---------------------------------------------------------------------------

print("\n== 3. Motif matching with direction, entity type and relation")
motif = nx.DiGraph()
motif.add_edge(0, 1, relation="wrote")
motif.add_edge(2, 1, relation="wrote")
motif.nodes[0]["type"] = "Person"
motif.nodes[2]["type"] = "Person"
coauthors = assign_labels_to_motif(kg.copy(), motif, "coauthorship")
print(f"  papers with two authors: {len(coauthors)} instance(s)")
for inst in coauthors:
    print(f"    nodes={sorted(inst['nodes'])}")

# ---------------------------------------------------------------------------
# 4. Perturbation through the pipeline
# ---------------------------------------------------------------------------

print("\n== 4. Corrupt one triple (relation swap) through the pipeline")
# Only works_at / founded triples are corrupted; the swap stays
# schema-consistent (both relations link Person -> Company).
pipeline = PerturbationPipeline(
    perturbations=[(CorruptTriplesPerturbation(num_triples=1, mode="relation",
                                               relations=["works_at", "founded"]), 1)],
    labeling_functions=[labeling],
    max_iterations=10,
)
results = pipeline.apply_and_check(kg)
if not results:
    print("  no perturbation changed any label")
    sys.exit(0)

result = results[0]
perturbed = result["perturbed_graph"]
apply_labeling_result(perturbed, labeling.label(perturbed), "observed_ground_truth")

for rec in result["changes"]["corrupted_triples"]:
    print(f"  ({rec['u']}, {rec['attrs']['relation']}, {rec['v']})  ->  "
          f"({rec['new_u']}, {rec['new_attrs']['relation']}, {rec['new_v']})")
print(f"  changed node labels: {result['changed_nodes']}")
print(f"  changed edge labels: {result['changed_edges']}")

# ---------------------------------------------------------------------------
# 5. Reversibility
# ---------------------------------------------------------------------------

print("\n== 5. Reconstruct the original from the perturbed graph")
restored = reconstruct_original(perturbed, result["changes"])


def triple_set(graph):
    return {(u, v, k, d.get("relation")) for u, v, k, d in iter_edges(graph)}


print(f"  triples identical to the original: {triple_set(restored) == triple_set(kg)}")

# ---------------------------------------------------------------------------
# 6. Visualise
# ---------------------------------------------------------------------------

output = REPO_ROOT / "pictures" / "kg_walkthrough.png"
visualize_graph(perturbed, title="Perturbed knowledge graph", filename=str(output))
print(f"\nSaved {output}")
