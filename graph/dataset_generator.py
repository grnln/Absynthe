import os
import warnings
from typing import Callable, List, Tuple, Dict, Any, Optional, Sequence

from interfaces import GraphGenerator, GraphExporter, LabelingFunction, Perturbation, GraphLike
from interfaces.exceptions import GraphSourceExhausted
from interfaces.labeling_result import LabelingResult
from graph.dataset_metadata import (
    CONFIG_FILENAME,
    METADATA_FILENAME,
    build_config,
    describe_components,
    save_config,
    save_metadata,
)
from graph.exporters import FormatSpec, export_graph, resolve_exporters
from graph.perturbation_engine import PerturbationPipeline
from utils.rng import get_seed
from utils.kg_utils import (
    apply_labeling_result,
    graph_kind,
    is_plain,
    unpack_ref,
)


def _safe_sorted(seq):
    """``sorted`` that falls back to string order for non-comparable items."""
    items = list(seq)
    try:
        return sorted(items)
    except TypeError:
        return sorted(items, key=lambda x: [str(t) for t in (x if isinstance(x, (list, tuple)) else [x])])


class GraphDatasetGenerator:
    """Orchestrator for generating datasets of graphs.

    Receives pluggable components:
    - A GraphGenerator instance for creating graphs
    - A list of LabelingFunction instances for computing labels
    - A list of (Perturbation, count) pairs for perturbing graphs
    - Output configuration, including the export formats

    For each original graph, the pipeline generates multiple perturbed
    variants (one per successful perturbation application). Each variant
    is saved as a separate graph in the dataset.

    The generator is agnostic to the graph class produced by the
    GraphGenerator: plain undirected graphs and directed multi-relational
    knowledge graphs (see ``utils.kg_utils``) are handled alike. Graphs are
    written through the exporters of ``graph.exporters``; by default GraphML
    plus, for knowledge graphs, a triples file next to it.

    Besides the graphs, every run writes ``metadata.json`` (one entry per
    variant with its reversible trace) and ``config.json`` (seed, the
    declarative ``config`` when given, a description of the live components
    and the generation arguments) — see ``graph.dataset_metadata``.
    """

    def __init__(
        self,
        graph_generator: GraphGenerator,
        labeling_functions: Optional[List[LabelingFunction]] = None,
        perturbations: Optional[List[Tuple[Perturbation, int]]] = None,
        output_dir: str = "output",
        max_perturbation_iterations: int = 10,
        formats: Optional[Sequence[FormatSpec]] = None,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            graph_generator: Instance of a GraphGenerator implementation.
            labeling_functions: List of LabelingFunction implementations.
                If None, no labeling is performed.
            perturbations: List of (Perturbation, desired_count) pairs.
                Each perturbation is applied to the ORIGINAL graph up to
                desired_count times, only accepting applications that
                cause label changes.
                If None, no perturbations are applied.
            output_dir: Directory where the dataset will be saved.
            max_perturbation_iterations: Maximum total attempts per
                perturbation to find label-changing applications.
            formats: Export formats, as registered names or ``GraphExporter``
                instances (see ``graph.exporters``). The first one is the
                primary format whose file becomes ``graph_path``; it must be
                readable by ``load_graph_file``. Defaults to
                ``("graphml", "triples")``: GraphML for every graph plus
                triples (``.nt`` for IRI node ids, ``.tsv`` otherwise) for
                knowledge graphs.
            seed: Seed recorded in ``config.json``. Defaults to the seed
                last passed to ``utils.rng.set_seed`` (``None`` if unseeded).
                This does not seed the RNG itself.
            config: Declarative configuration that produced the components
                (e.g. the JSON dataset entry or the web request), stored
                verbatim in ``config.json`` so the run can be replayed.
            provenance: Optional free-form provenance (config file path,
                position in a suite, ...) stored in ``config.json``.
        """
        self.graph_generator = graph_generator
        self.labeling_functions = labeling_functions or []
        self.perturbations = perturbations or []
        self.output_dir = output_dir
        self.max_perturbation_iterations = max_perturbation_iterations
        self.exporters: List[GraphExporter] = resolve_exporters(formats)
        self.seed = seed
        self.config = config
        self.provenance = provenance
        self.last_config: Optional[Dict[str, Any]] = None

        os.makedirs(self.output_dir, exist_ok=True)
        self.originals_dir = os.path.join(self.output_dir, "originals")
        os.makedirs(self.originals_dir, exist_ok=True)

        self._perturbation_dirs: Dict[str, str] = {}
        for perturbation, _ in self.perturbations:
            folder = perturbation.folder_name
            if folder in self._perturbation_dirs:
                raise ValueError(
                    f"Duplicate perturbation folder_name '{folder}': "
                    "use distinct names per perturbation instance."
                )
            pert_dir = os.path.join(self.output_dir, folder)
            os.makedirs(pert_dir, exist_ok=True)
            self._perturbation_dirs[folder] = pert_dir

    @property
    def formats(self) -> List[str]:
        """Names of the export formats, primary first."""
        return [exporter.name for exporter in self.exporters]

    def _compute_and_store_labels(self, graph: GraphLike, attribute_name: str) -> LabelingResult:
        """Compute labels using all labeling functions and store them as node/edge/graph attributes.

        Args:
            graph: The graph to label (modified in place).
            attribute_name: Node attribute name to store labels under
                            (e.g., 'expected_ground_truth', 'observed_ground_truth').

        Returns:
            The merged ``LabelingResult`` from all labeling functions.
        """
        merged = LabelingResult(node_labels={})
        for lf in self.labeling_functions:
            result = lf.label(graph)
            apply_labeling_result(graph, result, attribute_name)
            merged.node_labels.update(result.node_labels)
            merged.edge_labels.update(result.edge_labels)
            merged.graph_labels.update(result.graph_labels)
            for node, det in result.details.items():
                merged.details.setdefault(node, {}).update(det)
            merged.metadata.update(result.metadata)
        return merged

    @staticmethod
    def _serialize_changed_edges(changed_edges: Dict[Tuple, Tuple[Any, Any]]) -> List[Dict[str, Any]]:
        out = []
        for ref, (old, new) in changed_edges.items():
            u, v, key = unpack_ref(ref)
            rec: Dict[str, Any] = {"u": u, "v": v}
            if len(ref) >= 3:
                rec["key"] = key
            rec["old"] = old
            rec["new"] = new
            out.append(rec)
        return out

    def _path_entries(self, variant_files: Dict[str, str], original_files: Dict[str, str]) -> Dict[str, str]:
        """Metadata keys pointing at the files of a variant and of its original.

        The primary format gives ``graph_path`` / ``original_graph_path``;
        every other format that wrote a file gives ``<name>_path`` /
        ``original_<name>_path`` (e.g. ``triples_path``).
        """
        primary = self.exporters[0].name
        entry = {
            "graph_path": variant_files[primary],
            "original_graph_path": original_files[primary],
        }
        for name, path in variant_files.items():
            if name != primary:
                entry[f"{name}_path"] = path
                entry[f"original_{name}_path"] = original_files[name]
        return entry

    def generate_dataset(
        self,
        num_graphs: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **graph_kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate a dataset of perturbed graph variants.

        For each base graph:
        1. Generate the graph using graph_generator
        2. Compute expected labels (pre-perturbation) so the pipeline can
           detect label changes and surviving nodes inherit the labels
        3. Apply perturbations to the ORIGINAL graph, each producing an
           independent perturbed variant
        4. For each variant, compute observed labels and save it

        The base (unperturbed) graph is written to ``originals/graph_<i>``
        (plus the extension of each export format) for every base graph.
        Each perturbed variant is written to a per-perturbation subfolder
        named after the perturbation's ``folder_name``. Each variant's
        metadata entry references the original via ``original_graph_path``
        and carries a reversible ``perturbation_info.changes`` dict — see
        ``graph.reconstruction.reconstruct_original``.

        ``graph_path`` / ``original_graph_path`` point at the primary format;
        every other format that wrote a file adds ``<name>_path`` and
        ``original_<name>_path`` (``triples_path`` for knowledge graphs).
        Directed / multigraph / knowledge-graph inputs also carry
        ``graph_kind``, and edge-level label changes are listed under
        ``perturbation_info.changed_edges``.

        Args:
            num_graphs: Number of base graphs to generate.
            progress_callback: Optional ``callback(done, total)`` invoked
                after each base graph and its variants have been written,
                with ``done`` in ``1..total``. Used for progress reporting.
            **graph_kwargs: Extra keyword arguments passed to
                graph_generator.generate_graph().

        Returns:
            Metadata list with one entry per saved perturbed variant (the
            content of ``metadata.json``). The provenance document written
            to ``config.json`` is available as ``self.last_config``.
        """
        metadata = []
        graph_counter = 0

        for i in range(num_graphs):
            try:
                graph = self.graph_generator.generate_graph(**graph_kwargs)
            except GraphSourceExhausted as e:
                warnings.warn(
                    f"Graph source exhausted after {e.loaded} graphs "
                    f"(requested {e.requested}). Stopping generation early."
                )
                break

            labeling_result = None
            if self.labeling_functions:
                labeling_result = self._compute_and_store_labels(graph, 'expected_ground_truth')

            base_graph_labels = None
            base_motif_instances = None
            if labeling_result:
                if labeling_result.graph_labels:
                    base_graph_labels = dict(labeling_result.graph_labels)
                instances = labeling_result.metadata.get("instances")
                if instances:
                    base_motif_instances = [
                        {
                            "motif_name": inst["motif_name"],
                            "nodes": _safe_sorted(inst["nodes"]),
                            "edges": _safe_sorted([list(e) for e in inst["edges"]]),
                        }
                        for inst in instances
                    ]

            original_files = self.save_graph(graph, os.path.join(self.originals_dir, f"graph_{i}"))

            if self.perturbations and self.labeling_functions:
                pipeline = PerturbationPipeline(
                    perturbations=self.perturbations,
                    labeling_functions=self.labeling_functions,
                    max_iterations=self.max_perturbation_iterations,
                )
                results = pipeline.apply_and_check(graph)

                for result in results:
                    perturbed = result["perturbed_graph"]
                    self._compute_and_store_labels(perturbed, 'observed_ground_truth')

                    pert_folder = result["perturbation_folder"]
                    pert_dir = self._perturbation_dirs[pert_folder]
                    variant_files = self.save_graph(perturbed, os.path.join(pert_dir, f"graph_{graph_counter}"))
                    perturbation_info: Dict[str, Any] = {
                        "changes": result["changes"],
                        "changed_nodes": result["changed_nodes"],
                    }
                    if result.get("changed_edges"):
                        perturbation_info["changed_edges"] = self._serialize_changed_edges(
                            result["changed_edges"]
                        )
                    if result.get("changed_graph_labels"):
                        perturbation_info["changed_graph_labels"] = result["changed_graph_labels"]
                    entry: Dict[str, Any] = {
                        "graph_id": graph_counter,
                        "base_graph_id": i,
                        **self._path_entries(variant_files, original_files),
                        "perturbation_name": pert_folder,
                        "perturbation_info": perturbation_info,
                    }
                    if not is_plain(graph):
                        entry["graph_kind"] = graph_kind(graph)
                    if base_graph_labels:
                        entry["graph_labels"] = base_graph_labels
                    if base_motif_instances:
                        entry["motif_instances"] = base_motif_instances
                    metadata.append(entry)
                    graph_counter += 1

            if progress_callback is not None:
                progress_callback(i + 1, num_graphs)

        self.save_metadata(metadata, os.path.join(self.output_dir, METADATA_FILENAME))

        self.last_config = build_config(
            seed=self.seed if self.seed is not None else get_seed(),
            config=self.config,
            components=describe_components(
                self.graph_generator, self.labeling_functions, self.perturbations
            ),
            generation={
                "num_graphs": num_graphs,
                "graph_kwargs": graph_kwargs,
                "max_perturbation_iterations": self.max_perturbation_iterations,
                "formats": self.formats,
                "output_dir": self.output_dir,
            },
            provenance=self.provenance,
        )
        save_config(self.last_config, os.path.join(self.output_dir, CONFIG_FILENAME))
        return metadata

    def save_graph(self, graph: GraphLike, base_path: str) -> Dict[str, str]:
        """Write ``graph`` in every export format next to ``base_path``.

        ``base_path`` has no extension; each exporter appends its own.
        Returns ``{format_name: written_path}`` for the formats that wrote a
        file (e.g. ``{"graphml": ".../graph_0.graphml", "triples": ".../graph_0.tsv"}``).
        """
        return export_graph(graph, base_path, self.exporters)

    def save_metadata(self, metadata, path):
        """Save the variants list to a JSON file."""
        save_metadata(metadata, path)
