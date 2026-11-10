from typing import List, Tuple, Dict, Any, Optional

from interfaces import Perturbation, LabelingFunction, PerturbationHint, GraphLike
from interfaces.labeling_result import LabelingResult
from utils.distributions import CountSpec, resolve_count
from utils.kg_utils import unpack_ref
from utils.rng import get_rng


class PerturbationPipeline:
    """Applies perturbations to a graph, accepting only those that cause label changes.

    Receives a list of (Perturbation, desired_count) pairs and a list of
    LabelingFunction instances. For each pair, the perturbation is applied
    repeatedly to the ORIGINAL graph; each application is checked against
    the labeling functions and only accepted if it causes at least one label
    change (node, edge or graph level). The process continues until the
    desired count of successful (label-changing) applications is reached,
    or max_iterations total attempts are exhausted.

    ``desired_count`` may be a fixed int or an ``IntDistribution``; in the
    latter case it is re-sampled on every call to ``apply_and_check`` (i.e.
    per base graph), mirroring the random motif counts of ``MotifComposite``.

    Each successful application produces an independent perturbed graph
    derived from the same original.
    """

    def __init__(
        self,
        perturbations: List[Tuple[Perturbation, CountSpec]],
        labeling_functions: List[LabelingFunction],
        max_iterations: int = 10,
        rng=None,
    ):
        """
        Args:
            perturbations: List of (perturbation_instance, desired_count) pairs.
                For each pair, the pipeline will try to apply the perturbation
                desired_count times successfully (i.e., causing label changes).
                desired_count is an int or an ``IntDistribution`` sampled per
                call (clamped to ``>= 1``).
            labeling_functions: List of labeling functions. Labels from
                later functions overwrite labels from earlier ones for
                the same node.
            max_iterations: Maximum total attempts per perturbation to find
                applications that cause label changes.
            rng: Random source used to sample distribution counts. Defaults
                to the global RNG of ``utils.rng``.
        """
        self.perturbations = perturbations
        self.labeling_functions = labeling_functions
        self.max_iterations = max_iterations
        self.rng = rng

    def _compute_labels(self, graph: GraphLike) -> LabelingResult:
        """Compute labels using all labeling functions.

        Later labeling functions overwrite earlier ones for the same node.
        Edge labels, graph labels and metadata are merged across all functions. The
        perturbation hint is merged across functions; if no labeling
        function provides one, it is derived from ``details`` so that
        legacy labelers (which only fill ``motif_nodes`` / ``motif_edges``
        per node) still produce a usable hint for downstream perturbations.
        """
        merged = LabelingResult(node_labels={})
        for lf in self.labeling_functions:
            result = lf.label(graph)
            merged.node_labels.update(result.node_labels)
            merged.edge_labels.update(result.edge_labels)
            merged.graph_labels.update(result.graph_labels)
            for node, det in result.details.items():
                merged.details.setdefault(node, {}).update(det)
            merged.metadata.update(result.metadata)
            if result.hint is not None and not result.hint.is_empty():
                merged.hint = (
                    result.hint if merged.hint is None
                    else merged.hint.merge(result.hint)
                )

        if merged.hint is None or merged.hint.is_empty():
            derived = self._derive_hint_from_details(merged.details, graph)
            merged.hint = derived if not derived.is_empty() else None
        return merged

    @staticmethod
    def _derive_hint_from_details(
        details: Dict[Any, Dict[str, Any]],
        graph: Optional[GraphLike] = None,
    ) -> PerturbationHint:
        """Derive a PerturbationHint from per-node ``details``.

        Looks for ``motif_nodes`` and ``motif_edges`` entries and
        accumulates them. Edge references may be ``(u, v)`` or
        ``(u, v, key)``; endpoints are kept in order on directed graphs
        and normalized to ``(min, max)`` otherwise.
        """
        directed = bool(graph is not None and graph.is_directed())
        hint = PerturbationHint()
        for det in details.values():
            motif_nodes = det.get("motif_nodes")
            if motif_nodes:
                hint.nodes.update(motif_nodes)
            motif_edges = det.get("motif_edges")
            if motif_edges:
                for edge in motif_edges:
                    u, v, key = unpack_ref(edge)
                    hint.edges.add(
                        PerturbationHint.normalize_edge(u, v, key, directed=directed)
                    )
        return hint

    @staticmethod
    def _diff(original: Dict[Any, Any], candidate: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        return {
            k: (original.get(k), candidate[k])
            for k in candidate
            if candidate[k] != original.get(k)
        }

    def apply_and_check(self, graph: GraphLike) -> List[Dict[str, Any]]:
        """Apply perturbations to the original graph, only accepting those that change labels.

        For each (perturbation, desired_count) pair:
          - Apply the perturbation to the ORIGINAL graph (not accumulated)
          - Check if labels changed compared to the original (node labels,
            edge labels or graph-level labels)
          - If labels changed: accept (store the perturbed graph)
          - If labels didn't change: discard, try again
          - Stop when desired_count successful applications are reached
            or max_iterations total attempts are exhausted

        Args:
            graph: The original graph. All perturbations are applied to this graph.

        Returns:
            A list of result dicts, one per successful perturbation application.
            Each dict contains:
            - 'perturbed_graph': the independently perturbed graph
            - 'changes': what the perturbation changed (from Perturbation.apply)
            - 'changed_nodes': {node: (old_label, new_label)} for nodes whose label changed
            - 'changed_edges': {edge_ref: (old_label, new_label)} for edges whose label changed
            - 'changed_graph_labels': {name: (old, new)} for graph-level labels that changed
            - 'labeling_result': full LabelingResult for the perturbed graph
            - 'original_labeling_result': full LabelingResult for the original graph
            - 'perturbation_folder': folder_name of the perturbation that produced this variant
            - 'desired_count': the count targeted for this call (the sampled
              value when the pair carries a distribution)
        """
        original_result = self._compute_labels(graph)
        results = []
        rng = self.rng or get_rng()

        for perturbation, count_spec in self.perturbations:
            desired_count = resolve_count(count_spec, rng, min_value=1)
            successes = 0
            attempts = 0

            while successes < desired_count and attempts < self.max_iterations:
                attempts += 1

                # Always apply to the original graph
                candidate_graph, changes = perturbation.apply(
                    graph, hint=original_result.hint
                )

                # Check for label changes
                candidate_result = self._compute_labels(candidate_graph)
                changed_nodes = self._diff(original_result.node_labels, candidate_result.node_labels)
                changed_edges = self._diff(original_result.edge_labels, candidate_result.edge_labels)
                changed_graph = self._diff(original_result.graph_labels, candidate_result.graph_labels)

                if changed_nodes or changed_edges or changed_graph:
                    successes += 1
                    results.append({
                        "perturbed_graph": candidate_graph,
                        "changes": changes,
                        "changed_nodes": changed_nodes,
                        "changed_edges": changed_edges,
                        "changed_graph_labels": changed_graph,
                        "labeling_result": candidate_result,
                        "original_labeling_result": original_result,
                        "perturbation_folder": perturbation.folder_name,
                        "desired_count": desired_count,
                    })

        return results
