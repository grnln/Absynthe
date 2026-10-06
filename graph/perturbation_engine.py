from typing import List, Tuple, Dict, Any

import networkx as nx

from interfaces import Perturbation, LabelingFunction, PerturbationHint
from interfaces.labeling_result import LabelingResult


class PerturbationPipeline:
    """Applies perturbations to a graph, accepting only those that cause label changes.

    Receives a list of (Perturbation, desired_count) pairs and a list of
    LabelingFunction instances. For each pair, the perturbation is applied
    repeatedly to the ORIGINAL graph; each application is checked against
    the labeling functions and only accepted if it causes at least one label
    change. The process continues until the desired count of successful
    (label-changing) applications is reached, or max_iterations total
    attempts are exhausted.

    Each successful application produces an independent perturbed graph
    derived from the same original.
    """

    def __init__(
        self,
        perturbations: List[Tuple[Perturbation, int]],
        labeling_functions: List[LabelingFunction],
        max_iterations: int = 10,
        chain_length: int = 4
    ):
        """
        Args:
            perturbations: List of (perturbation_instance, desired_count) pairs.
                For each pair, the pipeline will try to apply the perturbation
                desired_count times successfully (i.e., causing label changes).
            labeling_functions: List of labeling functions. Labels from
                later functions overwrite labels from earlier ones for
                the same node.
            max_iterations: Maximum total attempts per perturbation to find
                applications that cause label changes.
        """
        self.perturbations = perturbations
        self.labeling_functions = labeling_functions
        self.max_iterations = max_iterations
        self.chain_length = chain_length

    def _compute_labels(self, graph: nx.Graph) -> LabelingResult:
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
            derived = self._derive_hint_from_details(merged.details)
            merged.hint = derived if not derived.is_empty() else None
        return merged

    @staticmethod
    def _derive_hint_from_details(
        details: Dict[int, Dict[str, Any]]
    ) -> PerturbationHint:
        """Derive a PerturbationHint from per-node ``details``.

        Looks for ``motif_nodes`` and ``motif_edges`` entries and
        accumulates them, normalizing edges to ``(min, max)``.
        """
        hint = PerturbationHint()
        for det in details.values():
            motif_nodes = det.get("motif_nodes")
            if motif_nodes:
                hint.nodes.update(motif_nodes)
            motif_edges = det.get("motif_edges")
            if motif_edges:
                for edge in motif_edges:
                    u, v = edge[0], edge[1]
                    hint.edges.add(PerturbationHint.normalize_edge(u, v))
        return hint

    def apply_single_perturbation(self, graph: nx.Graph, perturbation, desired_count, original_result):
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
            candidate_labels = candidate_result.node_labels
            changed_nodes = {
                node: (original_result.node_labels.get(node), candidate_labels[node])
                for node in candidate_labels
                if candidate_labels[node] != original_result.node_labels.get(node)
            }

            if changed_nodes:
                successes += 1
                return {
                    "perturbed_graph": candidate_graph,
                    "changes": changes,
                    "changed_nodes": changed_nodes,
                    "labeling_result": candidate_result,
                    "original_labeling_result": original_result,
                    "perturbation_folder": perturbation.folder_name,
                }
        return None

    def test_chain_perturbation(self, perturbation, graph: nx.Graph, original_result, desired_result = False):
        attempts = 0
        result = False

        candidate_graph = graph
        candidate_result = original_result

        changes = None
        changed_nodes = None

        while result is False and attempts < self.max_iterations:
            attempts += 1

            # Always apply to the original graph
            candidate_graph, changes = perturbation.apply(
                graph, hint = original_result.hint
            )

            # Check for label changes
            candidate_result = self._compute_labels(candidate_graph)
            candidate_labels = candidate_result.node_labels
            changed_nodes = {
                node: (original_result.node_labels.get(node), candidate_labels[node])
                for node in candidate_labels if candidate_labels[node] != original_result.node_labels.get(node)
            }

            if changed_nodes:
                result = desired_result
            else:
                result = not desired_result

        return (result, candidate_graph, candidate_result, changes)
    
    def test_subchain(self, chain, subchain, current_length, subchain_length, graph: nx.Graph, original_result):
        out = True
        
        if current_length >= subchain_length:
            acc_graph = graph
            acc_result = original_result

            result = None

            for perturbation, _, _ in subchain:
                result, acc_graph, acc_result, _ = self.test_chain_perturbation(perturbation, acc_graph, acc_result)
    
            out = result is True
        else:
            for perturbation in [p for p in chain if p not in subchain]:
                out = self.test_subchain(chain, subchain + [perturbation], len(subchain) + 1, subchain_length, graph, original_result)
                
                if out is False:
                    break;
        return out

    def find_chain(self, chain, current_length, chain_length, graph: nx.Graph, original_result, chains):
        if current_length >= chain_length:
            acc_graph = graph
            acc_result = original_result

            result = None

            for perturbation, _, _ in chain:
                result, acc_graph, acc_result, _ = self.test_chain_perturbation(perturbation, acc_graph, acc_result)
    
            if result is True:
                chains.append(chain)
        else:
            for perturbation, _ in self.perturbations:
                discard = False
                result, candidate_graph, _, changes = self.test_chain_perturbation(perturbation, graph, original_result)

                if result is False:
                    continue

                candidate_chain = chain + [(perturbation, candidate_graph, changes)]

                # Discard the perturbation if one of the possible sub-chains involving it changes labels
                for subchain_length in range(2, min(len(candidate_chain), chain_length)):
                    for perturb in candidate_chain:
                        if self.test_subchain(candidate_chain, [perturb], 1, subchain_length, graph, original_result) is False:
                            discard = True

                if discard is True:
                    continue;

                self.find_chain(chain + [(perturbation, candidate_graph, changes)], len(chain) + 1, chain_length, graph, original_result, chains)

    def find_chains(self, chain_length, graph: nx.Graph, original_result):
        chains = []

        for perturbation, _ in self.perturbations:
            result, candidate_graph, _, changes = self.test_chain_perturbation(perturbation, graph, original_result)

            if result is False:
                continue

            self.find_chain([(perturbation, candidate_graph, changes)], 1, chain_length, graph, original_result, chains)
        return chains
        
    def apply_and_check(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Apply perturbations to the original graph, only accepting those that change labels.

        For each (perturbation, desired_count) pair:
          - Apply the perturbation to the ORIGINAL graph (not accumulated)
          - Check if labels changed compared to the original
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
            - 'labeling_result': full LabelingResult for the perturbed graph
            - 'original_labeling_result': full LabelingResult for the original graph
            - 'perturbation_folder': folder_name of the perturbation that produced this variant
        """
        original_result = self._compute_labels(graph)
        results = []
        chain_results = []

        for perturbation, desired_count in self.perturbations:
            result = self.apply_single_perturbation(graph, perturbation, desired_count, original_result)

            if result is not None:
                results.append(result)

        # Search for chains of multiple perturbations
        for chain_length in range(2, self.chain_length):
            chain_results.extend(self.find_chains(chain_length, graph, original_result))

        if len(chain_results) > 0:
            print('')
            print('--------------------------')
            print('Perturbation chain results')
            print('--------------------------')
            print(chain_results)
            print('')
        return results
