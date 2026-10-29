import networkx as nx

from typing import Any
from graph.composite_graph_generator import MotifComposite

from graph.perturbations import RemoveEdgesPerturbation
from chain.perturbation_chain_pipelines import  MCTSPerturbationChainPipeline
from interfaces import LabelingFunction, LabelingResult, PerturbationWalk, Perturbation
from chain.perturbation_walks import NaturalOrderEdgePerturbationWalk, RandomEdgePerturbationWalk

class TestLabelingFunction(LabelingFunction):
    def __init__(self):
        pass
    
    def label(self, graph: nx.Graph[Any]) -> LabelingResult:
        work_graph = graph.copy()

        # Clear any pre-existing 'label' attributes so that nodes which
        # no longer match a motif are correctly labeled 'unknown' instead
        # of retaining a stale label from a previous labeling pass.
        for node in work_graph.nodes():
            work_graph.nodes[node].pop('label', None)

        node_labels: dict[int, Any] = {}
        all_instances: list[dict[str, Any]] = []
        details: dict[int, dict[str, Any]] = {}

        for node in work_graph.nodes():
            work_graph.nodes[node]['label'] = 'A' if work_graph.degree[node] > 0 else 'B'
            node_labels[node] = work_graph.nodes[node].get('label', 'unknown')

        return LabelingResult(
            node_labels=node_labels,
            graph_labels={},
            details=details,
            metadata={"instances": all_instances} if all_instances else {},
        )

def test_square() -> None:
    gen = MotifComposite(motifs = [["cycle", 4]])
    graph = gen.generate_graph()
    labeling = TestLabelingFunction()

    perturbations: list[tuple[Perturbation, PerturbationWalk[Any]]] = [(RemoveEdgesPerturbation(p_remove = 0.5), NaturalOrderEdgePerturbationWalk())]
    pipeline = MCTSPerturbationChainPipeline(
        graph,
        perturbations,
        [labeling],
        {perturbation: RandomEdgePerturbationWalk() for perturbation, _ in perturbations}
    )
    results = pipeline.find()

    print('------------------------')
    print('4-node square graph test')
    print('------------------------')
    print(f'Found {len(results)} chains:\n{results}\n')

def test_k5() -> None:
    graph = nx.complete_graph(5)
    labeling = TestLabelingFunction()

    perturbations: list[tuple[Perturbation, PerturbationWalk[Any]]] = [(RemoveEdgesPerturbation(0.5), NaturalOrderEdgePerturbationWalk())]
    pipeline = MCTSPerturbationChainPipeline(
        graph,
        perturbations,
        [labeling],
        {perturbation: RandomEdgePerturbationWalk() for perturbation, _ in perturbations}
    )
    results = pipeline.find()

    print('--------------')
    print('K-5 graph test')
    print('--------------')
    print(f'Found {len(results)} chains:\n{results}\n')

if __name__ == '__main__':
    test_square()
    test_k5()