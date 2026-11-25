import networkx as nx

from typing import Any
from graph.composite_graph_generator import MotifComposite

from graph.perturbations import RemoveEdgesPerturbation, RemoveNodesPerturbation
from chain.perturbation_chains import BasePerturbationChain
from chain.perturbation_chain_pipelines import  HeuristicPerturbationChainPipeline, MCTSPerturbationChainPipeline
from interfaces import GraphLike, LabelingFunction, LabelingResult, PerturbationHeuristic, Perturbation
from chain.perturbation_heuristics import NaturalOrderNodePerturbationHeuristic, RandomEdgePerturbationHeuristic

class TestLabelingFunction(LabelingFunction):
    def __init__(self):
        pass
    
    def label(self, graph: GraphLike) -> LabelingResult:
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

    perturbations: list[tuple[Perturbation, PerturbationHeuristic[Any]]] = [(RemoveNodesPerturbation(num_nodes = 1), NaturalOrderNodePerturbationHeuristic(graph))]
    pipeline = HeuristicPerturbationChainPipeline(
        graph,
        perturbations,
        [labeling],
        BasePerturbationChain,
        max_iterations = 100
    )
    results = pipeline.find()

    print('------------------------')
    print('4-node square graph test')
    print('------------------------')
    print(f'#Iterations = {pipeline.num_iterations} | Running time = {pipeline.running_time}')
    print(f'Found {len(results)} chains:\n{results}\n')

def test_k5() -> None:
    graph = nx.complete_graph(5)
    labeling = TestLabelingFunction()

    perturbations: list[tuple[Perturbation, PerturbationHeuristic[Any]]] = [(RemoveEdgesPerturbation(0.5), RandomEdgePerturbationHeuristic(graph))]
    pipeline = MCTSPerturbationChainPipeline(
        graph,
        perturbations,
        [labeling],
        BasePerturbationChain,
        max_iterations = 10,
        num_chains = 4
    )
    results = pipeline.find()

    print('--------------')
    print('K-5 graph test')
    print('--------------')
    print(f'#Iterations = {pipeline.num_iterations} | Running time = {pipeline.running_time}')
    print(f'Found {len(results)} chains:\n{results}\n')

if __name__ == '__main__':
    test_square()
    test_k5()