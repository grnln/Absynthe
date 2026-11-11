import networkx as nx
from interfaces import MotifGenerator
from motifs.utils import assign_labels_to_motif


class GateMotifGenerator(MotifGenerator):
    r"""Generator for gate motifs.

    A gate motif consists of an entry node, two parallel arms (paths) of equal
    `arm_length`, and an exit node where both arms reconnect. The structure is:

        entry -- a1 -- a2 -- ... -- a_k -- exit
              \
               \-- b1 -- b2 -- ... -- b_k --/

    Total nodes = 2*arm_length + 2 (entry + exit).
    """

    def generate_motif(self, start: int, arm_length: int, id: int = 0) -> nx.Graph:
        """Generate a gate motif starting at `start`.

        Args:
            start (int): Starting index for the nodes (entry node is `start`).
            arm_length (int): Number of intermediate nodes on each arm (>= 1).
            id (int): Identifier for the motif.

        Returns:
            nx.Graph: The gate motif graph.

        Raises:
            ValueError: If `arm_length` < 1.
        """
        if arm_length < 1:
            raise ValueError("arm_length must be at least 1 for a gate motif.")

        graph = nx.Graph()

        entry = start
        # left arm nodes: start+1 .. start+arm_length
        left_nodes = list(range(start + 1, start + 1 + arm_length))
        # right arm nodes: start+1+arm_length .. start+2*arm_length
        right_nodes = list(range(start + 1 + arm_length, start + 1 + 2 * arm_length))
        exit_node = start + 1 + 2 * arm_length

        # Add all nodes
        graph.add_node(entry)
        graph.add_nodes_from(left_nodes)
        graph.add_nodes_from(right_nodes)
        graph.add_node(exit_node)

        # Connect left arm: entry - left1 - left2 - ... - left_k - exit
        prev = entry
        for n in left_nodes:
            graph.add_edge(prev, n)
            prev = n
        graph.add_edge(prev, exit_node)

        # Connect right arm: entry - right1 - ... - right_k - exit
        prev = entry
        for n in right_nodes:
            graph.add_edge(prev, n)
            prev = n
        graph.add_edge(prev, exit_node)

        # Add motif attributes
        motif_name = f'gate_{arm_length}'
        motif_id = f'gate_{arm_length}_{id}'
        for node in graph.nodes():
            graph.nodes[node]['motif'] = motif_name
            graph.nodes[node]['motif_id'] = motif_id

        return graph

    @staticmethod
    def assign_labels(graph, arm_length=1):
        """
        Search for subgraphs shaped like a "gate" with arms of `arm_length` nodes and label their nodes.
        """
        motif = GateMotifGenerator().generate_motif(0, arm_length)
        return assign_labels_to_motif(graph, motif, f'gate_{arm_length}')
