import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


def visualize_graph(graph: nx.Graph, title: str = "Graph", filename: str = "pictures/graph.png") -> None:
    """Visualize the given NetworkX graph and save to PNG, coloring nodes by label.

    Args:
        graph (nx.Graph): The graph to visualize.
        title (str): Title for the plot.
        filename (str): Filename to save the PNG image (default: pictures/graph.png).
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)  # Layout for positioning nodes
    
    # Color nodes based on 'label' attribute
    node_colors = []
    for node in graph.nodes():
        label = graph.nodes[node].get('label', -1)
        if label == 'house':
            node_colors.append('red')  # House
        elif label == 'cycle_3':
            node_colors.append('blue')  # Cycle_3
        elif label == 'cycle_4':
            node_colors.append('green')  # Cycle_4
        else:
            node_colors.append('gray')  # Unknown
    
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='red', label='House (0)'),
        mpatches.Patch(color='blue', label='Cycle_3 (1)'),
        mpatches.Patch(color='green', label='Cycle_4 (2)'),
        mpatches.Patch(color='gray', label='Unknown')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig(filename)
    plt.close()  # Close to free memory