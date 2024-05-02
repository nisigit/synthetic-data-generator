import numpy as np
import networkx as nx


def generate_degree_sequence(degree_histogram):
    """
    Generate a degree sequence based on the degree histogram.

    Args:
        degree_histogram (list): The degree histogram.

    Returns:
        list: The generated degree sequence.
    """
    degree_sequence = []
    for degree, count in enumerate(degree_histogram):
        degree_sequence.extend([degree] * int(count))

    # Ensure the sum of degrees is even
    if sum(degree_sequence) % 2 != 0:
        index = np.random.randint(len(degree_sequence))
        degree_sequence[index] += 1

    return degree_sequence


def generate_configuration_model(degree_sequence):
    """
    Generate a graph using the configuration model based on the degree sequence.

    Args:
        degree_sequence (list): The degree sequence.

    Returns:
        nx.Graph: The generated graph.
    """
    graph = nx.configuration_model(degree_sequence)
    graph = nx.Graph(graph)  # Convert the multigraph to a simple graph
    return graph
