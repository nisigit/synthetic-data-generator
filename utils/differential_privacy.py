import networkx as nx
import numpy as np
from scipy.stats import laplace, cauchy


def stable_edge_order(G: nx.Graph) -> list:
    """
    Return a stable order of edges in the graph G.
    :param G: The input graph.
    :return: A list of edges sorted by the lexicographic order of their endpoints.
    """
    return sorted(G.edges, key=lambda x: (x[0], x[1]))  # type: ignore


def pi_graph_projection(G: nx.graph, theta: int, edge_order: list):
    """
    Project the graph G onto the theta-PI graph.
    :param G: The input graph.
    :param theta: The degree threshold.
    :param edge_order: The order of edges to be added to the projected graph.
    :return:
    """
    G_theta = nx.Graph()
    G_theta.add_nodes_from(G.nodes)
    for edge in edge_order:
        u, v = edge
        if G_theta.degree(u) < theta and G_theta.degree(v) < theta:
            G_theta.add_edge(u, v)

    return G_theta


def quality_function(G: nx.graph, theta: int, epsilon_2: float, Theta=200):
    """
    Calculate the quality score for a given theta value.

    Args:
        G (nx.Graph): The input graph.
        theta (int): The theta value.
        epsilon_2 (float): The privacy budget for the cumulative histogram.
        Theta (int): The maximum degree threshold for the projected graph.

    Returns:
        float: The quality score.
    """
    G_theta = pi_graph_projection(G, theta, stable_edge_order(G))
    projection_loss = 2 * len([v for v in G_theta.nodes() if G_theta.degree(v) > Theta])
    noise_magnitude = np.sqrt(theta * (theta + 1) / epsilon_2)
    quality_score = -(projection_loss + noise_magnitude)

    return quality_score


def exponential_mechanism(G, theta_candidates, epsilon_1, epsilon_2, Theta=200):
    """
    Select the optimal theta value using the exponential mechanism.

    Args:
        G (nx.Graph): The input graph.
        theta_candidates (list): The list of candidate theta values.
        epsilon_1 (float): The privacy budget for the exponential mechanism.
        epsilon_2 (float): The privacy budget for the cumulative histogram.
        Theta (int): The maximum degree threshold for the projected graph.

    Returns:
        int: The selected optimal theta value.
    """
    # Calculate the sensitivity of the quality function
    sensitivity = 2 * Theta + 2

    # Calculate the quality scores for each candidate theta value
    quality_scores = [quality_function(G, theta, epsilon_2, Theta) for theta in theta_candidates]

    # Calculate the probabilities based on the quality scores
    probabilities = np.exp(epsilon_1 * np.array(quality_scores) / (2 * sensitivity))
    probabilities /= np.sum(probabilities)

    # Select the optimal theta value based on the probabilities
    optimal_theta = np.random.choice(theta_candidates, p=probabilities)

    return optimal_theta


def get_cumulative_histogram(G_theta: nx.Graph, theta: int):
    """
    Compute the cumulative histogram of the theta projected graph.

    Args:
        G_theta (nx.Graph): The projected graph.
        theta (int): The degree bound.

    Returns:
        np.ndarray: The cumulative histogram.
    """
    degree_histogram = np.zeros(theta + 1)
    for v in G_theta.nodes():
        degree = G_theta.degree(v)
        if degree <= theta:
            degree_histogram[degree] += 1

    return np.cumsum(degree_histogram)


def add_laplace_noise(cumulative_histogram: np.ndarray, epsilon_2: float):
    """
    Add Laplace noise to the cumulative histogram.

    Args:
        cumulative_histogram (np.ndarray): The cumulative histogram.
        epsilon_2 (float): The privacy budget for the cumulative histogram.

    Returns:
        np.ndarray: The noisy cumulative histogram.
    """
    sensitivity = len(cumulative_histogram) - 1
    scale = sensitivity / epsilon_2
    noise = laplace.rvs(loc=0, scale=scale, size=len(cumulative_histogram))
    noisy_histogram = cumulative_histogram + noise
    noisy_histogram = np.maximum(noisy_histogram, 0)

    return noisy_histogram


def get_reconstructed_histogram(noisy_histogram: np.ndarray):
    """
    Reconstruct the noisy cumulative histogram.
    This step exploits the monotonicity property of the cumulative histogram to reduce the impact of noise.
    Iterate over the noisy cumulative histogram from left to right.
    If the current entry is smaller than the previous entry, find the first non-decreasing entry to the right and distribute the total count uniformly among the entries in between.
    Reconstruct the degree histogram by taking the differences between consecutive entries in the calibrated cumulative histogram

    Args:
        noisy_histogram (np.ndarray): The noisy cumulative histogram.

    Returns:
        np.ndarray: The reconstructed degree histogram.
    """
    n = len(noisy_histogram)
    calibrated_histogram = np.zeros(n)
    for i in range(1, n):
        if noisy_histogram[i] < noisy_histogram[i - 1]:
            j = i + 1
            while j < n and noisy_histogram[j] < noisy_histogram[j - 1]:
                j += 1
            count = noisy_histogram[i - 1] - noisy_histogram[j - 1]
            for k in range(i, j):
                calibrated_histogram[k] = noisy_histogram[k] + count / (j - i)

    degree_histogram = np.diff(calibrated_histogram)

    return degree_histogram


def calibrate_histogram(noisy_cumulative_histogram):
    """
    Calibrate the noisy cumulative histogram to ensure monotonicity.

    Args:
        noisy_cumulative_histogram (np.ndarray): The noisy cumulative histogram.

    Returns:
        np.ndarray: The calibrated cumulative histogram.
    """
    n = len(noisy_cumulative_histogram)
    calibrated_histogram = noisy_cumulative_histogram.copy()

    for i in range(1, n):
        if calibrated_histogram[i] < calibrated_histogram[i - 1]:
            j = i + 1
            while j < n and calibrated_histogram[j] < calibrated_histogram[i - 1]:
                j += 1

            if j == n:
                calibrated_histogram[i:] = calibrated_histogram[i - 1]
            else:
                total_count = calibrated_histogram[j] - calibrated_histogram[i - 1]
                num_bins = j - i + 1
                uniform_count = total_count // num_bins
                remainder = total_count % num_bins

                calibrated_histogram[i:j + 1] = calibrated_histogram[i - 1] + uniform_count
                calibrated_histogram[i:i + int(remainder)] += 1

    return calibrated_histogram

def reconstruct_histogram(calibrated_cumulative_histogram: np.ndarray):
    """
    Reconstruct the degree histogram from the calibrated cumulative histogram.

    Args:
        calibrated_cumulative_histogram (np.ndarray): The calibrated cumulative histogram.

    Returns:
        np.ndarray: The reconstructed degree histogram.
    """
    degree_histogram = np.diff(calibrated_cumulative_histogram)

    return degree_histogram

# Kasiviswanathan et al. 2011
def node_dp_degree_approximation(epsilon: float, theta: int, degree_dict: dict, deg_dist: np.ndarray):
    beta = epsilon / (np.sqrt(2) * (theta + 1))
    St_naive = 0  # smooth upper bound on local sensitivity of Naive Truncation
    for k in range(1, theta + 1):
        n = len([d for d in degree_dict.keys() if theta - k <= d <= theta + k])
        curr_s = (np.exp(-beta * k)) * (1 + k + n)
        St_naive = max(St_naive, curr_s)

    cauchy_scale = np.sqrt(2) * theta * St_naive / epsilon
    print('cauchy scale', cauchy_scale)
    cauchy_noise = cauchy.rvs(loc=0, scale=cauchy_scale, size=theta + 1)
    noisy_degs = deg_dist[:theta + 1] + cauchy_noise
    noisy_degs = np.maximum(noisy_degs, 0)
    noisy_degs = noisy_degs / sum(noisy_degs)

    return noisy_degs
