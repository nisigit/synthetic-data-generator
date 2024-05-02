# Email Data Generator

This project is a Python based implementation of a synthetic data generator for email log datasets. The generator is capable of generating email log datasets with a specified number of users, emails, and time periods. The generated datasets can be used for various data analysis tasks, such as network analysis, anomaly detection, and differential privacy. 

## File Descriptions

### `generation.ipynb`

This Jupyter notebook contains the code for generating synthetic graph data based on the differential privacy algorithms implemented in `DifferentialPrivacy.py`. Here's a brief overview of the key sections in this notebook:

- **Data Loading and Preprocessing**: This section loads the email data from a CSV file, processes the recipient lists, maps email addresses to unique IDs, and extracts the day and hour of the week for each email.

- **Email Activity Analysis**: This section calculates the number of reply, forward, and starter emails, and plots the number of emails sent by hour of the week. It also calculates the degree distribution of the number of emails sent and received by each person.

- **Graph Generation**: This section generates a random graph based on the degree sequence of the original graph. It uses the Havel-Hakimi algorithm to ensure that the generated graph has the same degree sequence as the original graph.

- **K-Means Clustering**: This section applies K-means clustering to the email activity data to identify groups of users with similar email activity patterns.

- **Synthetic Email Generation**: This section generates synthetic email events based on the activity patterns identified in the clustering step. It uses a Poisson process to model the number of emails sent by each user in each hour of the week.

- **Graph Analysis**: This section calculates various statistics and metrics for the synthetic graph, such as the preserved edge ratio, the Kolmogorov-Smirnov statistic for the degree distributions of the original and synthetic graph, the power law fit, the clustering coefficient, and the number and size of cliques. It also plots the number of emails sent by hour of the week for the synthetic data.

Please provide the information about the other files in the directory to continue with the README.

### `DifferentialPrivacy.py`

This is the main script of the project, containing the implementation of various differential privacy algorithms. Here's a brief overview of the functions defined in this file:

- `stable_edge_order(G: nx.Graph) -> list`: Returns a stable order of edges in the graph `G`.
- `pi_graph_projection(G: nx.graph, theta: int, edge_order: list)`: Projects the graph `G` onto the theta-PI graph.
- `quality_function(G: nx.graph, theta: int, epsilon_2: float, Theta=200) -> float`: Calculates the quality score for a given theta value.
- `exponential_mechanism(G, theta_candidates, epsilon_1, epsilon_2, Theta=200) -> int`: Selects the optimal theta value using the exponential mechanism.
- `get_cumulative_histogram(G_theta: nx.Graph, theta: int) -> np.ndarray`: Computes the cumulative histogram of the theta projected graph.
- `add_laplace_noise(cumulative_histogram: np.ndarray, epsilon_2: float) -> np.ndarray`: Adds Laplace noise to the cumulative histogram.
- `get_reconstructed_histogram(noisy_histogram: np.ndarray) -> np.ndarray`: Reconstructs the noisy cumulative histogram.
- `calibrate_histogram(noisy_cumulative_histogram) -> np.ndarray`: Calibrates the noisy cumulative histogram to ensure monotonicity.
- `reconstruct_histogram(calibrated_cumulative_histogram: np.ndarray) -> np.ndarray`: Reconstructs the degree histogram from the calibrated cumulative histogram.
- `node_dp_degree_approximation(epsilon: float, theta: int, degree_dict: dict, deg_dist: np.ndarray) -> np.ndarray`: Implements the Kasiviswanathan et al. 2011 node DP degree approximation.

