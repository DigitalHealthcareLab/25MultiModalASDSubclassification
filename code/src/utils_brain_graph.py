'''
- Consolidated utility functions for graph theory analysis of brain connectivity matrices.
- Works for both multi-subject batch processing and single-subject analysis.
- Threshold matrix while preserving sign information for signed graph analysis.
- Applies proportional thresholding separately to positive and negative connections.
- Suitable for rs-fMRI, where hypo- and hyper-connectivity carry distinct biological meaning.
Date modified: 04-02-2025
'''
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
import time
import logging

logger = logging.getLogger(__name__)

# Check if brainconn is installed and install if not
try:
    from brainconn import utils
    from brainconn.distance import charpath, distance_wei, efficiency_wei
    from brainconn.centrality import betweenness_wei, eigenvector_centrality_und
    from brainconn.degree import degrees_und, strengths_und
    from brainconn.core import assortativity_wei
    from brainconn.clustering import clustering_coef_wu_sign, clustering_coef_wu
    from brainconn.modularity import modularity_louvain_und
    from brainconn.centrality import participation_coef_sign
    from brainconn.reference import randmio_und
    # from brainconn.reference import randomize_graph_partial_und
    # print("✅ All brainconn components successfully imported.")
    brainconn_available = True
except Exception as e:
    print(f"Failed to import brainconn or its components: {e}")
    brainconn_available = False


def compute_graph_metrics(connectivity_matrices, subject_ids, threshold_method='density', 
                          threshold_value=0.2, n_jobs=1, max_random_networks=10):
    """
    Compute graph theory metrics for multiple subjects or a single subject.
    This is the main entry point for graph analysis.
    
    Parameters:
    -----------
    connectivity_matrices : list or numpy.ndarray
        List of connectivity matrices for subjects
    subject_ids : list
        List of subject identifiers
    threshold_method : str, default='density'
        Method to threshold the connectivity matrix
    threshold_value : float, default=0.2
        Threshold value (interpretation depends on threshold_method)
    n_jobs : int, default=1
        Number of parallel jobs
    max_random_networks : int, default=10
        Number of random networks to generate for small-world analysis
        
    Returns:
    --------
    dict
        Dictionary of graph theory metrics for each subject
    """
    if not brainconn_available:
        raise ImportError("brainconn package is required for graph metrics computation")

    logger.info(f"Computing graph metrics for {len(subject_ids)} subjects")
    logger.info(f"Threshold method: {threshold_method}, value: {threshold_value}")
    
    # Determine if we're processing a single subject or multiple
    if len(subject_ids) == 1:
        # Single subject processing - skip parallelization overhead
        subject_id = subject_ids[0]
        matrix = connectivity_matrices[0]
        
        start_time = time.time()
        logger.info(f"Computing graph metrics for single subject {subject_id}")
        
        try:
            metrics = _compute_subject_metrics(
                matrix, 
                threshold_method, 
                threshold_value,
                max_random_networks=max_random_networks
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Graph metrics computation completed in {elapsed_time:.2f} seconds")
            
            return {subject_id: metrics}
            
        except Exception as e:
            logger.error(f"Failed to compute graph metrics for subject {subject_id}: {e}")
            return {subject_id: get_empty_metrics_dict()}
    else:
        # Multiple subject processing
        try:
            if n_jobs != 1:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_compute_subject_metrics)(
                        matrix, 
                        threshold_method, 
                        threshold_value,
                        max_random_networks=max_random_networks
                    )
                    for matrix in connectivity_matrices
                )
            else:
                results = [
                    _compute_subject_metrics(
                        matrix, 
                        threshold_method, 
                        threshold_value,
                        max_random_networks=max_random_networks
                    )
                    for matrix in connectivity_matrices
                ]
                
            return dict(zip(subject_ids, results))
            
        except Exception as e:
            logger.error(f"Failed to compute graph metrics in parallel: {e}")
            raise


def _compute_subject_metrics(connectivity_matrix, threshold_method, threshold_value, max_random_networks=10):
    """
    Compute graph theory metrics for a single subject.
    
    Parameters:
    -----------
    connectivity_matrix : numpy.ndarray
        Connectivity matrix for a single subject
    threshold_method : str
        Method for thresholding
    threshold_value : float
        Threshold value
    max_random_networks : int, default=10
        Number of random networks for small-world analysis
        
    Returns:
    --------
    dict
        Dictionary of graph metrics for the subject
    """
    try:
        # Apply thresholding
        # adj_matrix = threshold_matrix(connectivity_matrix, threshold_method, threshold_value)
        adj_matrix_signed = threshold_matrix(connectivity_matrix, threshold_method, threshold_value)
        adj_matrix = np.abs(adj_matrix_signed)

        
        # Check if matrix contains any connections after thresholding
        if np.sum(adj_matrix != 0) == 0:
            logger.warning(f"Thresholded matrix contains no connections")
            return get_empty_metrics_dict()
        
        # Compute all metrics
        metrics = calculate_graph_metrics(adj_matrix, max_random_networks=max_random_networks)
        return metrics
        
    except Exception as e:
        logger.error(f"Metric computation failed: {e}")
        return get_empty_metrics_dict()


def threshold_matrix(matrix, threshold_method, threshold_value):
    """
    Threshold connectivity matrix using various methods.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Input connectivity matrix
    threshold_method : str
        Method for thresholding ('density', 'absolute', 'proportional', 'significance')
    threshold_value : float
        Value for thresholding (interpretation depends on method)
        
    Returns:
    --------
    numpy.ndarray
        Thresholded connectivity matrix
    """
    if threshold_method == 'density':
        return threshold_preserving_sign(matrix, threshold_value)
    elif threshold_method == 'absolute':
        abs_matrix = np.abs(matrix)
        mask = abs_matrix >= threshold_value
        return matrix * mask
    elif threshold_method == 'significance':
        n = matrix.shape[0]
        r = matrix.copy()
        np.fill_diagonal(r, 0)
        t_vals = r * np.sqrt((n - 2) / (1 - r**2))
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), n - 2))
        adj_matrix = np.zeros_like(matrix)
        adj_matrix[p_vals < threshold_value] = matrix[p_vals < threshold_value]
        return adj_matrix
    elif threshold_method == 'proportional':
        return utils.threshold_proportional(matrix, threshold_value)
    else:
        raise ValueError(f"Unsupported threshold method: {threshold_method}")


def threshold_preserving_sign(matrix, density, mode='both'):
    """
    Threshold matrix while preserving sign information for signed graph analysis.

    Parameters:
    - matrix: 2D numpy array
    - density: float, proportion of connections to retain
    - mode: str, 'both' (default), 'positive_only', or 'negative_only'

    Returns:
    - thresholded matrix with preserved sign structure
    """
    if mode not in {'both', 'positive_only', 'negative_only'}:
        raise ValueError("mode must be one of: 'both', 'positive_only', 'negative_only'")

    result = np.zeros_like(matrix)
    if mode in {'both', 'positive_only'}:
        pos_matrix = matrix * (matrix > 0)
        result += utils.threshold_proportional(pos_matrix, density)
    if mode in {'both', 'negative_only'}:
        neg_matrix = matrix * (matrix < 0)
        result -= utils.threshold_proportional(np.abs(neg_matrix), density)

    return result


def get_empty_metrics_dict():
    """
    Return a dictionary with empty metrics structures.
    Used when computation fails.
    
    Returns:
    --------
    dict
        Empty metrics dictionary
    """
    return {
        'global': {},
        'nodal': {},
        'small_world': {},
        'modularity_communities': None
    }


def calculate_graph_metrics(adj_matrix, max_random_networks=3, use_brainconn=True):
    metrics = {
        'global': {},
        'nodal': {},
        'small_world': {},
        'modularity_communities': None
    }

    try:
        # Distance and path-based metrics
        D, _ = distance_wei(adj_matrix)
        L, _, _, _, _ = charpath(D)
        metrics['global']['char_path_length'] = L

        # Efficiency
        Eglob = efficiency_wei(adj_matrix, local=False)
        Eloc = efficiency_wei(adj_matrix, local=True)
        metrics['global']['global_efficiency'] = Eglob
        metrics['global']['local_efficiency_mean'] = np.mean(Eloc)
        metrics['nodal']['local_efficiency'] = Eloc

        # Clustering
        if use_brainconn:
            C, _ = clustering_coef_wu_sign(adj_matrix)
        else:
            C = clustering_coef_wu(np.abs(adj_matrix))
        metrics['global']['clustering_coef_mean'] = np.mean(C)
        metrics['nodal']['clustering_coef'] = C

        # Centrality + strength
        BC = betweenness_wei(D)
        metrics['nodal']['betweenness_centrality'] = BC
        Deg = degrees_und(np.abs(adj_matrix) > 0)
        Str = strengths_und(adj_matrix)
        metrics['nodal']['degree'] = Deg
        metrics['nodal']['strength'] = Str

        # Assortativity
        Assort = assortativity_wei(adj_matrix)
        metrics['global']['assortativity'] = Assort

        # Eigenvector centrality (catching failures)
        try:
            EC = eigenvector_centrality_und(adj_matrix)
            metrics['nodal']['eigenvector_centrality'] = EC
        except Exception as e:
            logger.warning(f"Eigenvector centrality computation failed: {e}")
            metrics['nodal']['eigenvector_centrality'] = np.zeros_like(Deg)

        # ---------------------
        # Community detection
        # ---------------------
        ci, Q = modularity_louvain_und(adj_matrix)
        metrics['global']['modularity'] = Q
        metrics['modularity_communities'] = ci

        # Community composition
        unique_communities, counts = np.unique(ci, return_counts=True)
        metrics['global']['num_communities'] = len(unique_communities)
        metrics['global']['community_size_mean'] = np.mean(counts)
        metrics['global']['community_size_std'] = np.std(counts)
        metrics['global']['community_affiliations'] = ci.tolist()  # optional for viz

        # Participation coefficient
        try:
            Ppos, Pneg = participation_coef_sign(adj_matrix, ci)
            metrics['nodal']['participation_coefficient_positive'] = Ppos
            metrics['nodal']['participation_coefficient_negative'] = Pneg
        except Exception as e:
            logger.warning(f"Participation coefficient computation failed: {e}")
            metrics['nodal']['participation_coefficient_positive'] = np.zeros_like(Deg)
            metrics['nodal']['participation_coefficient_negative'] = np.zeros_like(Deg)

        # Within-module Z score
        def within_module_z_score(W, ci):
            n = W.shape[0]
            z_scores = np.zeros(n)
            for comm in np.unique(ci):
                idx = np.where(ci == comm)[0]
                if len(idx) <= 1:
                    continue
                strengths = np.sum(W[np.ix_(idx, idx)], axis=1)
                mean = np.mean(strengths)
                std = np.std(strengths)
                z_scores[idx] = (strengths - mean) / std if std != 0 else 0
            return z_scores

        within_z = within_module_z_score(adj_matrix, ci)
        metrics['nodal']['within_module_z_score'] = within_z

        # Connector hubs (optional)
        try:
            if Ppos is not None:
                connector_hubs = np.where((Ppos > 0.62) & (within_z > 2.5))[0]
                metrics['nodal']['connector_hubs'] = connector_hubs.tolist()
            else:
                metrics['nodal']['connector_hubs'] = []
        except Exception as e:
            logger.warning(f"Connector hub identification failed: {e}")
            metrics['nodal']['connector_hubs'] = []

        # ---------------------
        # Small-world metrics
        # ---------------------
        small_world_metrics = calculate_small_worldness_brainconn(
            adj_matrix, num_rand=max_random_networks, rewiring_iters=10
        )
        
        # small_world_metrics = calculate_small_worldness_standalone(
        #     adj_matrix, np.mean(C), L, num_rand=max_random_networks
        # )
        metrics['small_world'] = small_world_metrics

    except Exception as e:
        logger.error(f"Error computing graph metrics: {e}")
        metrics = get_empty_metrics_dict()

    return metrics

def calculate_small_worldness_brainconn(adj_matrix, num_rand=20, rewiring_iters=10):
    """
    Correctly calculate small-world properties using standard methodology.
    
    Parameters:
    - adj_matrix: Adjacency matrix (already thresholded)
    - num_rand: Number of random networks to generate (20+ recommended)
    - rewiring_iters: Number of rewiring iterations (higher = better randomization)
    
    Returns:
    - Dictionary with small-world metrics
    """

    # Ensure matrix is binary or weighted but not signed
    adj_abs = np.abs(adj_matrix)
    
    # 1. Calculate metrics on real network
    C_real = np.mean(clustering_coef_wu(adj_abs))
    D_real, _ = distance_wei(adj_abs)
    L_real, _, _, _, _ = charpath(D_real)
    
    # 2. Generate random networks and calculate their metrics
    C_rand_vals = []
    L_rand_vals = []
    
    for i in range(num_rand):
        # Generate random network preserving degree distribution
        R, _ = randmio_und(adj_abs, rewiring_iters)
        
        # Calculate metrics
        C_rand = np.mean(clustering_coef_wu(R))
        D_rand, _ = distance_wei(R)
        L_rand, _, _, _, _ = charpath(D_rand)
        
        C_rand_vals.append(C_rand)
        L_rand_vals.append(L_rand)
    
    # 3. Calculate average values
    mean_C_rand = np.mean(C_rand_vals)
    mean_L_rand = np.mean(L_rand_vals)
    
    # 4. Calculate small-world parameters
    gamma = C_real / mean_C_rand
    lambda_val = L_real / mean_L_rand
    sigma = gamma / lambda_val
    
    # A small-world network should have gamma > 1 and lambda ≈ 1
    is_small_world = (gamma > 1) and (lambda_val >= 0.8) and (lambda_val <= 1.2)
    
    return {
        'is_small_world': is_small_world,
        'normalized_clustering_coef': gamma,
        'normalized_path_length': lambda_val,
        'small_worldness': sigma,
        'valid_random_networks': num_rand,
        'random_networks_mean_clustering': mean_C_rand,
        'random_networks_mean_path_length': mean_L_rand
    }


# # Old version - not using brainconn
# def calculate_small_worldness_standalone(adj_matrix, C_mean, L, num_rand=3):
#     """
#     Standalone implementation of small-worldness calculation.
#     No dependency on brainconn package, using numpy operations directly.
    
#     Parameters:
#     -----------
#     adj_matrix : numpy.ndarray
#         Adjacency/connectivity matrix
#     C_mean : float
#         Mean clustering coefficient of the original network
#     L : float
#         Characteristic path length of the original network
#     num_rand : int, default=3
#         Number of random networks to generate
        
#     Returns:
#     --------
#     dict
#         Dictionary of small-world metrics
#     """
#     import numpy as np
#     import time
#     from scipy.sparse import csr_matrix
#     from scipy.sparse.csgraph import shortest_path
    
#     start_time = time.time()
#     print(f"Starting standalone small-world computation with {num_rand} random networks")
    
#     # Get binary matrix
#     binary_matrix = (np.abs(adj_matrix) > 0).astype(int)
#     n_nodes = binary_matrix.shape[0]
    
#     # Calculate node degrees
#     degrees = np.sum(binary_matrix, axis=0)
    
#     # Define function to create random network preserving degree distribution
#     def create_random_network():
#         # Start with an empty network
#         random_net = np.zeros((n_nodes, n_nodes), dtype=int)
        
#         # Create "stubs" (half-edges) for each node based on its degree
#         stubs = []
#         for node, degree in enumerate(degrees):
#             stubs.extend([node] * int(degree))
        
#         # Shuffle stubs to randomly connect them
#         np.random.shuffle(stubs)
        
#         # Connect pairs of stubs to form edges
#         for i in range(0, len(stubs), 2):
#             if i + 1 >= len(stubs):
#                 break
                
#             node1, node2 = stubs[i], stubs[i+1]
            
#             # Avoid self-loops and multi-edges
#             if node1 != node2 and random_net[node1, node2] == 0:
#                 random_net[node1, node2] = 1
#                 random_net[node2, node1] = 1
        
#         return random_net
    
#     # Calculate clustering coefficient for a network
#     def calculate_clustering(matrix):
#         C = np.zeros(n_nodes)
#         for i in range(n_nodes):
#             neighbors = np.where(matrix[i, :] > 0)[0]
#             k = len(neighbors)
            
#             if k < 2:
#                 C[i] = 0
#                 continue
                
#             # Count connections between neighbors
#             S = 0
#             for j in neighbors:
#                 for h in neighbors:
#                     if j != h and matrix[j, h] > 0:
#                         S += 1
            
#             C[i] = S / (k * (k - 1))
        
#         return np.mean(C)
    
#     # Calculate shortest path length efficiently
#     def calculate_path_length(matrix):
#         # Convert to sparse matrix for efficient path calculation
#         sparse_matrix = csr_matrix(matrix)
        
#         # Calculate shortest paths
#         dist_matrix = shortest_path(sparse_matrix, directed=False)
        
#         # Calculate average shortest path (ignoring disconnected paths)
#         finite_mask = np.isfinite(dist_matrix)
#         if not np.any(finite_mask):
#             return float('inf')
            
#         L = np.sum(dist_matrix[finite_mask]) / np.sum(finite_mask)
#         return L
    
#     # Generate random networks and calculate their metrics
#     clustering_rand = []
#     path_length_rand = []
    
#     for i in range(num_rand):
#         try:
#             # Generate random network
#             random_net = create_random_network()
            
#             # Calculate metrics
#             C_rand = calculate_clustering(random_net)
#             L_rand = calculate_path_length(random_net)
            
#             clustering_rand.append(C_rand)
#             path_length_rand.append(L_rand)
            
#         except Exception as e:
#             print(f"Error in random network {i+1}: {e}")
#             continue
    
#     # If no valid random networks, return NaN
#     if len(clustering_rand) == 0:
#         print("No valid random networks generated")
#         return {
#             'is_small_world': False,
#             'normalized_clustering_coef': np.nan,
#             'normalized_path_length': np.nan,
#             'small_worldness': np.nan,
#             'valid_random_networks': 0
#         }
    
#     # Calculate small-world metrics
#     mean_C_rand = np.mean(clustering_rand)
#     mean_L_rand = np.mean(path_length_rand)
    
#     gamma = C_mean / mean_C_rand  # Normalized clustering
#     lambda_val = L / mean_L_rand  # Normalized path length
#     sigma = gamma / lambda_val  # Small-worldness measure
    
#     # Check if network satisfies small-world criteria
#     is_small_world = (gamma > 1) and (lambda_val >= 1) and (sigma > 1)
    
#     elapsed_time = time.time() - start_time
#     print(f"Standalone small-world computation completed in {elapsed_time:.2f} seconds")
    
#     return {
#         'is_small_world': is_small_world,
#         'normalized_clustering_coef': gamma,
#         'normalized_path_length': lambda_val,
#         'small_worldness': sigma,
#         'valid_random_networks': len(clustering_rand),
#         'random_networks': {
#             'mean_clustering': mean_C_rand,
#             'mean_path_length': mean_L_rand 
#         }
#     }
