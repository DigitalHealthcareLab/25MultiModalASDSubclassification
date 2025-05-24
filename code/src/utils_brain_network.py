'''
Utility functions for functional network connectivity (FNC) analysis
Including node-wise and edge-wise FNC calculations with parallel processing
'''
"""
Utility functions for functional network connectivity (FNC) analysis
Including node-wise and edge-wise FNC calculations with parallel processing
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_network_connectome(connectome, region_to_network, networks, symmetrize=True):
    """
    Aggregates ROI-to-ROI functional connectivity matrix into network-level FC matrix.

    Parameters:
    - connectome: 2D numpy array (ROI x ROI Pearson correlation matrix)
    - region_to_network: mapping from ROI index to network index or name
    - networks: list of network names or indices
    - symmetrize: if True, symmetrizes the output matrix

    Returns:
    - network_conn: symmetrized (or raw) network-level FC matrix (N_networks x N_networks)
    """
    n_networks = len(networks)
    network_conn = np.zeros((n_networks, n_networks))
    network_counts = np.zeros((n_networks, n_networks))
    n_regions = connectome.shape[0]

    network_to_idx = {name: idx for idx, name in enumerate(networks)}

    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                continue  # skip diagonal (self-connections)

            net_i = region_to_network[i] if isinstance(region_to_network, list) else region_to_network.get(i)
            net_j = region_to_network[j] if isinstance(region_to_network, list) else region_to_network.get(j)

            if net_i is None or net_j is None:
                continue

            if isinstance(net_i, str):
                net_i = network_to_idx.get(net_i)
            if isinstance(net_j, str):
                net_j = network_to_idx.get(net_j)

            if net_i is not None and net_j is not None:
                network_conn[net_i, net_j] += connectome[i, j]
                network_counts[net_i, net_j] += 1

    mask = network_counts > 0
    network_conn[mask] /= network_counts[mask]

    if symmetrize:
        network_conn = (network_conn + network_conn.T) / 2

    return network_conn

def compute_network_metrics(network_connectome):
    """
    Computes basic network-level FNC summary metrics.

    Returns:
    - mean_connectivity: average connectivity per network (row-wise mean)
    - network_strength: L1 norm (sum of abs values) per network
    """
    mean_connectivity = np.mean(network_connectome, axis=1)
    network_strength = np.sum(np.abs(network_connectome), axis=1)
    return {
        'mean_connectivity': mean_connectivity,
        'network_strength': network_strength
    }

def compute_fnc_network_only(connectomes, subject_ids, atlas_mapping):
    if isinstance(atlas_mapping, tuple):
        region_to_network, _, networks, _ = atlas_mapping
    else:
        region_to_network = atlas_mapping.get('region_to_network')
        networks = atlas_mapping.get('networks')

    results = {}
    for idx, subject_id in enumerate(subject_ids):
        conn = connectomes[idx]
        try:
            network_conn = compute_network_connectome(conn, region_to_network, networks, symmetrize=True)
            network_metrics = compute_network_metrics(network_conn)

            results[subject_id] = {
                'network_connectome': network_conn,
                'network_metrics': network_metrics
            }
        except Exception as e:
            logger.error(f"Failed FNC computation for subject {subject_id}: {e}")
            results[subject_id] = {
                'network_connectome': None,
                'network_metrics': None
            }
    return results, networks

# ---------------------------------------------------------------------------
# NETWORK-AVERAGED SEGREGATION & INTEGRATION FUNCTIONS
# ---------------------------------------------------------------------------
def compute_within_between(connectome: np.ndarray, labels: np.ndarray):
    """
    Compute mean within- and between-network connectivity for a subject.
    Returns:
      within_mean: array of shape (K,) for K networks
      between_dict: dict mapping network-pair tuple to mean connectivity
    """
    networks = np.unique(labels)
    K = len(networks)
    within_mean = np.zeros(K)
    between_dict = {}
    # within
    for idx, net in enumerate(networks):
        mask = labels == net
        submat = connectome[np.ix_(mask, mask)]
        triu_vals = submat[np.triu_indices_from(submat, k=1)]
        within_mean[idx] = np.nanmean(triu_vals)
    # between
    from itertools import combinations
    for (i, j) in combinations(networks, 2):
        mask_i = labels == i
        mask_j = labels == j
        vals = connectome[np.ix_(mask_i, mask_j)].ravel()
        between_dict[(i, j)] = np.nanmean(vals)
    return within_mean, between_dict


def compute_segregation_index(within_vals: np.ndarray, between_dict: dict) -> float:
    """
    Segregation Index = (mean(within) - mean(between)) / mean(within)
    """
    m_within = np.nanmean(within_vals)
    m_between = np.nanmean(list(between_dict.values()))
    return (m_within - m_between) / m_within if m_within != 0 else np.nan


def compute_integration_metric(between_dict: dict) -> float:
    """
    Integration metric as mean between-network connectivity
    """
    return np.nanmean(list(between_dict.values()))


def compute_summary_from_network_matrix(netmat):
    """
    netmat: 2D array of shape (K,K) from your pre‑computed Power-network results,
            where netmat[i,i] is the within‑network mean for network i,
                  netmat[i,j] is the between‑network mean for networks i and j.
    Returns:
      seg_index: (mean(diag) – mean(off-diag)) / mean(diag)
      integ: mean(off-diag)
    """
    # Within‑network = diagonal
    within = np.diag(netmat)
    # Between‑network = strictly upper‑triangle
    i, j = np.triu_indices_from(netmat, k=1)
    between = netmat[i, j]

    mean_within  = np.mean(within)
    mean_between = np.mean(between)
    seg_index    = ((mean_within - mean_between) / mean_within
                    if mean_within else np.nan)
    integ        = mean_between
    return seg_index, integ

def aggregate_nodal_metrics_to_network(nodal_metrics, region_to_network, networks):
    """
    Aggregate region-level graph metrics to network level.
    
    Parameters:
    -----------
    nodal_metrics : dict
        Dictionary of nodal metrics, each value is an array of shape (n_regions,)
    region_to_network : dict
        Dictionary mapping region indices to network names
    networks : list
        List of unique network names
        
    Returns:
    --------
    dict
        Dictionary of network-level metrics, each value is an array of shape (n_networks,)
    """
    network_metrics = {}
    
    # Create mapping from network to region indices
    network_to_regions = {net: [] for net in networks}
    for region, network in region_to_network.items():
        if network in networks:
            network_to_regions[network].append(region)
    
    # For each nodal metric, aggregate to network level
    for metric_name, region_values in nodal_metrics.items():
        network_values = np.zeros(len(networks))
        
        for net_idx, network in enumerate(networks):
            regions = network_to_regions[network]
            if regions:
                # Extract metric values for regions in this network
                network_values[net_idx] = np.mean([region_values[r] for r in regions 
                                                  if r < len(region_values)])
            
        network_metrics[metric_name] = network_values
        
    return network_metrics