"""
Utility functions for statistical computation for cluster brain connectivity analysis & comparison
"""
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


def cohens_d(x, y):
    """Calculate Cohen's d effect size between two groups"""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0
    
    # Pooled standard deviation
    dof = nx + ny - 2
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    pooled_std = np.sqrt(pooled_var)
    
    # Effect size
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std else 0


def perform_global_tests(feature_values: dict):
    """
    Run ANOVA/Kruskal-Wallis for a feature across all 4 clusters.
    
    Parameters:
    -----------
    feature_values : dict
        Mapping from cluster_id to list of values for a given feature.
        
    Returns:
    --------
    dict
        Dictionary with test result: test type, p-value, and group means.
    """
    from scipy.stats import f_oneway, kruskal

    clusters = list(feature_values.keys())
    groups = [feature_values[cid] for cid in clusters if feature_values[cid]]
    
    if any(len(g) < 8 for g in groups):
        test = 'kruskal'
        stat, p_val = kruskal(*groups)
    else:
        test = 'anova'
        stat, p_val = f_oneway(*groups)
    
    return {
        'test_type': test,
        'statistic': stat,
        'pvalue': p_val,
        'means': {cid: np.mean(feature_values[cid]) for cid in clusters}
    }

def compute_statistical_significance(matrices1, matrices2, alpha=0.05, apply_correction=True):
    """
    Compute statistical significance of differences between groups with multiple comparison correction.
    
    Parameters:
    -----------
    matrices1 : list
        List of connectivity matrices for cluster 1
    matrices2 : list
        List of connectivity matrices for cluster 2
    alpha : float, optional
        Significance level (default: 0.05)
    apply_correction : bool, optional
        Whether to apply FDR correction for multiple comparisons (default: True)
        
    Returns:
    --------
    tuple
        (pval_matrix, tstat_matrix, effect_size_matrix)
    """
    n_nodes = matrices1[0].shape[0]
    pval_matrix = np.ones((n_nodes, n_nodes))
    tstat_matrix = np.zeros((n_nodes, n_nodes))
    effect_size_matrix = np.zeros((n_nodes, n_nodes))
    
    # Collect p-values for multiple comparison correction
    pvals_flat = []
    upper_tri_indices = np.triu_indices_from(pval_matrix, k=1)
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):  # Upper triangular only
            # Extract connection values for this edge from all subjects in each cluster
            vals1 = [m[i, j] for m in matrices1]
            vals2 = [m[i, j] for m in matrices2]
            
            # Choose appropriate test based on sample size
            if len(vals1) < 8 or len(vals2) < 8:
                try:
                    # Non-parametric test for small samples
                    stat, p_val = stats.mannwhitneyu(vals1, vals2)
                    tstat_matrix[i, j] = stat
                    tstat_matrix[j, i] = stat
                except ValueError:
                    # Fallback to t-test if Mann-Whitney fails
                    t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
                    tstat_matrix[i, j] = t_stat
                    tstat_matrix[j, i] = t_stat
            else:
                # Parametric test for larger samples
                t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
                tstat_matrix[i, j] = t_stat
                tstat_matrix[j, i] = t_stat
            
            # Calculate effect size (Cohen's d)
            if vals1 and vals2 and len(vals1) > 1 and len(vals2) > 1:
                nx, ny = len(vals1), len(vals2)
                dof = nx + ny - 2
                pooled_var = ((nx-1)*np.var(vals1, ddof=1) + (ny-1)*np.var(vals2, ddof=1)) / dof
                if pooled_var > 0:
                    d = (np.mean(vals1) - np.mean(vals2)) / np.sqrt(pooled_var)
                    effect_size_matrix[i, j] = d
                    effect_size_matrix[j, i] = d
            
            # Store p-value
            pval_matrix[i, j] = p_val
            pval_matrix[j, i] = p_val
            pvals_flat.append(p_val)
    
    # Apply FDR correction if requested and if there are p-values to correct
    if apply_correction and pvals_flat:
        rejected, pvals_corrected = fdrcorrection(pvals_flat, alpha=alpha)
        
        # Reconstruct the corrected p-value matrix
        pval_corrected_matrix = np.ones_like(pval_matrix)
        pval_corrected_matrix[upper_tri_indices] = pvals_corrected
        pval_corrected_matrix = pval_corrected_matrix + pval_corrected_matrix.T - np.eye(n_nodes)
        
        return pval_corrected_matrix, tstat_matrix, effect_size_matrix
    
    return pval_matrix, tstat_matrix, effect_size_matrix


def analyze_task_comparison(task, params, cluster_connectomes, avg_connectomes, 
                           region_to_network, networks, subject_graph_metrics,
                           subject_network_metrics, region_coords=None):
    """
    Analyze connectivity differences for a specific joint attention task using pre-computed metrics.
    
    Parameters:
    -----------
    task : str
        Task name (e.g., 'IJA', 'RJA_low', 'RJA_high')
    params : dict
        Parameters specifying clusters to compare and features to analyze
    cluster_connectomes : dict
        Dictionary of connectivity matrices organized by cluster
    avg_connectomes : dict
        Dictionary of average connectivity matrices by cluster
    region_to_network : dict
        Mapping from regions to networks
    networks : list
        List of network names
    subject_graph_metrics : dict
        Pre-computed graph metrics for all subjects by cluster
    subject_network_metrics : dict
        Pre-computed network metrics for all subjects by cluster
    region_coords : dict, optional
        Region coordinates for visualization
        
    Returns:
    --------
    dict
        Results of comparisons
    """
    results = {}
    features = params['features']
    
    # Handle either single comparison or multiple comparisons
    if isinstance(params['clusters'][0], tuple):
        comparisons = params['clusters']
    else:
        comparisons = [params['clusters']]
    
    for comparison in comparisons:
        cluster1, cluster2 = comparison
        print(f"  Comparing clusters {cluster1} vs {cluster2}")
        
        # Get average connectomes
        avg_conn1 = avg_connectomes[cluster1]
        avg_conn2 = avg_connectomes[cluster2]
        
        # Calculate difference
        diff_matrix = avg_conn1 - avg_conn2
        
        # Get matrices for statistical testing
        matrices1 = cluster_connectomes[cluster1]
        matrices2 = cluster_connectomes[cluster2]
        
        # Calculate statistical significance with multiple comparison correction
        print(f"  Computing connection-wise statistical significance...")
        pval_matrix, tstat_matrix, effect_size_matrix = compute_statistical_significance(
            matrices1, matrices2, alpha=0.05, apply_correction=True
        )
        
        # Calculate network-level differences
        network_diff = diff_matrix  # Placeholder if compute_network_connectome is unavailable
        try:
            from src.utils_brain_network import compute_network_connectome
            network_diff = compute_network_connectome(diff_matrix, region_to_network, networks)
        except ImportError:
            print("Warning: compute_network_connectome not available, using raw difference matrix")
        
        # FEATURE-SPECIFIC ANALYSIS using pre-computed metrics
        print(f"  Analyzing features: {features}")
        feature_results = {}
        
        for feature in features:
            print(f"    Analyzing feature: {feature}")
            
            # Extract feature type and use pre-computed metrics
            if 'within_module_z_score' in feature:
                feature_results[feature] = analyze_module_feature(
                    feature, 
                    subject_graph_metrics[cluster1], 
                    subject_graph_metrics[cluster2]
                )
            elif 'network_strength' in feature:
                network_name = feature.split('_')[-1]
                feature_results[feature] = analyze_network_strength(
                    network_name, 
                    subject_network_metrics[cluster1], 
                    subject_network_metrics[cluster2], 
                    networks
                )
            elif 'connector_hub' in feature:
                feature_results[feature] = analyze_hub_feature(
                    subject_graph_metrics[cluster1], 
                    subject_graph_metrics[cluster2]
                )
            elif 'mean_connectivity' in feature:
                network_name = feature.split('_')[-1]
                feature_results[feature] = analyze_mean_connectivity(
                    network_name, 
                    subject_network_metrics[cluster1], 
                    subject_network_metrics[cluster2], 
                    networks
                )
            
            print(f"    Feature {feature} analysis completed")
        
        # Store results
        key = f"{cluster1}_vs_{cluster2}"
        results[key] = {
            'diff_matrix': diff_matrix,
            'pval_matrix': pval_matrix,
            'tstat_matrix': tstat_matrix,
            'effect_size_matrix': effect_size_matrix,
            'network_diff': network_diff,
            'significant_conns': pval_matrix < 0.05,
            'features': params['features'],
            'feature_results': feature_results
        }
        
        print(f"  Comparison {key} completed")
    
    return results


def analyze_module_feature(feature, graph_metrics1, graph_metrics2):
    """
    Analyze module-based features using pre-computed metrics.
    
    Parameters:
    -----------
    feature : str
        Feature name (e.g., 'within_module_z_score_max', 'within_module_z_score_std')
    graph_metrics1 : dict
        Pre-computed graph metrics for cluster 1 subjects
    graph_metrics2 : dict
        Pre-computed graph metrics for cluster 2 subjects
        
    Returns:
    --------
    dict
        Dictionary of results
    """
    # Extract feature type
    feature_type = feature.split('_')[-1]  # 'max', 'std', etc.
    
    # Extract values from pre-computed metrics
    z_scores1 = []
    z_scores2 = []
    
    # Process pre-computed metrics for cluster 1
    for subject_id, metrics in graph_metrics1.items():
        z_score = metrics['nodal'].get('within_module_z_score', [])
        
        if len(z_score) > 0:
            # Apply requested aggregation
            if feature_type == 'max':
                z_scores1.append(np.max(z_score))
            elif feature_type == 'std':
                z_scores1.append(np.std(z_score))
            elif feature_type == 'mean':
                z_scores1.append(np.mean(z_score))
            elif feature_type == 'skew':
                z_scores1.append(stats.skew(z_score))
            else:
                z_scores1.append(np.mean(z_score))  # Default to mean
    
    # Same for cluster 2
    for subject_id, metrics in graph_metrics2.items():
        z_score = metrics['nodal'].get('within_module_z_score', [])
        
        if len(z_score) > 0:
            if feature_type == 'max':
                z_scores2.append(np.max(z_score))
            elif feature_type == 'std':
                z_scores2.append(np.std(z_score))
            elif feature_type == 'mean':
                z_scores2.append(np.mean(z_score))
            elif feature_type == 'skew':
                z_scores2.append(stats.skew(z_score))
            else:
                z_scores2.append(np.mean(z_score))
    
    # Statistical analysis
    if len(z_scores1) < 8 or len(z_scores2) < 8:
        # Non-parametric for small samples
        try:
            stat, p_val = stats.mannwhitneyu(z_scores1, z_scores2)
            test_type = 'mannwhitneyu'
        except ValueError:
            # Fall back if Mann-Whitney fails
            t_stat, p_val = stats.ttest_ind(z_scores1, z_scores2, equal_var=False)
            test_type = 'welch_ttest'
            stat = t_stat
    else:
        # Parametric for larger samples
        t_stat, p_val = stats.ttest_ind(z_scores1, z_scores2, equal_var=False)
        test_type = 'welch_ttest'
        stat = t_stat
    
    # Calculate effect size
    if z_scores1 and z_scores2:
        nx, ny = len(z_scores1), len(z_scores2)
        if nx > 1 and ny > 1:
            dof = nx + ny - 2
            pooled_var = ((nx-1)*np.var(z_scores1, ddof=1) + 
                         (ny-1)*np.var(z_scores2, ddof=1)) / dof
            if pooled_var > 0:
                effect_size = (np.mean(z_scores1) - np.mean(z_scores2)) / np.sqrt(pooled_var)
            else:
                effect_size = np.nan
        else:
            effect_size = np.nan
    else:
        effect_size = np.nan
    
    return {
        'cluster1_values': z_scores1,
        'cluster2_values': z_scores2,
        'cluster1_mean': np.mean(z_scores1) if z_scores1 else np.nan,
        'cluster2_mean': np.mean(z_scores2) if z_scores2 else np.nan,
        'difference': np.mean(z_scores1) - np.mean(z_scores2) if z_scores1 and z_scores2 else np.nan,
        'test_type': test_type,
        'test_stat': stat,
        'pvalue': p_val,
        'effect_size': effect_size,
        'sample_size1': len(z_scores1),
        'sample_size2': len(z_scores2)
    }


def analyze_network_strength(network_name, network_metrics1, network_metrics2, networks):
    """
    Analyze network strength feature for a specific network using pre-computed metrics.
    
    Parameters:
    -----------
    network_name : str
        Name of the network to analyze
    network_metrics1 : dict
        Pre-computed network metrics for cluster 1 subjects
    network_metrics2 : dict
        Pre-computed network metrics for cluster 2 subjects
    networks : list
        List of network names
        
    Returns:
    --------
    dict
        Dictionary of results
    """
    # Get network index
    network_idx = networks.index(network_name) if network_name in networks else None
    
    if network_idx is None:
        print(f"Warning: Network {network_name} not found")
        return None
    
    # Extract pre-computed network strength values
    network_strength1 = []
    network_strength2 = []
    
    # Process cluster 1
    for subject_id, metrics in network_metrics1.items():
        if 'network_strength' in metrics and len(metrics['network_strength']) > network_idx:
            network_strength1.append(metrics['network_strength'][network_idx])
    
    # Process cluster 2
    for subject_id, metrics in network_metrics2.items():
        if 'network_strength' in metrics and len(metrics['network_strength']) > network_idx:
            network_strength2.append(metrics['network_strength'][network_idx])
    
    # Statistical analysis
    if len(network_strength1) < 8 or len(network_strength2) < 8:
        # Non-parametric for small samples
        try:
            stat, p_val = stats.mannwhitneyu(network_strength1, network_strength2)
            test_type = 'mannwhitneyu'
        except ValueError:
            # Fall back if Mann-Whitney fails
            t_stat, p_val = stats.ttest_ind(network_strength1, network_strength2, equal_var=False)
            test_type = 'welch_ttest'
            stat = t_stat
    else:
        # Parametric for larger samples
        t_stat, p_val = stats.ttest_ind(network_strength1, network_strength2, equal_var=False)
        test_type = 'welch_ttest'
        stat = t_stat
    
    # Calculate effect size
    if network_strength1 and network_strength2:
        nx, ny = len(network_strength1), len(network_strength2)
        if nx > 1 and ny > 1:
            dof = nx + ny - 2
            pooled_var = ((nx-1)*np.var(network_strength1, ddof=1) + 
                         (ny-1)*np.var(network_strength2, ddof=1)) / dof
            if pooled_var > 0:
                effect_size = (np.mean(network_strength1) - np.mean(network_strength2)) / np.sqrt(pooled_var)
            else:
                effect_size = np.nan
        else:
            effect_size = np.nan
    else:
        effect_size = np.nan
    
    return {
        'cluster1_values': network_strength1,
        'cluster2_values': network_strength2,
        'cluster1_mean': np.mean(network_strength1) if network_strength1 else np.nan,
        'cluster2_mean': np.mean(network_strength2) if network_strength2 else np.nan,
        'difference': np.mean(network_strength1) - np.mean(network_strength2) if network_strength1 and network_strength2 else np.nan,
        'test_type': test_type,
        'test_stat': stat,
        'pvalue': p_val,
        'effect_size': effect_size,
        'sample_size1': len(network_strength1),
        'sample_size2': len(network_strength2)
    }


def analyze_hub_feature(graph_metrics1, graph_metrics2):
    """
    Analyze hub-based connectivity features using pre-computed metrics.
    
    Parameters:
    -----------
    graph_metrics1 : dict
        Pre-computed graph metrics for cluster 1 subjects
    graph_metrics2 : dict
        Pre-computed graph metrics for cluster 2 subjects
        
    Returns:
    --------
    dict
        Dictionary of results
    """
    # Extract pre-computed hub counts
    hub_counts1 = []
    hub_counts2 = []
    
    # Process cluster 1
    for subject_id, metrics in graph_metrics1.items():
        hubs = metrics['nodal'].get('connector_hubs', [])
        hub_counts1.append(len(hubs))
    
    # Process cluster 2
    for subject_id, metrics in graph_metrics2.items():
        hubs = metrics['nodal'].get('connector_hubs', [])
        hub_counts2.append(len(hubs))
    
    # Statistical analysis
    if len(hub_counts1) < 8 or len(hub_counts2) < 8:
        # Non-parametric for small samples
        try:
            stat, p_val = stats.mannwhitneyu(hub_counts1, hub_counts2)
            test_type = 'mannwhitneyu'
        except ValueError:
            # Fall back if Mann-Whitney fails
            t_stat, p_val = stats.ttest_ind(hub_counts1, hub_counts2, equal_var=False)
            test_type = 'welch_ttest'
            stat = t_stat
    else:
        # Parametric for larger samples
        t_stat, p_val = stats.ttest_ind(hub_counts1, hub_counts2, equal_var=False)
        test_type = 'welch_ttest'
        stat = t_stat
    
    # Calculate effect size
    if hub_counts1 and hub_counts2:
        nx, ny = len(hub_counts1), len(hub_counts2)
        if nx > 1 and ny > 1:
            dof = nx + ny - 2
            pooled_var = ((nx-1)*np.var(hub_counts1, ddof=1) + 
                         (ny-1)*np.var(hub_counts2, ddof=1)) / dof
            if pooled_var > 0:
                effect_size = (np.mean(hub_counts1) - np.mean(hub_counts2)) / np.sqrt(pooled_var)
            else:
                effect_size = np.nan
        else:
            effect_size = np.nan
    else:
        effect_size = np.nan
    
    return {
        'cluster1_values': hub_counts1,
        'cluster2_values': hub_counts2,
        'cluster1_mean': np.mean(hub_counts1) if hub_counts1 else np.nan,
        'cluster2_mean': np.mean(hub_counts2) if hub_counts2 else np.nan,
        'difference': np.mean(hub_counts1) - np.mean(hub_counts2) if hub_counts1 and hub_counts2 else np.nan,
        'test_type': test_type,
        'test_stat': stat,
        'pvalue': p_val,
        'effect_size': effect_size,
        'sample_size1': len(hub_counts1),
        'sample_size2': len(hub_counts2)
    }


def analyze_mean_connectivity(network_name, network_metrics1, network_metrics2, networks):
    """
    Analyze mean connectivity for a specific network using pre-computed metrics.
    
    Parameters:
    -----------
    network_name : str
        Name of the network to analyze
    network_metrics1 : dict
        Pre-computed network metrics for cluster 1 subjects
    network_metrics2 : dict
        Pre-computed network metrics for cluster 2 subjects
    networks : list
        List of network names
        
    Returns:
    --------
    dict
        Dictionary of results
    """
    # Get network index
    network_idx = networks.index(network_name) if network_name in networks else None
    
    if network_idx is None:
        print(f"Warning: Network {network_name} not found")
        return None
    
    # Extract pre-computed mean connectivity values
    mean_conn1 = []
    mean_conn2 = []
    
    # Process cluster 1
    for subject_id, metrics in network_metrics1.items():
        if 'mean_connectivity' in metrics and len(metrics['mean_connectivity']) > network_idx:
            mean_conn1.append(metrics['mean_connectivity'][network_idx])
    
    # Process cluster 2
    for subject_id, metrics in network_metrics2.items():
        if 'mean_connectivity' in metrics and len(metrics['mean_connectivity']) > network_idx:
            mean_conn2.append(metrics['mean_connectivity'][network_idx])
    
    # Statistical analysis
    if len(mean_conn1) < 8 or len(mean_conn2) < 8:
        # Non-parametric for small samples
        try:
            stat, p_val = stats.mannwhitneyu(mean_conn1, mean_conn2)
            test_type = 'mannwhitneyu'
        except ValueError:
            # Fall back if Mann-Whitney fails
            t_stat, p_val = stats.ttest_ind(mean_conn1, mean_conn2, equal_var=False)
            test_type = 'welch_ttest'
            stat = t_stat
    else:
        # Parametric for larger samples
        t_stat, p_val = stats.ttest_ind(mean_conn1, mean_conn2, equal_var=False)
        test_type = 'welch_ttest'
        stat = t_stat
    
    # Calculate effect size
    if mean_conn1 and mean_conn2:
        nx, ny = len(mean_conn1), len(mean_conn2)
        if nx > 1 and ny > 1:
            dof = nx + ny - 2
            pooled_var = ((nx-1)*np.var(mean_conn1, ddof=1) + 
                         (ny-1)*np.var(mean_conn2, ddof=1)) / dof
            if pooled_var > 0:
                effect_size = (np.mean(mean_conn1) - np.mean(mean_conn2)) / np.sqrt(pooled_var)
            else:
                effect_size = np.nan
        else:
            effect_size = np.nan
    else:
        effect_size = np.nan
    
    return {
        'cluster1_values': mean_conn1,
        'cluster2_values': mean_conn2,
        'cluster1_mean': np.mean(mean_conn1) if mean_conn1 else np.nan,
        'cluster2_mean': np.mean(mean_conn2) if mean_conn2 else np.nan,
        'difference': np.mean(mean_conn1) - np.mean(mean_conn2) if mean_conn1 and mean_conn2 else np.nan,
        'test_type': test_type,
        'test_stat': stat,
        'pvalue': p_val,
        'effect_size': effect_size,
        'sample_size1': len(mean_conn1),
        'sample_size2': len(mean_conn2)
    }

def calculate_network_level_significance(comparison_results, network_to_regions, alpha=0.05):
    """
    Calculate network-level statistical significance by aggregating region-level p-values
    
    Parameters:
    -----------
    comparison_results : dict
        Results from task comparisons
    network_to_regions : dict
        Mapping from networks to region indices
    alpha : float, optional
        Significance level (default: 0.05)
        
    Returns:
    --------
    dict
        Updated comparison_results with network-level significance masks
    """
    for task, results in comparison_results.items():
        for comparison_key, data in results.items():
            if 'pval_matrix' in data and 'network_diff' in data:
                # Get dimensions
                n_networks = data['network_diff'].shape[0]
                
                # Create empty matrices for network-level stats
                network_pvals = np.ones((n_networks, n_networks))
                network_significant = np.zeros((n_networks, n_networks), dtype=bool)
                
                # For each pair of networks, aggregate p-values from region pairs
                for i, network1 in enumerate(network_to_regions.keys()):
                    regions1 = network_to_regions[network1]
                    
                    for j, network2 in enumerate(network_to_regions.keys()):
                        regions2 = network_to_regions[network2]
                        
                        # Get all p-values between regions in these networks
                        p_values = []
                        for r1 in regions1:
                            for r2 in regions2:
                                if r1 < data['pval_matrix'].shape[0] and r2 < data['pval_matrix'].shape[1]:
                                    p_values.append(data['pval_matrix'][r1, r2])
                        
                        if p_values:
                            # Use Fisher's method to combine p-values
                            # -2 * sum(log(p)) follows chi-squared with 2k degrees of freedom
                            # where k is the number of p-values being combined
                            from scipy import stats
                            
                            # Filter out p-values that are exactly 1.0 to avoid log(1)=0
                            p_values = [p for p in p_values if p < 1.0]
                            
                            if p_values:
                                statistic = -2 * np.sum(np.log(p_values))
                                p_combined = 1 - stats.chi2.cdf(statistic, 2 * len(p_values))
                                network_pvals[i, j] = p_combined
                
                # Apply FDR correction to network-level p-values
                mask = ~np.eye(n_networks, dtype=bool)  # Exclude diagonal
                p_flat = network_pvals[mask]
                
                if len(p_flat) > 0:
                    _, p_corrected = fdrcorrection(p_flat, alpha=alpha)
                    
                    # Reconstruct the corrected p-value matrix
                    p_corrected_matrix = np.ones_like(network_pvals)
                    p_corrected_matrix[mask] = p_corrected
                    
                    # Create significance mask
                    network_significant = p_corrected_matrix < alpha
                
                # Add network-level significance to results
                data['network_pvals'] = network_pvals
                data['significant_network_conns'] = network_significant
    
    return comparison_results

def calculate_nodal_effect_sizes(comparison_results, subject_graph_metrics, region_info):
    """
    Calculate effect sizes for each brain region based on graph metrics
    
    Parameters:
    -----------
    comparison_results : dict
        Results from task comparisons
    subject_graph_metrics : dict
        Pre-computed graph metrics for all subjects by cluster
    region_info : pd.DataFrame
        Region information including coordinates
        
    Returns:
    --------
    dict
        Updated comparison_results with nodal effect sizes
    """
    metrics_to_use = ['participation_coef', 'within_module_z_score', 'clustering_coef', 'degree']
    
    for task, results in comparison_results.items():
        for comparison_key, data in results.items():
            # Extract cluster IDs
            cluster1, cluster2 = comparison_key.split('_vs_')
            cluster1 = int(cluster1)
            cluster2 = int(cluster2)
            
            # Initialize effect size array
            n_regions = len(region_info)
            effect_sizes = np.zeros(n_regions)
            
            # Get subjects from each cluster
            subjects1 = list(subject_graph_metrics.get(cluster1, {}).keys())
            subjects2 = list(subject_graph_metrics.get(cluster2, {}).keys())
            
            if not subjects1 or not subjects2:
                print(f"Warning: Missing subjects for clusters {cluster1} or {cluster2}")
                continue
            
            # Calculate effect size for each region using combined metrics
            for region_idx in range(min(n_regions, len(region_info))):
                # Collect metric values for each subject
                for metric_name in metrics_to_use:
                    values1 = []
                    values2 = []
                    
                    # Extract nodal metrics for cluster 1
                    for subject_id in subjects1:
                        if subject_id in subject_graph_metrics.get(cluster1, {}):
                            nodal_metrics = subject_graph_metrics[cluster1][subject_id].get('nodal', {})
                            if metric_name in nodal_metrics and region_idx < len(nodal_metrics[metric_name]):
                                values1.append(nodal_metrics[metric_name][region_idx])
                    
                    # Extract nodal metrics for cluster 2
                    for subject_id in subjects2:
                        if subject_id in subject_graph_metrics.get(cluster2, {}):
                            nodal_metrics = subject_graph_metrics[cluster2][subject_id].get('nodal', {})
                            if metric_name in nodal_metrics and region_idx < len(nodal_metrics[metric_name]):
                                values2.append(nodal_metrics[metric_name][region_idx])
                    
                    # Calculate effect size if we have enough data
                    if len(values1) > 1 and len(values2) > 1:
                        from scipy import stats
                        
                        # Calculate Cohen's d
                        nx, ny = len(values1), len(values2)
                        dof = nx + ny - 2
                        pooled_var = ((nx-1)*np.var(values1, ddof=1) + (ny-1)*np.var(values2, ddof=1)) / dof
                        
                        if pooled_var > 0:
                            d = (np.mean(values1) - np.mean(values2)) / np.sqrt(pooled_var)
                            
                            # Update effect size (use absolute largest effect)
                            if abs(d) > abs(effect_sizes[region_idx]):
                                effect_sizes[region_idx] = d
            
            # Add to results
            data['effect_sizes'] = effect_sizes
    
    return comparison_results


def export_feature_summary(comparison_results, output_path="feature_summary.csv"):
    summary = []

    for task, results in comparison_results.items():
        for comparison_key, data in results.items():
            for feature_name, stats in data['feature_results'].items():
                summary.append({
                    'task': task,
                    'comparison': comparison_key,
                    'feature': feature_name,
                    'cluster1_mean': stats.get('cluster1_mean'),
                    'cluster2_mean': stats.get('cluster2_mean'),
                    'difference': stats.get('difference'),
                    'pvalue': stats.get('pvalue'),
                    'effect_size': stats.get('effect_size'),
                    'sample_size1': stats.get('sample_size1'),
                    'sample_size2': stats.get('sample_size2'),
                    'test_type': stats.get('test_type')
                })

    df = pd.DataFrame(summary)
    df.to_csv(output_path, index=False)
    print(f"Feature summary exported to {output_path}")


def generate_summary_report(comparison_results, output_dir):
    """
    Generate a summary report of findings from cluster comparisons.
    
    Parameters:
    -----------
    comparison_results : dict
        Nested dict of results from compare_clusters.
    output_dir : pathlib.Path
        Directory where summary reports will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating statistical summary report...")
    
    # Create a summary for each comparison
    for comparison_key, results in comparison_results.items():
        # Skip special keys like 'omnibus' or 'pairwise_seg_int' that aren't cluster comparisons
        if not isinstance(results, dict) or '_vs_' not in comparison_key:
            continue
            
        try:
            # Try to parse the comparison key
            c1, c2 = comparison_key.split('_vs_')
            
            # Create a report for this comparison
            report_path = output_dir / f"{comparison_key}_summary.txt"
            
            with open(report_path, 'w') as f:
                f.write(f"SUMMARY REPORT: Cluster {c1} vs Cluster {c2}\n")
                f.write("=" * 50 + "\n\n")
                
                # Network-level summary (segregation & integration)
                if 'network_summary' in results:
                    f.write("NETWORK SEGREGATION & INTEGRATION\n")
                    f.write("-" * 40 + "\n")
                    
                    ns = results['network_summary']
                    for metric in ['segregation', 'integration']:
                        if metric in ns:
                            m = ns[metric]
                            f.write(f"{metric.upper()}:\n")
                            f.write(f"  Cluster {c1}: {m.get('c1_mean', 'N/A'):.4f} ± {m.get('c1_std', 'N/A'):.4f}\n")
                            f.write(f"  Cluster {c2}: {m.get('c2_mean', 'N/A'):.4f} ± {m.get('c2_std', 'N/A'):.4f}\n")
                            f.write(f"  t-statistic: {m.get('t', 'N/A'):.4f}\n")
                            f.write(f"  p-value: {m.get('p', 'N/A'):.4f}")
                            if m.get('p_fdr') is not None:
                                f.write(f" (FDR-corrected: {m['p_fdr']:.4f})")
                            f.write("\n")
                            if 'effect_size_d' in m:
                                f.write(f"  Effect size (Cohen's d): {m['effect_size_d']:.4f}\n")
                            f.write("\n")
                
                # Graph metrics
                if 'graph' in results and isinstance(results['graph'], dict):
                    # Global metrics
                    if 'global' in results['graph']:
                        f.write("GLOBAL GRAPH METRICS\n")
                        f.write("-" * 40 + "\n")
                        
                        for metric, data in results['graph']['global'].items():
                            if isinstance(data, dict):
                                f.write(f"{metric.upper()}:\n")
                                f.write(f"  Cluster {c1}: {data.get('c1_mean', 'N/A'):.4f} ± {data.get('c1_std', 'N/A'):.4f}\n")
                                f.write(f"  Cluster {c2}: {data.get('c2_mean', 'N/A'):.4f} ± {data.get('c2_std', 'N/A'):.4f}\n")
                                f.write(f"  t-statistic: {data.get('t', 'N/A'):.4f}\n")
                                f.write(f"  p-value: {data.get('p', 'N/A'):.4f}")
                                if data.get('p_fdr') is not None:
                                    f.write(f" (FDR-corrected: {data['p_fdr']:.4f})")
                                f.write("\n")
                                if 'effect_size_d' in data:
                                    f.write(f"  Effect size (Cohen's d): {data['effect_size_d']:.4f}\n")
                                f.write("\n")
                    
                    # Network-level metrics (if available)
                    if 'network_level' in results['graph']:
                        f.write("NETWORK-LEVEL GRAPH METRICS\n")
                        f.write("-" * 40 + "\n")
                        
                        for metric, data in results['graph']['network_level'].items():
                            if isinstance(data, dict):
                                f.write(f"{metric.upper()}:\n")
                                if 'significant_networks' in data and data['significant_networks']:
                                    f.write(f"  Significantly different networks (p < 0.05): {', '.join(data['significant_networks'])}\n")
                                if 'significant_networks_fdr' in data and data['significant_networks_fdr']:
                                    f.write(f"  FDR-corrected significant networks: {', '.join(data['significant_networks_fdr'])}\n")
                                f.write("\n")
                
                # Network matrix comparison summary
                if 'network' in results and isinstance(results['network'], dict):
                    f.write("NETWORK CONNECTIVITY MATRIX COMPARISON\n")
                    f.write("-" * 40 + "\n")
                    
                    net = results['network']
                    f.write(f"  Cluster {c1}: {net.get('cluster1_n', 'N/A')} subjects\n")
                    f.write(f"  Cluster {c2}: {net.get('cluster2_n', 'N/A')} subjects\n")
                    f.write(f"  Significant connections: {net.get('n_significant', 'N/A')}\n")
                    if 'n_fdr_significant' in net:
                        f.write(f"  FDR-corrected significant connections: {net['n_fdr_significant']}\n")
                    f.write("\n")
                
            print(f"Summary report generated: {report_path}")
            
        except Exception as e:
            print(f"Error generating summary for {comparison_key}: {e}")
            continue
    
    print("Summary report generation complete.")


def save_comparison_results(comparison_results, output_path):
    """
    Save comparison results as pickles and export a flat CSV summary
    (including FDR-corrected p-values) in the specified output directory.

    Parameters:
    -----------
    comparison_results : dict
        Nested dict of results from compare_clusters.
    output_path : pathlib.Path
        Directory where results and summary CSV will be saved.
    """
    
    # 1) Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # 2) Check if we have a valid dictionary
    if not isinstance(comparison_results, dict):
        print(f"Warning: comparison_results is not a dictionary. Type: {type(comparison_results)}")
        # Save whatever we received
        with open(output_path / 'raw_results.pkl', 'wb') as f:
            pickle.dump(comparison_results, f)
        print(f"Raw results saved to {output_path / 'raw_results.pkl'}")
        return

    # 3) Save the entire results dict
    with open(output_path / 'all_comparisons.pkl', 'wb') as f:
        pickle.dump(comparison_results, f)

    # 4) Save each comparison's detailed results
    for comp_key, comp_data in comparison_results.items():
        if not isinstance(comp_data, dict):
            # Handle non-dictionary data within the results
            with open(output_path / f"{comp_key}_data.pkl", 'wb') as f:
                pickle.dump(comp_data, f)
            continue
            
        comp_dir = output_path / comp_key
        comp_dir.mkdir(exist_ok=True)
        for result_type, data in comp_data.items():
            fname = comp_dir / f"{result_type}_comparison.pkl"
            with open(fname, 'wb') as f:
                pickle.dump(data, f)
    
    # 5) Export pairwise segmentation and integration results
    if 'pairwise_seg_int' in comparison_results:
        df = pd.DataFrame(comparison_results['pairwise_seg_int'])
        df.to_csv(output_path / 'seg_int_pairwise.csv', index=False)

    # 6) Build a flat summary of key features
    rows = []
    for comp_key, res in comparison_results.items():
        # Skip if not a dictionary
        if not isinstance(res, dict):
            continue
            
        # a) Network summary (segregation & integration)
        if 'network_summary' in res:
            ns = res['network_summary']
            for feat in ('segregation', 'integration'):
                if feat in ns:
                    vals = ns[feat]
                    rows.append({
                        'comparison': comp_key,
                        'feature': feat,
                        'cluster1_mean': vals.get('c1_mean'),
                        'cluster2_mean': vals.get('c2_mean'),
                        'difference': vals.get('c1_mean', 0) - vals.get('c2_mean', 0),
                        'tstat': vals.get('t'),
                        'pvalue': vals.get('p'),
                        'p_fdr': vals.get('p_fdr'),
                        'test': 'Welch t-test'
                    })
        # b) Global graph metrics
        if 'graph' in res and isinstance(res['graph'], dict) and 'global' in res['graph']:
            for metric, gdata in res['graph']['global'].items():
                if isinstance(gdata, dict):
                    rows.append({
                        'comparison': comp_key,
                        'feature': metric,
                        'cluster1_mean': gdata.get('c1_mean'),
                        'cluster2_mean': gdata.get('c2_mean'),
                        'difference': gdata.get('c1_mean', 0) - gdata.get('c2_mean', 0),
                        'tstat': gdata.get('t'),
                        'pvalue': gdata.get('p'),
                        'p_fdr': gdata.get('p_fdr'),
                        'test': gdata.get('test_type', 'Welch t-test')
                    })

    # 7) Export summary to CSV
    if rows:
        df = pd.DataFrame(rows)
        csv_path = output_path / 'feature_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"Feature summary CSV saved to {csv_path}")
    else:
        print("No features to summarize in CSV.")

    print(f"All comparison results and summaries have been saved under {output_path}")