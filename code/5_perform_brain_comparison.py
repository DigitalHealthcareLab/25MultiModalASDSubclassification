'''
Brain Network Analysis Pipeline - Cluster Comparison
===============================================

A modular pipeline for resting-state fMRI analyses comparing different clusters:
- Comparing functional network connectivity between clusters
- Analyzing graph theory metrics differences between clusters
- Network-Based Statistic (NBS) for identifying subnetwork differences

This script performs cluster-level analyses on pre-computed individual results.

Author: Chanyoung Ko (Modified version)
Date: 04-16-2025
'''
import argparse
import numpy as np
import pandas as pd
import pickle
import itertools

import scipy.stats as stats
from statsmodels.stats.weightstats import ttest_ind

from scipy.stats import f_oneway, kruskal, ttest_ind
from statsmodels.stats.multitest import multipletests

# Local imports - utility modules
from src.utils_seed import set_seed
from src.utils_brain_mapping import get_atlas_mapping
from src.utils_brain_network import compute_summary_from_network_matrix, aggregate_nodal_metrics_to_network
from src.utils_brain_stats import generate_summary_report, save_comparison_results, cohens_d

# Import configuration settings
from src.utils_config import (
    FMRI_PATH, ATLAS_PATH, CLUSTER_DF_PATH, 
    MULTIMODAL_DF_PATH, OUTPUT_BRAIN_DIR,
    dataset_name, atlas_name, timeseries_type, connectome_type, load_id_mapping
)

def parse_args():
    """Parse command-line arguments for configuring analysis parameters."""
    parser = argparse.ArgumentParser(description="Brain Network Analysis Pipeline - Cluster Comparison")
    
    # Cluster comparison (only required parameter)
    parser.add_argument('--cluster_comparison', type=str, nargs='+', 
                        help='Clusters to compare, format: 1,3 4,2 (for comparing cluster 1 vs 3 and 4 vs 2). ' +
                             'If omitted, all pairs of clusters will be compared.')
    
    # Optional overrides
    parser.add_argument('--cluster_column', type=str, default='cluster',
                        help='Column name in dataframe containing cluster labels')
    parser.add_argument('--id_column', type=str, default='id',
                        help='Column name in dataframe containing subject IDs')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for statistical tests')
    parser.add_argument('--apply_fdr', action='store_true', default=True,
                        help='Apply FDR correction for multiple comparisons')
    parser.add_argument('--no_fdr', dest='apply_fdr', action='store_false',
                        help='Do NOT apply FDR correction for multiple comparisons')
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Number of parallel jobs')
    parser.add_argument('--seed', type=int, default=14,
                        help='Random seed for reproducibility')
    parser.add_argument('--visualization', action='store_true',
                        help='Generate visualizations')
    
    return parser.parse_args()


def setup_paths():
    """Set up all necessary paths based on configuration."""
    # Create a timestamp for unique output directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base paths for individual results (from batch processing)
    results_base = FMRI_PATH / dataset_name / 'results'
    base = results_base / f"{atlas_name}_{timeseries_type}_{connectome_type}"
    
    # Output directory for cluster analysis
    cluster_output_dir = OUTPUT_BRAIN_DIR
    
    # Create path dictionary
    paths = {
        # Input paths (individual results)
        'fnc_network': base / 'fnc_analyses' / 'network_wise',
        'rss_features': base / 'fnc_analyses' / 'rss_features',
        'graph_results': base / 'graph_analyses',
        'merged_features': base / 'merged_fmri_features',
        
        # Output paths (cluster results)
        'cluster_comparisons': cluster_output_dir / 'comparisons',
        'visualizations': cluster_output_dir / 'visualizations',
        'statistics': cluster_output_dir / 'statistics',
        
        # Atlas data
        'atlas': ATLAS_PATH,
        
        # Cluster data
        'cluster_dataframe': CLUSTER_DF_PATH
    }
    
    # Create output directories
    for path_name in ['cluster_comparisons', 'visualizations', 'statistics']:
        paths[path_name].mkdir(parents=True, exist_ok=True)
    
    return paths


def load_cluster_data(cluster_path, id_column, cluster_column):
    """Load cluster assignments from dataframe."""
    try:
        df = pd.read_csv(cluster_path)
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in cluster dataframe")
        if cluster_column not in df.columns:
            raise ValueError(f"Cluster column '{cluster_column}' not found in cluster dataframe")
        
        # Ensure subject IDs are strings
        df[id_column] = df[id_column].astype(str)
        
        # Extract cluster mapping
        cluster_mapping = dict(zip(df[id_column], df[cluster_column]))
        
        # Organize subjects by cluster
        clusters = {}
        for cluster_id in sorted(df[cluster_column].unique()):
            clusters[cluster_id] = list(df[df[cluster_column] == cluster_id][id_column])
        
        print(f"Loaded {len(df)} subjects with {len(clusters)} clusters")
        for cluster_id, subjects in clusters.items():
            print(f"  Cluster {cluster_id}: {len(subjects)} subjects")
        
        return cluster_mapping, clusters
    
    except Exception as e:
        print(f"Error loading cluster data: {e}")
        return None, None


def load_individual_results(paths, subject_ids=None):
    """Load pre-computed individual analysis results for specified subjects."""
    results = {}
    
    # For network-wise FNC results
    network_results = {}
    network_file_pattern = "network_connectome_*.pkl"
    
    for file_path in paths['fnc_network'].glob(network_file_pattern):
        subject_id = file_path.stem.split('_')[-1]
        if subject_ids is None or subject_id in subject_ids:
            try:
                with open(file_path, 'rb') as f:
                    network_results[subject_id] = pickle.load(f)
            except Exception as e:
                print(f"Error loading network results for {subject_id}: {e}")
    
    print(f"Loaded network results for {len(network_results)} subjects")
    results['network'] = network_results
    
    # Load network names if available
    network_names_path = paths['fnc_network'] / "network_names.pkl"
    if network_names_path.exists():
        with open(network_names_path, 'rb') as f:
            results['network_names'] = pickle.load(f)
    
    # For graph metrics
    graph_results = {}
    graph_file_pattern = "graph_metrics_*.pkl"
    
    for file_path in paths['graph_results'].glob(graph_file_pattern):
        subject_id = file_path.stem.split('_')[-1]
        if subject_ids is None or subject_id in subject_ids:
            try:
                with open(file_path, 'rb') as f:
                    graph_results[subject_id] = pickle.load(f)
            except Exception as e:
                print(f"Error loading graph results for {subject_id}: {e}")
    
    print(f"Loaded graph metrics for {len(graph_results)} subjects")
    results['graph'] = graph_results
    
    # For merged features
    merged_results = {}
    merged_file_pattern = "merged_fmri_features_*.pkl"
    
    for file_path in paths['merged_features'].glob(merged_file_pattern):
        subject_id = file_path.stem.split('_')[-1]
        if subject_ids is None or subject_id in subject_ids:
            try:
                with open(file_path, 'rb') as f:
                    merged_results[subject_id] = pickle.load(f)
            except Exception as e:
                print(f"Error loading merged features for {subject_id}: {e}")
    
    print(f"Loaded merged features for {len(merged_results)} subjects")
    results['merged'] = merged_results
    
    return results


def organize_by_cluster(results, cluster_mapping):
    """Organize individual results by cluster."""
    cluster_comparisons = {}
    
    # For each result type (network, graph, merged)
    for result_type, result_data in results.items():
        if isinstance(result_data, dict):
            cluster_comparisons[result_type] = {}
            
            # Group subjects by cluster
            for subject_id, subject_data in result_data.items():
                if subject_id in cluster_mapping:
                    cluster_id = cluster_mapping[subject_id]
                    
                    if cluster_id not in cluster_comparisons[result_type]:
                        cluster_comparisons[result_type][cluster_id] = {}
                    
                    cluster_comparisons[result_type][cluster_id][subject_id] = subject_data
    
    # Print summary of organized results
    for result_type, clusters in cluster_comparisons.items():
        print(f"\nOrganized {result_type} results by cluster:")
        for cluster_id, subjects in clusters.items():
            print(f"  Cluster {cluster_id}: {len(subjects)} subjects")
    
    return cluster_comparisons


def compare_clusters(cluster_comparisons, cluster_pairs, atlas_mapping=None, atlas_labels=None, alpha=0.05, apply_fdr=True):
    """
    Compare metrics between specified pairs of clusters, with both region-level 
    and network-level analysis of graph metrics.
    """
    # Unpack atlas mapping
    if atlas_mapping and isinstance(atlas_mapping, tuple) and len(atlas_mapping) >= 3:
        region_to_network, _, networks, _ = atlas_mapping
    else:
        region_to_network = None
        networks = None
    
    labels = atlas_labels  # 1D numpy array of network IDs

    comparison_results = {}

    for c1, c2 in cluster_pairs:
        key = f"{c1}_vs_{c2}"
        comparison_results[key] = {}

        # --- 1) NETWORK‑AVERAGED SUMMARY (segregation & integration) ---
        nets1 = [d['network_connectome'] for d in cluster_comparisons.get('network', {}).get(c1, {}).values()]
        nets2 = [d['network_connectome'] for d in cluster_comparisons.get('network', {}).get(c2, {}).values()]

        if labels is not None and nets1 and nets2:
            # Compute per‐subject indices from network matrices
            seg1, int1 = zip(*(compute_summary_from_network_matrix(net) for net in nets1))
            seg2, int2 = zip(*(compute_summary_from_network_matrix(net) for net in nets2))
            # turn tuples back into lists for consistency
            seg1, int1 = list(seg1), list(int1)
            seg2, int2 = list(seg2), list(int2)

            # Two‐sample t‑tests
            t_seg, p_seg = stats.ttest_ind(seg1, seg2, equal_var=False)
            t_int, p_int = stats.ttest_ind(int1, int2, equal_var=False)
            
            # Effect sizes (Cohen's d)
            d_seg = cohens_d(seg1, seg2)
            d_int = cohens_d(int1, int2)

            # FDR correction across these two tests
            pvals = [p_seg, p_int]
            if apply_fdr:
                _, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
            else:
                pvals_fdr = [None, None]

            comparison_results[key]['network_summary'] = {
                'segregation': {
                    'c1_mean': float(np.nanmean(seg1)),
                    'c2_mean': float(np.nanmean(seg2)),
                    'c1_std': float(np.nanstd(seg1)),
                    'c2_std': float(np.nanstd(seg2)),
                    't': float(t_seg),
                    'p': float(p_seg),
                    'p_fdr': float(pvals_fdr[0]) if pvals_fdr[0] is not None else None,
                    'effect_size_d': float(d_seg)
                },
                'integration': {
                    'c1_mean': float(np.nanmean(int1)),
                    'c2_mean': float(np.nanmean(int2)),
                    'c1_std': float(np.nanstd(int1)),
                    'c2_std': float(np.nanstd(int2)),
                    't': float(t_int),
                    'p': float(p_int),
                    'p_fdr': float(pvals_fdr[1]) if pvals_fdr[1] is not None else None,
                    'effect_size_d': float(d_int)
                }
            }
        
        # --- 2) FULL NETWORK‑MATRIX COMPARISON (edge‑wise) ---
        if 'network' in cluster_comparisons:
            # Extract network connectomes for each cluster
            cluster1_network = [data['network_connectome'] for data in cluster_comparisons['network'].get(c1, {}).values() 
                               if data and data.get('network_connectome') is not None]
            
            cluster2_network = [data['network_connectome'] for data in cluster_comparisons['network'].get(c2, {}).values() 
                               if data and data.get('network_connectome') is not None]
            
            if cluster1_network and cluster2_network:
                # Calculate average network connectomes
                avg_cluster1_network = np.mean(cluster1_network, axis=0)
                avg_cluster2_network = np.mean(cluster2_network, axis=0)
                
                # Calculate difference
                network_diff = avg_cluster1_network - avg_cluster2_network
                
                # Perform statistical testing
                n_networks = avg_cluster1_network.shape[0]
                t_values = np.zeros((n_networks, n_networks))
                p_values = np.zeros((n_networks, n_networks))
                
                # Iterate through each network connection
                for i in range(n_networks):
                    for j in range(n_networks):
                        # Extract connection values for all subjects, forcing each value to float
                        conn1 = [float(net[i, j]) for net in cluster1_network]
                        conn2 = [float(net[i, j]) for net in cluster2_network]
                        print("conn1 sample type:", type(conn1[0]), "and shape (if array):", np.shape(conn1[0]))
                        
                        # Perform t-test
                        t_stat, p_val = stats.ttest_ind(conn1, conn2, equal_var=False)
                        
                        # Ensure t_stat and p_val are scalars
                        try:
                            t_stat = float(np.squeeze(t_stat))
                        except Exception as e:
                            print(f"Error converting t_stat at connection ({i},{j}):", e)
                        try:
                            p_val = float(np.squeeze(p_val))
                        except Exception as e:
                            print(f"Error converting p_val at connection ({i},{j}):", e)
                        
                        # Store values in matrices
                        t_values[i, j] = t_stat
                        p_values[i, j] = p_val
                
                # Create significance mask
                significant_mask = p_values < alpha
                
                # Apply FDR correction if requested
                if apply_fdr:
                    # Flatten p-values for correction
                    flat_p = p_values.flatten()
                    # Apply correction
                    _, p_fdr_corrected, _, _ = multipletests(flat_p, alpha=alpha, method='fdr_bh')
                    # Reshape back to matrix
                    p_fdr_corrected = p_fdr_corrected.reshape(p_values.shape)
                    # Create FDR-corrected mask
                    fdr_corrected_mask = p_fdr_corrected < alpha
                else:
                    p_fdr_corrected = None
                    fdr_corrected_mask = None
                
                # Count significant connections
                n_significant = np.sum(significant_mask)
                n_fdr_significant = np.sum(fdr_corrected_mask) if fdr_corrected_mask is not None else 0
                
                # Print statistical summary
                print(f"  Network comparison: {len(cluster1_network)} vs {len(cluster2_network)} subjects")
                print(f"  Found {n_significant} significant connections (p < {alpha})")
                if apply_fdr:
                    print(f"  After FDR correction: {n_fdr_significant} significant connections")
                
                # Store results
                comparison_results[key]['network'] = {
                    'cluster1_avg': avg_cluster1_network,
                    'cluster2_avg': avg_cluster2_network,
                    'difference': network_diff,
                    'cluster1_n': len(cluster1_network),
                    'cluster2_n': len(cluster2_network),
                    'tvalues': t_values,
                    'pvalues': p_values,
                    'significant_mask': significant_mask,
                    'fdr_corrected_pvalues': p_fdr_corrected,
                    'fdr_corrected_mask': fdr_corrected_mask,
                    'n_significant': n_significant,
                    'n_fdr_significant': n_fdr_significant
                }
                
                # Print top 5 most significant connections
                if n_significant > 0:
                    # Find indices of lowest p-values
                    flat_indices = np.argsort(p_values.flatten())
                    top_indices = flat_indices[:5]  # Top 5
                    
                    print("\n  Top 5 most significant connections:")
                    print("  ---------------------------------------")
                    print("  | Network 1 | Network 2 |  T-value  |  P-value  | Significant after FDR |")
                    print("  |-----------|-----------|-----------|-----------|------------------------|")
                    
                    for idx in top_indices:
                        # Convert flat index to 2D
                        i, j = np.unravel_index(idx, p_values.shape)
                        t_val = t_values[i, j]
                        p_val = p_values[i, j]
                        fdr_sig = "Yes" if fdr_corrected_mask is not None and fdr_corrected_mask[i, j] else "No"
                        
                        print(f"  | {i:9} | {j:9} | {t_val:9.3f} | {p_val:9.5f} | {fdr_sig:22} |")
            else:
                print(f"  Skipping network comparison: insufficient data")
        
        # --- 3) GRAPH METRICS COMPARISON ---
        graph_block = {
            'global': {},
            'nodal': {},
            'network_level': {}  # New section for network-level aggregated metrics
        }

        # 3.1) Global metrics comparison (unchanged)
        group1_global = list(cluster_comparisons.get('graph', {}).get(c1, {}).values())
        group2_global = list(cluster_comparisons.get('graph', {}).get(c2, {}).values())

        if group1_global and group2_global:
            sample = group1_global[0]['global']
            for name, example in sample.items():
                # Skip non-scalar metrics
                if not np.isscalar(example):
                    continue

                # Gather values
                vals1 = [s['global'].get(name) for s in group1_global if s.get('global')]
                vals2 = [s['global'].get(name) for s in group2_global if s.get('global')]
                
                # Remove None values
                vals1 = [v for v in vals1 if v is not None]
                vals2 = [v for v in vals2 if v is not None]
                
                if not vals1 or not vals2:
                    continue

                # Two‑sample t‑test
                t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
                
                # Effect size (Cohen's d)
                d = cohens_d(vals1, vals2)

                graph_block['global'][name] = {
                    'c1_mean': float(np.mean(vals1)),
                    'c2_mean': float(np.mean(vals2)),
                    'c1_std': float(np.std(vals1)),
                    'c2_std': float(np.std(vals2)),
                    't': float(t_stat),
                    'p': float(p_val),
                    'effect_size_d': float(d)
                }

        # 3.2) Network-level comparison (new section)
        if region_to_network and networks and group1_global and group2_global:
            # First, aggregate nodal metrics to network level for each subject
            network_metrics1 = []
            network_metrics2 = []
            
            for subject in group1_global:
                if subject.get('nodal'):
                    net_metrics = aggregate_nodal_metrics_to_network(subject['nodal'], region_to_network, networks)
                    network_metrics1.append(net_metrics)
            
            for subject in group2_global:
                if subject.get('nodal'):
                    net_metrics = aggregate_nodal_metrics_to_network(subject['nodal'], region_to_network, networks)
                    network_metrics2.append(net_metrics)
            
            # Compare network-level metrics between groups
            if network_metrics1 and network_metrics2:
                # Get list of all metrics
                all_metrics = set()
                for metrics in network_metrics1 + network_metrics2:
                    all_metrics.update(metrics.keys())
                
                for metric_name in all_metrics:
                    # Extract network-level values for this metric
                    metric_vals1 = [m.get(metric_name, []) for m in network_metrics1 if m.get(metric_name) is not None]
                    metric_vals2 = [m.get(metric_name, []) for m in network_metrics2 if m.get(metric_name) is not None]
                    
                    if not metric_vals1 or not metric_vals2:
                        continue
                    
                    # Stack into arrays (subjects x networks)
                    try:
                        arr1 = np.vstack(metric_vals1)
                        arr2 = np.vstack(metric_vals2)
                    except:
                        continue
                    
                    # Skip if shapes don't match
                    if arr1.shape[1] != arr2.shape[1] or arr1.shape[1] != len(networks):
                        continue
                    
                    # Compare each network separately
                    tvals = []
                    pvals = []
                    effect_sizes = []
                    c1_means = []
                    c2_means = []
                    c1_stds = []
                    c2_stds = []
                    
                    for net_idx in range(len(networks)):
                        net_vals1 = arr1[:, net_idx]
                        net_vals2 = arr2[:, net_idx]
                        
                        # t-test
                        t_stat, p_val = stats.ttest_ind(net_vals1, net_vals2, equal_var=False)
                        
                        # Effect size
                        d = cohens_d(net_vals1, net_vals2)
                        
                        tvals.append(float(t_stat))
                        pvals.append(float(p_val))
                        effect_sizes.append(float(d))
                        c1_means.append(float(np.mean(net_vals1)))
                        c2_means.append(float(np.mean(net_vals2)))
                        c1_stds.append(float(np.std(net_vals1)))
                        c2_stds.append(float(np.std(net_vals2)))
                    
                    # Apply FDR correction across networks
                    if apply_fdr:
                        _, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
                    else:
                        pvals_fdr = [None] * len(pvals)
                    
                    graph_block['network_level'][metric_name] = {
                        'networks': networks,
                        'tvalues': tvals,
                        'pvalues': pvals,
                        'p_fdr': pvals_fdr if all(p is not None for p in pvals_fdr) else None,
                        'effect_sizes': effect_sizes,
                        'c1_means': c1_means,
                        'c2_means': c2_means,
                        'c1_stds': c1_stds,
                        'c2_stds': c2_stds
                    }
                    
                    # Flag significant findings (uncorrected and FDR-corrected)
                    sig_networks = [networks[i] for i, p in enumerate(pvals) if p < alpha]
                    if apply_fdr:
                        sig_networks_fdr = [networks[i] for i, p in enumerate(pvals_fdr) if p < alpha]
                        graph_block['network_level'][metric_name]['significant_networks_fdr'] = sig_networks_fdr
                    
                    graph_block['network_level'][metric_name]['significant_networks'] 

        if graph_block['global'] or graph_block['nodal']:
            comparison_results[key]['graph'] = graph_block
    
    return comparison_results


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Set up paths (based on config, not args)
    paths = setup_paths()
    print("\nUsing paths from config:")
    print(f"Input FNC network path: {paths['fnc_network']}")
    print(f"Input graph results path: {paths['graph_results']}")
    print(f"Output cluster results path: {paths['cluster_comparisons']}")
    print(f"Atlas path: {paths['atlas']}")
    print(f"Cluster dataframe path: {paths['cluster_dataframe']}")
    
    # Load cluster data
    cluster_mapping, clusters = load_cluster_data(
        paths['cluster_dataframe'], 
        args.id_column, 
        args.cluster_column
    )
    
    if cluster_mapping is None:
        print("Failed to load cluster data. Exiting.")
        return 1
    
    # Load ID mapping between participant IDs and fMRI IDs
    id_to_fmri = load_id_mapping(MULTIMODAL_DF_PATH)
    if not id_to_fmri:
        print("Failed to load ID mapping. Exiting.")
        return 1
    
    # Parse cluster comparisons
    cluster_pairs = []
    if args.cluster_comparison:
        for pair_str in args.cluster_comparison:
            try:
                c1, c2 = map(int, pair_str.split(','))
                cluster_pairs.append((c1, c2))
            except ValueError:
                print(f"Invalid cluster pair format: {pair_str}. Should be 'cluster1,cluster2'")
                continue
    else:
        # If no specific comparisons provided, compare all pairs of clusters
        cluster_ids = sorted(clusters.keys())
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                cluster_pairs.append((cluster_ids[i], cluster_ids[j]))
    
    print(f"Will compare the following cluster pairs: {cluster_pairs}")
    
    # Convert cluster subjects to fMRI IDs
    fmri_subjects = []
    id_to_cluster = {}  # For mapping fMRI IDs back to clusters
    
    for cluster_id, subjects in clusters.items():
        for subject_id in subjects:
            if subject_id in id_to_fmri:
                fmri_id = id_to_fmri[subject_id]
                fmri_subjects.append(fmri_id)
                id_to_cluster[fmri_id] = cluster_id
                print(f"Mapped subject {subject_id} to fMRI ID {fmri_id} in cluster {cluster_id}")
            else:
                print(f"Warning: No fMRI ID mapping found for subject {subject_id}")
    
    print(f"Found fMRI ID mappings for {len(fmri_subjects)} out of {sum(len(s) for s in clusters.values())} participants")
    
    # --------------------------------------------------
    # 1) Unpack your atlas mapping
    # --------------------------------------------------
    # get_atlas_mapping now returns:
    #   region_to_network : dict {region_idx → network_name}
    #   region_df         : pandas.DataFrame of all region info
    #   networks          : list of unique network names
    #   network_to_regions: dict {network_name → [region_idx, ...]}
    region_to_network, region_df, networks, network_to_regions = get_atlas_mapping(paths['atlas'])
    print(f"Atlas mapping loaded with {len(region_to_network)} regions and {len(networks)} networks")

    # --------------------------------------------------
    # 2) Build a labels array of shape (n_regions,)
    #    where labels[i] = integer index of network for region i
    # --------------------------------------------------
    n_regions = region_df.shape[0]
    labels = np.empty(n_regions, dtype=int)

    for region_idx, net_name in region_to_network.items():
        # find which index this network has in your `networks` list
        labels[region_idx] = networks.index(net_name)

    # Load pre-computed individual results using fMRI IDs
    print("\nLoading individual results using fMRI IDs...")
    results = load_individual_results(paths, fmri_subjects)
    print("Individual analysis results loaded")
    
    # Create a direct mapping from fMRI subject IDs to cluster IDs for organization
    fmri_cluster_mapping = {fmri_id: id_to_cluster[fmri_id] for fmri_id in fmri_subjects if fmri_id in id_to_cluster}
    
    # Organize results by cluster
    print("\nOrganizing results by cluster using fMRI ID mapping...")
    
    cluster_comparisons = organize_by_cluster(results, fmri_cluster_mapping)

    # Initialize the results dictionary
    comparison_results = {}

    # --------------------------------------------------
    # A) build seg/int lists for each cluster
    # --------------------------------------------------
    segs = {}
    ints = {}
    for cid, subs in cluster_comparisons['network'].items():
        nets = [d['network_connectome'] for d in subs.values()]
        segs[cid], ints[cid] = zip(
            *[compute_summary_from_network_matrix(net) for net in nets]
        )
        # convert tuples → lists
        segs[cid], ints[cid] = list(segs[cid]), list(ints[cid])

    # --------------------------------------------------
    # B) omnibus across *all* clusters
    # --------------------------------------------------
    all_seg = [segs[c] for c in sorted(segs)]
    all_int = [ints[c] for c in sorted(ints)]

    # choose parametric vs nonparametric
    if any(len(g)<8 for g in all_seg):
        stat_seg, p_seg = kruskal(*all_seg); test_seg='kruskal'
    else:
        stat_seg, p_seg = f_oneway(*all_seg);   test_seg='anova'

    if any(len(g)<8 for g in all_int):
        stat_int, p_int = kruskal(*all_int);   test_int='kruskal'
    else:
        stat_int, p_int = f_oneway(*all_int);   test_int='anova'

    comparison_results['omnibus'] = {
        'segregation': {'test':test_seg, 'stat':stat_seg, 'p':p_seg},
        'integration': {'test':test_int, 'stat':stat_int, 'p':p_int}
    }

    pairs    = list(itertools.combinations(sorted(segs), 2))
    p_seg, p_int = [], []
    for c1,c2 in pairs:
        _, ps = ttest_ind(segs[c1], segs[c2], equal_var=False);  p_seg.append(ps)
        _, pi = ttest_ind(ints[c1], ints[c2], equal_var=False);  p_int.append(pi)

    # apply FDR
    if args.apply_fdr:
        _, seg_fdr, _, _ = multipletests(p_seg, alpha=args.alpha, method='fdr_bh')
        _, int_fdr, _, _ = multipletests(p_int, alpha=args.alpha, method='fdr_bh')
    else:
        seg_fdr = [None]*len(p_seg);  int_fdr = [None]*len(p_int)

    pairwise = []
    for idx,(c1,c2) in enumerate(pairs):
        pairwise.append({
        'comparison': f"{c1}_vs_{c2}",
        'seg_delta': np.mean(segs[c1]) - np.mean(segs[c2]),
        'seg_p':      p_seg[idx],
        'seg_p_fdr':  seg_fdr[idx],
        'int_delta':  np.mean(ints[c1]) - np.mean(ints[c2]),
        'int_p':      p_int[idx],
        'int_p_fdr':  int_fdr[idx]
        })

    # stash & save CSV
    comparison_results['pairwise_seg_int'] = pairwise
    pd.DataFrame(pairwise).to_csv(
        paths['cluster_comparisons']/ 'seg_int_pairwise.csv', index=False
    )

    # Compare clusters with statistical testing
    print("\nPerforming cluster comparisons with statistical testing...")
    
   # Store the result in a separate variable first
    pairwise_results = compare_clusters(
        cluster_comparisons,
        cluster_pairs,
        atlas_labels=labels,
        alpha=args.alpha,
        apply_fdr=args.apply_fdr
    )

    # Check if we received a dictionary and merge properly
    if isinstance(pairwise_results, dict):
        # Merge the two dictionaries
        comparison_results.update(pairwise_results)
    else:
        # Handle case where it's a list
        comparison_results['pairwise_detailed'] = pairwise_results

    # Save results
    save_comparison_results(comparison_results, paths['cluster_comparisons'])
        
    # Generate summary report of findings
    print("\nGenerating summary report of findings...")
    
    # Create a dedicated stats folder
    stat_dir = paths['cluster_comparisons'] / 'statistics'
    stat_dir.mkdir(exist_ok=True)
    generate_summary_report(comparison_results, stat_dir)
    
    # Generate visualizations if requested
    if args.visualization:
        print("Generating visualizations...")
        # You would need to implement or adapt a visualization function here
        print("Visualization would be implemented here")
    
    print("\nCluster comparison pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    main()
