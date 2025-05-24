'''
- After running 'perform_brain_comparison.py', this script checks the results of the brain comparison. 
- Analysis of brain connectivity comparison results with focus on integration/segregation metrics
'''
# %%
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

# Set paths (modify these as needed)
STAT_DIR = Path('/home/data/2025_asd_multimodal/feature_brain/comparisons')

# Import atlas mapping if available (commented out - uncomment if needed)
# from src.utils_brain_mapping import get_atlas_mapping
# from src.utils_config import ATLAS_PATH
# region_to_network, region_df, networks, network_to_regions = get_atlas_mapping(ATLAS_PATH)
# region_df = region_df.set_index('Region_Number')

def load_summary_metrics():
    """Load and analyze the summary metrics CSV."""
    csv_path = STAT_DIR / 'feature_summary.csv'
    if not csv_path.exists():
        print(f"⚠️  Missing summary CSV at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print("Summary CSV columns:", df.columns.tolist())
    print(f"Total comparisons: {df.shape[0]}")
    
    # Focus on segregation and integration
    seg_int_df = df[df['feature'].isin(['segregation', 'integration'])]
    print("\n=== Segregation and Integration Metrics ===")
    print(seg_int_df)
    
    # Significant findings
    sig = df[df.get('p_fdr', pd.Series([])) < 0.05]
    print(f"\n=== Significant Summary Metrics (p_fdr < 0.05): {len(sig)} findings ===")
    if not sig.empty:
        print(sig)
    else:
        print("  <none>")
    
    # Significant segregation/integration findings
    sig_seg_int = seg_int_df[seg_int_df.get('p_fdr', pd.Series([])) < 0.05]
    print(f"\n=== Significant Segregation/Integration Metrics (p_fdr < 0.05): {len(sig_seg_int)} findings ===")
    if not sig_seg_int.empty:
        print(sig_seg_int)
    else:
        print("  <none>")
    
    return df

def analyze_network_comparisons():
    """Analyze the network comparison results with detailed network information."""
    comp_folders = [d for d in STAT_DIR.iterdir()
                   if d.is_dir() and re.match(r'^\d+_vs_\d+$', d.name)]
    
    # Try to import network names if available
    networks = None
    try:
        from src.utils_brain_mapping import get_atlas_mapping
        from src.utils_config import ATLAS_PATH
        region_to_network, region_df, networks, network_to_regions = get_atlas_mapping(ATLAS_PATH)
    except (ImportError, FileNotFoundError) as e:
        print(f"Note: Could not import network names: {e}")
        print("Will display network indices instead of names.")
    
    results = {}
    print("\n=== Network Comparisons ===")
    for d in comp_folders:
        pkl = d / "network_comparison.pkl"
        if not pkl.exists():
            print(f"⚠️  {d.name} missing network_comparison.pkl → skipping")
            continue
        
        xs = pickle.load(open(pkl, 'rb'))
        results[d.name] = xs
        
        pmat, tmat = xs['pvalues'], xs['tvalues']
        mask = np.triu(pmat < 0.05, k=1)
        ii, jj = np.where(mask)
        sig_count = np.sum(mask)
        
        print(f"\n{d.name}: {sig_count} significant connections (p < 0.05, uncorrected)")
        
        # Show specific network connections if networks are available
        if ii.size:
            print(f"  Significant network connections:")
            for i, j in zip(ii, jj):
                if networks is not None:
                    net1 = networks[i]
                    net2 = networks[j]
                else:
                    # Use indices if network names aren't available
                    net1 = f"Network #{i}"
                    net2 = f"Network #{j}"
                
                print(f"    {net1:30s} ↔ {net2:30s}  t={tmat[i,j]:.2f}, p={pmat[i,j]:.4f}")
    
    return results

def analyze_graph_metrics():
    """Analyze global and nodal graph metrics."""    
    # First define comp_folders
    comp_folders = [d for d in STAT_DIR.iterdir()
                   if d.is_dir() and re.match(r'^\d+_vs_\d+$', d.name)]
    
    # # Then use it for the sample structure analysis
    # sample_folder = comp_folders[0] if comp_folders else None
    # if sample_folder:
    #     sample_path = sample_folder / "graph_comparison.pkl"
    #     if sample_path.exists():
    #         sample_data = pickle.load(open(sample_path, 'rb'))
    #         print("Sample structure of graph_comparison.pkl:")
    #         print(json.dumps({k: "..." for k in sample_data.keys()}, indent=2))
    #         if 'global' in sample_data:
    #             metric_sample = next(iter(sample_data['global'].values()), {})
    #             print("Sample metric structure:")
    #             print(json.dumps({k: "..." for k in metric_sample.keys()}, indent=2))
    
    graph_stats = {}
    print("\n=== Graph Metrics Analysis ===")
    for d in comp_folders:
        pkl = d / "graph_comparison.pkl"
        if not pkl.exists():
            print(f"⚠️  {d.name} missing graph_comparison.pkl → skipping")
            continue
        
        g = pickle.load(open(pkl, 'rb'))
        graph_stats[d.name] = g
        
        # Global metrics
        global_hits = [(m, res) for m, res in g.get('global', {}).items() if res['p'] < 0.05]
        print(f"\n{d.name} - Global metrics (p < 0.05): {len(global_hits)} significant findings")
        if global_hits:
            for m, res in global_hits:
                print(f"  {m:25s}  t={res['t']:.2f}, p={res['p']:.4f}, d={res['effect_size_d']:.2f}")
                
        # Focus on specific integration/segregation related metrics
        integration_metrics = [m for m in g.get('global', {}).keys() 
                              if 'path' in m.lower() or 'efficiency' in m.lower()]
        segregation_metrics = [m for m in g.get('global', {}).keys() 
                              if 'clustering' in m.lower() or 'modularity' in m.lower()]
        
        if integration_metrics:
            print(f"\n  Integration-related metrics:")
            for m in integration_metrics:
                res = g['global'][m]
                print(f"    {m:25s}  t={res['t']:.2f}, p={res['p']:.4f}, d={res['effect_size_d']:.2f}")
                
        if segregation_metrics:
            print(f"\n  Segregation-related metrics:")
            for m in segregation_metrics:
                res = g['global'][m]
                print(f"    {m:25s}  t={res['t']:.2f}, p={res['p']:.4f}, d={res['effect_size_d']:.2f}")
    
    return graph_stats

def compare_traditional_vs_graph_theory():
    """Compare traditional segregation/integration indices with graph-theory based metrics."""
    # Load summary metrics
    csv_path = STAT_DIR / 'feature_summary.csv'
    if not csv_path.exists():
        print("Cannot compare metrics: missing summary CSV")
        return
    
    summary_df = pd.read_csv(csv_path)
    
    # Filter for traditional metrics
    trad_metrics = summary_df[summary_df['feature'].isin(['segregation', 'integration'])]
    
    # Load graph metrics for comparison
    comp_folders = [d for d in STAT_DIR.iterdir()
                   if d.is_dir() and re.match(r'^\d+_vs_\d+$', d.name)]
    
    print("\n=== Traditional vs Graph Theory Metrics ===")
    for d in comp_folders:
        pkl = d / "graph_comparison.pkl"
        if not pkl.exists():
            continue
        
        g = pickle.load(open(pkl, 'rb'))
        
        # Get traditional metrics for this comparison
        comp_trad = trad_metrics[trad_metrics['comparison'] == d.name]
        if comp_trad.empty:
            print(f"No traditional metrics for {d.name}")
            continue

        print(f"\n{d.name}:")
        print("  Traditional metrics:")
        for _, row in comp_trad.iterrows():
            tstat = row.get('tstat', 'N/A')
            if isinstance(tstat, float):
                tstat_str = f"{tstat:.2f}"
            else:
                tstat_str = str(tstat)
            print(f"    {row['feature']:15s}  t={tstat_str:>8}, p={row['pvalue']:.4f}")
        
        # Integration-related graph theory metrics
        integration_metrics = [m for m in g.get('global', {}).keys() 
                              if 'path' in m.lower() or 'efficiency' in m.lower()]
        if integration_metrics:
            print("  Graph theory integration metrics:")
            for m in integration_metrics:
                res = g['global'][m]
                t_val = res.get('t', 'N/A')
                p_val = res.get('p', 'N/A')
                if isinstance(t_val, (float, int)):
                    t_str = f"{t_val:.2f}"
                else:
                    t_str = str(t_val)
                if isinstance(p_val, (float, int)):
                    p_str = f"{p_val:.4f}"
                else:
                    p_str = str(p_val)
                print(f"    {m:25s}  t={t_str}, p={p_str}")
                
        # Segregation-related graph theory metrics
        segregation_metrics = [m for m in g.get('global', {}).keys() 
                              if 'clustering' in m.lower() or 'modularity' in m.lower()]
        if segregation_metrics:
            print("  Graph theory segregation metrics:")
            for m in segregation_metrics:
                res = g['global'][m]
                t_val = res.get('t', 'N/A')
                p_val = res.get('p', 'N/A')
                if isinstance(t_val, (float, int)):
                    t_str = f"{t_val:.2f}"
                else:
                    t_str = str(t_val)
                if isinstance(p_val, (float, int)):
                    p_str = f"{p_val:.4f}"
                else:
                    p_str = str(p_val)
                print(f"    {m:25s}  t={t_str}, p={p_str}")

def identify_best_metrics():
    """Identify which metrics best differentiate clusters."""
    # Load summary metrics
    csv_path = STAT_DIR / 'feature_summary.csv'
    if not csv_path.exists():
        print("Cannot identify best metrics: missing summary CSV")
        return
    
    df = pd.read_csv(csv_path)
    
    # Add columns for significance
    df['significant'] = df.get('p_fdr', df['pvalue']) < 0.05
    df['abs_effect'] = df['difference'].abs()
    
    # Group by feature and calculate how often it's significant
    feature_stats = df.groupby('feature').agg(
        sig_count=('significant', 'sum'),
        comparison_count=('comparison', 'count'),
        mean_effect=('abs_effect', 'mean'),
        max_effect=('abs_effect', 'max')
    )
    feature_stats['sig_rate'] = feature_stats['sig_count'] / feature_stats['comparison_count']
    
    print("\n=== Metrics Ranked by Significance Rate ===")
    print(feature_stats.sort_values('sig_rate', ascending=False))
    
    # Top discriminating features
    top_features = feature_stats.sort_values('sig_rate', ascending=False).head(5)
    print("\n=== Top 5 Discriminating Features ===")
    print(top_features)
    
    return feature_stats

def identify_cluster_patterns():
    """Determine if any clusters show distinctive patterns of segregation or integration."""
    # Load summary metrics
    csv_path = STAT_DIR / 'feature_summary.csv'
    if not csv_path.exists():
        print("Cannot identify cluster patterns: missing summary CSV")
        return
    
    df = pd.read_csv(csv_path)
    seg_int_df = df[df['feature'].isin(['segregation', 'integration'])]
    
    # Extract cluster numbers from comparison names
    cluster_patterns = {}
    
    for _, row in seg_int_df.iterrows():
        comp = row['comparison']
        match = re.match(r'(\d+)_vs_(\d+)', comp)
        if not match:
            continue
            
        c1, c2 = match.groups()
        feature = row['feature']
        difference = row['difference']
        significant = row.get('p_fdr', row['pvalue']) < 0.05
        
        # Record which cluster had higher values
        if difference > 0:  # cluster1 > cluster2
            higher_cluster, lower_cluster = c1, c2
        else:  # cluster2 > cluster1
            higher_cluster, lower_cluster = c2, c1
            
        # Initialize if not exists
        if higher_cluster not in cluster_patterns:
            cluster_patterns[higher_cluster] = {'higher': {}, 'lower': {}}
        if lower_cluster not in cluster_patterns:
            cluster_patterns[lower_cluster] = {'higher': {}, 'lower': {}}
            
        # Record pattern
        if feature not in cluster_patterns[higher_cluster]['higher']:
            cluster_patterns[higher_cluster]['higher'][feature] = 0
        cluster_patterns[higher_cluster]['higher'][feature] += 1
        
        if feature not in cluster_patterns[lower_cluster]['lower']:
            cluster_patterns[lower_cluster]['lower'][feature] = 0
        cluster_patterns[lower_cluster]['lower'][feature] += 1
    
    print("\n=== Cluster Patterns of Segregation/Integration ===")
    for cluster, patterns in cluster_patterns.items():
        print(f"\nCluster {cluster}:")
        print("  Higher than other clusters in:")
        for feature, count in patterns['higher'].items():
            total = patterns['higher'].get(feature, 0) + patterns['lower'].get(feature, 0)
            print(f"    {feature}: {count}/{total} comparisons")
            
        print("  Lower than other clusters in:")
        for feature, count in patterns['lower'].items():
            total = patterns['higher'].get(feature, 0) + patterns['lower'].get(feature, 0)
            print(f"    {feature}: {count}/{total} comparisons")
    
    return cluster_patterns

def identify_top_discriminative_metrics():
    """
    Analyze all graph theory metrics across all comparisons to identify
    the metrics with the greatest discriminative power based on effect size.
    Returns separate rankings for integration and segregation metrics.
    """
    comp_folders = [d for d in STAT_DIR.iterdir() 
                   if d.is_dir() and re.match(r'^\d+_vs_\d+$', d.name)]
    
    # Collect all metrics and their effect sizes
    all_metrics = []
    
    for d in comp_folders:
        pkl = d / "graph_comparison.pkl"
        if not pkl.exists():
            continue
        
        g = pickle.load(open(pkl, 'rb'))
        
        # Process global metrics
        for metric, res in g.get('global', {}).items():
            effect_size = res.get('effect_size_d')
            if effect_size is not None:
                all_metrics.append({
                    'comparison': d.name,
                    'metric': metric,
                    'effect_size': abs(effect_size),  # Use absolute value for ranking
                    'effect_size_raw': effect_size,   # Keep original for direction
                    'p_value': res.get('p'),
                    'c1_mean': res.get('c1_mean'),
                    'c2_mean': res.get('c2_mean'),
                    'significant': res.get('p', 1.0) < 0.05
                })
    
    # Convert to DataFrame for easier analysis
    metrics_df = pd.DataFrame(all_metrics)
    
    # Group by metric and calculate average effect size
    metric_summary = metrics_df.groupby('metric').agg(
        mean_effect=('effect_size', 'mean'),
        max_effect=('effect_size', 'max'),
        significant_count=('significant', 'sum'),
        comparison_count=('comparison', 'count')
    )
    metric_summary['significant_rate'] = metric_summary['significant_count'] / metric_summary['comparison_count']
    
    # Sort by mean effect size
    ranked_metrics = metric_summary.sort_values('mean_effect', ascending=False)
    
    print("\n=== Graph Theory Metrics Ranked by Effect Size ===")
    print(ranked_metrics)
    
    # Classify metrics as integration or segregation related
    integration_keywords = ['efficiency', 'path', 'characteristic', 'distance']
    segregation_keywords = ['modularity', 'clustering', 'community', 'transitivity']
    
    integration_metrics = []
    segregation_metrics = []
    
    for metric in ranked_metrics.index:
        if any(kw in metric.lower() for kw in integration_keywords):
            integration_metrics.append(metric)
        elif any(kw in metric.lower() for kw in segregation_keywords):
            segregation_metrics.append(metric)
    
    # Get top metrics for each category
    top_integration = ranked_metrics.loc[integration_metrics].sort_values('mean_effect', ascending=False)
    top_segregation = ranked_metrics.loc[segregation_metrics].sort_values('mean_effect', ascending=False)
    
    print("\n=== Top Integration Metrics ===")
    print(top_integration)
    
    print("\n=== Top Segregation Metrics ===")
    print(top_segregation)
    
    # Detailed look at top metrics
    if not top_integration.empty:
        top_int_metric = top_integration.index[0]
        print(f"\nDetails of top integration metric: {top_int_metric}")
        metric_details = metrics_df[metrics_df['metric'] == top_int_metric]
        for _, row in metric_details.iterrows():
            c1, c2 = row['comparison'].split('_vs_')
            print(f"  {row['comparison']}: effect_size={row['effect_size']:.2f}, p={row['p_value']:.4f}, significant: {row['significant']}")
    
    if not top_segregation.empty:
        top_seg_metric = top_segregation.index[0]
        print(f"\nDetails of top segregation metric: {top_seg_metric}")
        metric_details = metrics_df[metrics_df['metric'] == top_seg_metric]
        for _, row in metric_details.iterrows():
            c1, c2 = row['comparison'].split('_vs_')
            print(f"  {row['comparison']}: effect_size={row['effect_size']:.2f}, p={row['p_value']:.4f}, significant: {row['significant']}")
    
    return {
        'all_ranked': ranked_metrics,
        'top_integration': top_integration,
        'top_segregation': top_segregation,
        'metrics_df': metrics_df
    }

def analyze_modularity_in_detail():
    """
    Perform a detailed analysis of modularity across clusters,
    as this is likely to be one of the top segregation metrics.
    """
    comp_folders = [d for d in STAT_DIR.iterdir() 
                   if d.is_dir() and re.match(r'^\d+_vs_\d+$', d.name)]
    
    modularity_results = []
    
    for d in comp_folders:
        match = re.match(r'(\d+)_vs_(\d+)', d.name)
        if not match:
            continue
            
        c1, c2 = match.groups()
        
        # Load graph comparison results
        pkl = d / "graph_comparison.pkl"
        if not pkl.exists():
            continue
        
        g = pickle.load(open(pkl, 'rb'))
        
        # Look for modularity metric
        modularity_metrics = {k: v for k, v in g.get('global', {}).items() 
                              if 'modul' in k.lower()}
        
        for metric, res in modularity_metrics.items():
            modularity_results.append({
                'comparison': d.name,
                'c1': c1,
                'c2': c2,
                'metric': metric,
                'effect_size': res.get('effect_size_d'),
                'p_value': res.get('p'),
                'c1_mean': res.get('c1_mean'),
                'c2_mean': res.get('c2_mean'),
                'c1_std': res.get('c1_std'),
                'c2_std': res.get('c2_std'),
                'significant': res.get('p', 1.0) < 0.05
            })
    
    # Convert to DataFrame
    modul_df = pd.DataFrame(modularity_results)
    
    if modul_df.empty:
        print("\nNo modularity metrics found in the data")
        return None
    
    print("\n=== Modularity Analysis Across Comparisons ===")
    print(modul_df[['comparison', 'metric', 'effect_size', 'p_value', 'significant']])
    
    # Identify which clusters have higher/lower modularity
    cluster_modularity = defaultdict(list)
    
    for _, row in modul_df.iterrows():
        c1, c2 = row['c1'], row['c2']
        c1_mean, c2_mean = row['c1_mean'], row['c2_mean']
        
        if c1_mean > c2_mean:
            cluster_modularity[c1].append(('higher', c2))
            cluster_modularity[c2].append(('lower', c1))
        else:
            cluster_modularity[c1].append(('lower', c2))
            cluster_modularity[c2].append(('higher', c1))
    
    print("\n=== Modularity Patterns by Cluster ===")
    for cluster, comparisons in cluster_modularity.items():
        higher = [c for direction, c in comparisons if direction == 'higher']
        lower = [c for direction, c in comparisons if direction == 'lower']
        
        higher_ratio = len(higher) / len(comparisons) if comparisons else 0
        lower_ratio = len(lower) / len(comparisons) if comparisons else 0
        
        status = "higher" if higher_ratio > 0.75 else "lower" if lower_ratio > 0.75 else "mixed"
        
        print(f"Cluster {cluster}: Modularity generally {status}")
        print(f"  Higher than clusters: {higher}")
        print(f"  Lower than clusters: {lower}")
    
    return modul_df

def analyze_cluster4_distinctive_metrics():
    """
    Focus specifically on Cluster 4's distinctive metrics,
    particularly the top integration and segregation metrics.
    """
    # Get all comparisons involving Cluster 4
    comp_folders = [d for d in STAT_DIR.iterdir() 
                   if d.is_dir() and re.match(r'^\d+_vs_4$|^4_vs_\d+$', d.name)]
    
    metrics_results = []
    
    for d in comp_folders:
        match = re.match(r'(\d+)_vs_(\d+)', d.name)
        if not match:
            continue
            
        c1, c2 = match.groups()
        is_cluster4_first = (c1 == '4')
        other_cluster = c2 if is_cluster4_first else c1
        
        # Load graph comparison results
        pkl = d / "graph_comparison.pkl"
        if not pkl.exists():
            continue
        
        g = pickle.load(open(pkl, 'rb'))
        
        # Process all global metrics
        for metric, res in g.get('global', {}).items():
            if res.get('effect_size_d') is None:
                continue
                
            # Determine whether Cluster 4 has higher or lower values
            c4_mean = res['c1_mean'] if is_cluster4_first else res['c2_mean']
            other_mean = res['c2_mean'] if is_cluster4_first else res['c1_mean']
            
            # The sign convention means we need to check the effect direction carefully
            effect_size = res['effect_size_d']
            if is_cluster4_first:
                # If Cluster 4 is first, positive effect means it's higher
                c4_higher = (effect_size > 0)
            else:
                # If Cluster 4 is second, negative effect means it's higher
                c4_higher = (effect_size < 0)
            
            metrics_results.append({
                'comparison': d.name,
                'other_cluster': other_cluster,
                'metric': metric,
                'effect_size': abs(effect_size),
                'effect_size_raw': effect_size,
                'p_value': res.get('p'),
                'c4_mean': c4_mean,
                'other_mean': other_mean,
                'c4_higher': c4_higher,
                'significant': res.get('p', 1.0) < 0.05
            })
    
    # Convert to DataFrame
    cluster4_df = pd.DataFrame(metrics_results)
    
    if cluster4_df.empty:
        print("\nNo metrics found for comparisons involving Cluster 4")
        return None
    
    # Classify metrics as integration or segregation related
    integration_keywords = ['efficiency', 'path', 'characteristic', 'distance']
    segregation_keywords = ['modularity', 'clustering', 'community', 'transitivity']
    
    def classify_metric(metric_name):
        if any(kw in metric_name.lower() for kw in integration_keywords):
            return "Integration"
        elif any(kw in metric_name.lower() for kw in segregation_keywords):
            return "Segregation"
        else:
            return "Other"
    
    cluster4_df['category'] = cluster4_df['metric'].apply(classify_metric)
    
    # Print summary of Cluster 4's metrics
    print("\n=== Cluster 4's Distinctive Metrics ===")
    for category in ["Integration", "Segregation"]:
        cat_df = cluster4_df[cluster4_df['category'] == category]
        if cat_df.empty:
            continue
            
        print(f"\n{category} Metrics:")
        
        # Group by metric and determine consistency
        for metric in cat_df['metric'].unique():
            metric_rows = cat_df[cat_df['metric'] == metric]
            higher_count = metric_rows['c4_higher'].sum()
            total = len(metric_rows)
            higher_pct = (higher_count / total) * 100
            
            direction = "HIGHER" if higher_pct > 75 else "LOWER" if higher_pct < 25 else "MIXED"
            
            sig_count = metric_rows['significant'].sum()
            sig_pct = (sig_count / total) * 100
            
            print(f"  {metric}: Cluster 4 is consistently {direction} ({higher_pct:.0f}% of comparisons)")
            print(f"    Significant in {sig_pct:.0f}% of comparisons")
            print(f"    Effect size: mean={metric_rows['effect_size'].mean():.2f}, max={metric_rows['effect_size'].max():.2f}")
    
    return cluster4_df

def compare_integration_segregation_by_cluster():
    """
    Compare the top integration and segregation metrics across clusters
    to establish a clear profile for each cluster.
    """
    # First get top metrics
    top_metrics = identify_top_discriminative_metrics()
    
    if (top_metrics['top_integration'].empty or 
        top_metrics['top_segregation'].empty):
        print("Cannot identify top metrics for both categories")
        return
    
    # Get the top metric for each category
    top_int_metric = top_metrics['top_integration'].index[0]
    top_seg_metric = top_metrics['top_segregation'].index[0]
    
    print(f"\n=== Comparing Top Metrics Across Clusters ===")
    print(f"Top Integration Metric: {top_int_metric}")
    print(f"Top Segregation Metric: {top_seg_metric}")
    
    # Collect values for each cluster from pairwise comparisons
    metrics_df = top_metrics['metrics_df']
    int_df = metrics_df[metrics_df['metric'] == top_int_metric]
    seg_df = metrics_df[metrics_df['metric'] == top_seg_metric]
    
    # Extract cluster-specific values
    cluster_values = defaultdict(lambda: {'integration': [], 'segregation': []})
    
    # Process integration metric
    for _, row in int_df.iterrows():
        c1, c2 = row['comparison'].split('_vs_')
        cluster_values[c1]['integration'].append(row['c1_mean'])
        cluster_values[c2]['integration'].append(row['c2_mean'])
    
    # Process segregation metric
    for _, row in seg_df.iterrows():
        c1, c2 = row['comparison'].split('_vs_')
        cluster_values[c1]['segregation'].append(row['c1_mean'])
        cluster_values[c2]['segregation'].append(row['c2_mean'])
    
    # Calculate average values for each cluster
    cluster_averages = {}
    for cluster, values in cluster_values.items():
        int_values = values['integration']
        seg_values = values['segregation']
        
        if int_values and seg_values:
            cluster_averages[cluster] = {
                'integration_mean': np.mean(int_values),
                'integration_std': np.std(int_values) if len(int_values) > 1 else 0,
                'segregation_mean': np.mean(seg_values),
                'segregation_std': np.std(seg_values) if len(seg_values) > 1 else 0
            }
    
    # Print cluster profiles
    print("\n=== Cluster Profiles Based on Top Metrics ===")
    for cluster, values in sorted(cluster_averages.items()):
        print(f"\nCluster {cluster}:")
        print(f"  {top_int_metric}: {values['integration_mean']:.3f} ± {values['integration_std']:.3f}")
        print(f"  {top_seg_metric}: {values['segregation_mean']:.3f} ± {values['segregation_std']:.3f}")
    
    # Characterize each cluster based on high/low values
    # First, calculate overall means for each metric
    all_int = [v['integration_mean'] for v in cluster_averages.values()]
    all_seg = [v['segregation_mean'] for v in cluster_averages.values()]
    
    int_mean = np.mean(all_int)
    seg_mean = np.mean(all_seg)
    
    print("\n=== Cluster Characterization ===")
    for cluster, values in sorted(cluster_averages.items()):
        int_status = "high" if values['integration_mean'] > int_mean else "low"
        seg_status = "high" if values['segregation_mean'] > seg_mean else "low"
        
        profile = f"{int_status} integration, {seg_status} segregation"
        print(f"Cluster {cluster}: {profile}")
    
    return cluster_averages


#%%
if __name__ == "__main__":
    print("=== Brain Connectivity Analysis ===")
    
    # 1. Analyze summary metrics
    summary_df = load_summary_metrics()
    
    # 2. Analyze network comparisons
    network_results = analyze_network_comparisons()
    
    # 3. Analyze graph metrics
    graph_results = analyze_graph_metrics()
    
    # 4. Compare traditional vs graph theory metrics
    compare_traditional_vs_graph_theory()
    
    # 5. Identify best metrics
    feature_stats = identify_best_metrics()
    
    # 6. Identify cluster patterns
    cluster_patterns = identify_cluster_patterns()

    # Additional analysis to determine best metrics
    # 7. Identify top discriminative metrics
    top_metrics = identify_top_discriminative_metrics()
    
    # 8. Analyze modularity in detail
    modularity_df = analyze_modularity_in_detail()
    
    # 9. Analyze cluster 4's distinctive metrics
    cluster4_df = analyze_cluster4_distinctive_metrics()
    
    # 10. Compare integration/segregation by cluster
    cluster_averages = compare_integration_segregation_by_cluster()


#%%
'''
old code:
import re
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# atlas → network names + coords
from src.utils_brain_mapping import get_atlas_mapping
from src.utils_config import ATLAS_PATH

# ─── 0) Paths ────────────────────────────────────────────────────────────────
STAT_DIR = Path('/home/data/2025_asd_multimodal/feature_brain/comparisons')

# ─── 1) Atlas Mapping ────────────────────────────────────────────────────────
region_to_network, region_df, networks, network_to_regions = get_atlas_mapping(ATLAS_PATH)
region_df = region_df.set_index('Region_Number')

#%%
# ─── 2) Summary Metrics CSV ──────────────────────────────────────────────────
csv_path = STAT_DIR / 'feature_summary.csv'
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print("Summary CSV columns:", df.columns.tolist())

    sig = df[df.get('p_fdr', pd.Series([])) < 0.05]
    print("\n=== Significant Summary Metrics (p_fdr < 0.05) ===")
    print(sig if not sig.empty else "  <none>")
else:
    print(f"⚠️  Missing summary CSV at {csv_path}")

# Show just your two summary metrics:
print(df[df['feature'].isin(['segregation', 'integration'])])

#%%
print(df[['comparison','feature','cluster1_mean','cluster2_mean','difference','pvalue']])


#%%
# ─── 3) Edge‑wise Network Comparisons ────────────────────────────────────────
comp_folders = [d for d in STAT_DIR.iterdir()
                if d.is_dir() and re.match(r'^\d+_vs_\d+$', d.name)]

print("\n--- Uncorrected p<.05 edges (13×13 Power networks) ---")
for d in comp_folders:
    pkl = d / "network_comparison.pkl"
    if not pkl.exists():
        print(f"⚠️  {d.name} missing network_comparison.pkl → skipping")
        continue

    xs = pickle.load(pkl.open('rb'))
    pmat, tmat = xs['pvalues'], xs['tvalues']
    mask = np.triu(pmat < 0.05, k=1)
    ii, jj = np.where(mask)

    if ii.size:
        print(f"\nEdges p<.05 in {d.name}:")
        for i, j in zip(ii, jj):
            print(f"  {networks[i]:30s} ↔ {networks[j]:30s}  "
                  f"t={tmat[i,j]:.2f}, p={pmat[i,j]:.4f}")
    else:
        print(f"\nNo uncorrected p<.05 edges in {d.name}")


#%%
# ─── 4) Global Graph Metrics ─────────────────────────────────────────────────
print("\n--- Uncorrected p<.05 global graph metrics ---")
graph_stats = {}
for d in comp_folders:
    pkl = d / "graph_comparison.pkl"
    if not pkl.exists():
        print(f"⚠️  {d.name} missing graph_comparison.pkl → skipping")
        continue

    g = pickle.load(pkl.open('rb'))
    graph_stats[d.name] = g

    hits = [(m, res) for m, res in g.get('global', {}).items() if res['p'] < 0.05]
    print(f"\nGlobal metrics in {d.name}:")
    if hits:
        for m, res in hits:
            print(f"  {m:25s}  t={res['t']:.2f}, p={res['p']:.4f}, d={res['effect_size']:.2f}")
    else:
        print("  <none>")


#%%
# ─── 5) Nodal Metrics (Annotated) ────────────────────────────────────────────
print("\n--- Annotated Nodal Metrics ---")
for comp, g in graph_stats.items():
    print(f"\n+++ {comp} +++")
    nodal = g.get('nodal', {})
    if not nodal:
        print(" (no nodal metrics passed shape check)")
        continue

    for metric, out in nodal.items():
        pvals = np.array(out['pvalues'])
        pfdr  = np.array(out.get('p_fdr') or [])

        hits_unc = np.where(pvals < 0.05)[0]
        hits_fdr = np.where(pfdr < 0.05)[0] if pfdr.size else []

        if not hits_unc.size and not hits_fdr.size:
            continue

        print(f"\nMetric: {metric}")
        if hits_unc.size:
            print(" Uncorrected p<.05:")
            for idx in hits_unc:
                reg = region_df.loc[idx+1]
                print(f"   • Node {idx:3d} (Power #{idx+1:3d}) | "
                      f"{reg.Network:30s} | MNI=({reg.MNI_X:.1f},{reg.MNI_Y:.1f},{reg.MNI_Z:.1f}) | "
                      f"p={pvals[idx]:.4f}")
        if hits_fdr.size:
            print(" FDR‑corrected p<.05:")
            for idx in hits_fdr:
                reg = region_df.loc[idx+1]
                print(f"   • Node {idx:3d} (Power #{idx+1:3d}) | "
                      f"{reg.Network:30s} | MNI=({reg.MNI_X:.1f},{reg.MNI_Y:.1f},{reg.MNI_Z:.1f}) | "
                      f"p_fdr={pfdr[idx]:.4f}")

# %%

'''