"""
Statistical analysis utilities for genetic variant cluster analysis.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings('ignore')

def perform_one_vs_all_analysis(long_df, variant_types=None, categories=None):
    """
    Perform one-vs-all analysis for each cluster against all other clusters.
    
    Args:
        long_df (pd.DataFrame): Long-format variant data
        variant_types (list): List of variant types to analyze (e.g., ['rare_coding', 'common_coding'])
        categories (list): List of variant categories to analyze (e.g., ['PTV', 'Missense'])
        
    Returns:
        pd.DataFrame: Results of statistical tests
    """
    # Default all variant types and categories if none specified
    if variant_types is None:
        variant_types = long_df['variant_type'].unique()
    
    if categories is None:
        categories = long_df['category'].unique()
    
    # Get list of all clusters
    clusters = sorted(long_df['cluster'].unique())
    
    # Initialize results list
    results = []
    
    # Get participant counts by cluster
    participant_counts = long_df.groupby('cluster')['participant_id'].nunique()
    
    # Process each variant type
    for variant_type in variant_types:
        # Filter for this variant type
        var_df = long_df[long_df['variant_type'] == variant_type]
        
        # Skip if no data for this variant type
        if var_df.empty:
            continue
        
        # Process each category
        for category in categories:
            # Filter for this category
            cat_df = var_df[var_df['category'] == category]
            
            # Skip if no data for this category
            if cat_df.empty:
                continue
            
            # Process each impact bin
            for impact in cat_df['impact_score_bin'].unique():
                # Filter for this impact
                impact_df = cat_df[cat_df['impact_score_bin'] == impact]
                
                # Skip if no data for this impact
                if impact_df.empty:
                    continue
                
                # Focus on aggregated data (non-gene-specific)
                agg_df = impact_df[impact_df['gene_name'] == '--']
                
                # Process each cluster (one vs all)
                for cluster in clusters:
                    # Get data for this cluster
                    cluster_df = agg_df[agg_df['cluster'] == cluster]
                    other_df = agg_df[agg_df['cluster'] != cluster]
                    
                    # Skip if no data in either group
                    if cluster_df.empty or other_df.empty:
                        continue
                    
                    # Calculate per-participant rates
                    cluster_sum = cluster_df['count'].sum()
                    other_sum = other_df['count'].sum()
                    
                    # Get participant counts
                    cluster_participants = participant_counts.get(cluster, 0)
                    other_participants = participant_counts.sum() - cluster_participants
                    
                    # Calculate rates
                    cluster_rate = cluster_sum / cluster_participants if cluster_participants > 0 else 0
                    other_rate = other_sum / other_participants if other_participants > 0 else 0
                    
                    # Calculate log2 fold change
                    log2fc = np.log2(cluster_rate / other_rate) if cluster_rate > 0 and other_rate > 0 else np.nan
                    
                    # Perform Fisher's exact test
                    # FIX: Ensure values in contingency table are non-negative
                    # Instead of assuming 100 variants max per participant, use actual variant counts
                    # and ensure the "non-variant" count is positive
                    non_cluster_variants = max(0, cluster_participants - cluster_sum)
                    non_other_variants = max(0, other_participants - other_sum)
                    
                    contingency = np.array([
                        [cluster_sum, non_cluster_variants],
                        [other_sum, non_other_variants]
                    ])
                    
                    # Only perform test if all values are non-negative
                    if np.all(contingency >= 0):
                        oddsratio, pvalue = stats.fisher_exact(contingency)
                    else:
                        # If we can't perform the test, skip this comparison
                        print(f"Warning: Negative values in contingency table for {variant_type} {category} {impact}, cluster {cluster}")
                        continue
                    
                    # Also calculate for gene counts
                    cluster_genes = impact_df[(impact_df['cluster'] == cluster) & 
                                             (impact_df['gene_name'] != '--')]['gene_name'].nunique()
                    other_genes = impact_df[(impact_df['cluster'] != cluster) & 
                                           (impact_df['gene_name'] != '--')]['gene_name'].nunique()
                    
                    # Calculate gene rates
                    cluster_gene_rate = cluster_genes / cluster_participants if cluster_participants > 0 else 0
                    other_gene_rate = other_genes / other_participants if other_participants > 0 else 0
                    
                    # Calculate gene log2 fold change
                    gene_log2fc = np.log2(cluster_gene_rate / other_gene_rate) if cluster_gene_rate > 0 and other_gene_rate > 0 else np.nan
                    
                    # Add to results
                    results.append({
                        'cluster': cluster,
                        'variant_type': variant_type,
                        'category': category,
                        'impact': impact,
                        'cluster_variants': cluster_sum,
                        'cluster_participants': cluster_participants,
                        'cluster_rate': cluster_rate,
                        'other_variants': other_sum,
                        'other_participants': other_participants,
                        'other_rate': other_rate,
                        'log2fc': log2fc,
                        'oddsratio': oddsratio,
                        'pvalue': pvalue,
                        'cluster_genes': cluster_genes,
                        'other_genes': other_genes,
                        'cluster_gene_rate': cluster_gene_rate,
                        'other_gene_rate': other_gene_rate,
                        'gene_log2fc': gene_log2fc
                    })
    
    # Create dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate multiple testing correction
    if not results_df.empty and 'pvalue' in results_df.columns:
        # Add adjusted p-values (FDR)
        results_df['padj'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
    
    return results_df

def analyze_variant_burden_by_category(long_df, output_path=None):
    """
    Analyze variant burden by category for all clusters.
    
    Args:
        long_df (pd.DataFrame): Long-format variant data
        output_path (str, optional): Path to save results CSV
        
    Returns:
        pd.DataFrame: Summary of variant burden by category
    """
    # Filter for non-gene-specific entries
    agg_df = long_df[long_df['gene_name'] == '--']
    
    # Get participant counts
    participant_counts = long_df.groupby('cluster')['participant_id'].nunique().to_dict()
    
    # Create summary
    results = []
    
    # Group by cluster, variant type, category, and impact
    grouped = agg_df.groupby(['cluster', 'variant_type', 'category', 'impact_score_bin'])
    
    for (cluster, var_type, category, impact), group in grouped:
        # Calculate variant and gene metrics
        variant_count = group['count'].sum()
        variant_rate = variant_count / participant_counts[cluster] if cluster in participant_counts else 0
        
        # Get gene counts for this combination
        gene_df = long_df[
            (long_df['cluster'] == cluster) &
            (long_df['variant_type'] == var_type) &
            (long_df['category'] == category) &
            (long_df['impact_score_bin'] == impact) &
            (long_df['gene_name'] != '--')
        ]
        
        gene_count = gene_df['gene_name'].nunique()
        
        # Add to results
        results.append({
            'cluster': cluster,
            'variant_type': var_type,
            'category': category,
            'impact': impact,
            'participant_count': participant_counts.get(cluster, 0),
            'variant_count': variant_count,
            'variant_rate': variant_rate,
            'gene_count': gene_count
        })
    
    # Create dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by variant type, category, impact, cluster
    results_df = results_df.sort_values(['variant_type', 'category', 'impact', 'cluster'])
    
    # Save if requested
    if output_path:
        results_df.to_csv(output_path, index=False)
    
    return results_df

def get_enriched_genes_by_cluster(long_df, cluster, variant_type=None, category=None, impact=None, min_count=2):
    """
    Get genes that are enriched in a specific cluster compared to others.
    
    Args:
        long_df (pd.DataFrame): Long-format variant data
        cluster (int): Cluster to analyze
        variant_type (str, optional): Filter by variant type
        category (str, optional): Filter by category
        impact (str, optional): Filter by impact
        min_count (int): Minimum number of variants in the cluster
        
    Returns:
        pd.DataFrame: Enriched genes with statistics
    """
    # Filter for gene-specific entries and the specified cluster
    gene_df = long_df[long_df['gene_name'] != '--'].copy()
    
    # Apply filters if specified
    if variant_type:
        gene_df = gene_df[gene_df['variant_type'] == variant_type]
    
    if category:
        gene_df = gene_df[gene_df['category'] == category]
    
    if impact:
        gene_df = gene_df[gene_df['impact_score_bin'] == impact]
    
    # Get participant counts by cluster
    participant_counts = long_df.groupby('cluster')['participant_id'].nunique().to_dict()
    
    # Get unique genes
    all_genes = gene_df['gene_name'].unique()
    
    # Initialize results
    results = []
    
    # Analyze each gene
    for gene in all_genes:
        # Get gene data
        gene_data = gene_df[gene_df['gene_name'] == gene]
        
        # Count variants in this cluster and others
        cluster_variants = gene_data[gene_data['cluster'] == cluster]['count'].sum()
        other_variants = gene_data[gene_data['cluster'] != cluster]['count'].sum()
        
        # Skip if below minimum count in cluster
        if cluster_variants < min_count:
            continue
        
        # Get participant counts
        cluster_participants = participant_counts.get(cluster, 0)
        other_participants = sum(v for k, v in participant_counts.items() if k != cluster)
        
        # Calculate rates
        cluster_rate = cluster_variants / cluster_participants if cluster_participants > 0 else 0
        other_rate = other_variants / other_participants if other_participants > 0 else 0
        
        # Calculate enrichment
        enrichment = cluster_rate / other_rate if other_rate > 0 else np.inf
        log2fc = np.log2(enrichment) if enrichment != np.inf and enrichment > 0 else np.nan
        
        # Perform Fisher's exact test
        # Create contingency table - ensure non-negative values
        non_cluster_variants = max(0, cluster_participants - cluster_variants)
        non_other_variants = max(0, other_participants - other_variants)
        
        contingency = np.array([
            [cluster_variants, non_cluster_variants],
            [other_variants, non_other_variants]
        ])
        
        # Perform test if all values are non-negative
        if np.all(contingency >= 0):
            oddsratio, pvalue = stats.fisher_exact(contingency)
        else:
            print(f"Warning: Negative values in contingency table for gene {gene}, cluster {cluster}")
            continue
        
        # Add to results
        results.append({
            'gene': gene,
            'cluster': cluster,
            'cluster_variants': cluster_variants,
            'other_variants': other_variants,
            'cluster_rate': cluster_rate,
            'other_rate': other_rate,
            'enrichment': enrichment,
            'log2fc': log2fc,
            'oddsratio': oddsratio,
            'pvalue': pvalue
        })
    
    # Create dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate multiple testing correction if we have results
    if not results_df.empty and 'pvalue' in results_df.columns:
        # Add adjusted p-values (FDR)
        results_df['padj'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
        
        # Sort by adjusted p-value
        results_df = results_df.sort_values('padj')
    
    return results_df

def find_cluster_specific_genes(long_df, min_participants=2):
    """
    Find genes that are specific to one cluster and absent or rare in others.
    
    Args:
        long_df (pd.DataFrame): Long-format variant data
        min_participants (int): Minimum number of participants in a cluster with the gene
        
    Returns:
        pd.DataFrame: Cluster-specific genes with statistics
    """
    # Filter for gene-specific entries
    gene_df = long_df[long_df['gene_name'] != '--'].copy()
    
    # Get clusters
    clusters = sorted(gene_df['cluster'].unique())
    
    # Get per-cluster participant counts
    participant_counts = gene_df.groupby('cluster')['participant_id'].nunique().to_dict()
    
    # Find all genes
    all_genes = gene_df['gene_name'].unique()
    
    # Initialize results
    results = []
    
    # Check each gene
    for gene in all_genes:
        # Get data for this gene
        gene_data = gene_df[gene_df['gene_name'] == gene]
        
        # Count participants with this gene in each cluster
        participants_per_cluster = {}
        for cluster in clusters:
            cluster_data = gene_data[gene_data['cluster'] == cluster]
            participants_per_cluster[cluster] = cluster_data['participant_id'].nunique()
        
        # Check if gene is specific to one cluster
        for cluster in clusters:
            # Skip if too few participants in this cluster have the gene
            if participants_per_cluster[cluster] < min_participants:
                continue
            
            # Calculate percentage of cluster participants with this gene
            cluster_percentage = participants_per_cluster[cluster] / participant_counts[cluster] * 100
            
            # Check other clusters
            other_clusters = [c for c in clusters if c != cluster]
            other_total_participants = sum(participant_counts[c] for c in other_clusters)
            other_participants_with_gene = sum(participants_per_cluster[c] for c in other_clusters)
            
            # Calculate percentage in other clusters
            other_percentage = (other_participants_with_gene / other_total_participants * 100 
                               if other_total_participants > 0 else 0)
            
            # Calculate specificity ratio (how many times more common in this cluster)
            specificity = cluster_percentage / other_percentage if other_percentage > 0 else np.inf
            
            # Only include if specificity is high
            if specificity >= 2:  # At least twice as common in this cluster
                # Get variant info
                variant_info = gene_data.groupby(['variant_type', 'category', 'impact_score_bin'])['count'].sum()
                variant_info_str = ', '.join(f"{k[0]}-{k[1]}-{k[2]}: {v}" for k, v in variant_info.items())
                
                # Add to results
                results.append({
                    'gene': gene,
                    'cluster': cluster,
                    'cluster_participants': participants_per_cluster[cluster],
                    'cluster_total': participant_counts[cluster],
                    'cluster_percentage': cluster_percentage,
                    'other_participants': other_participants_with_gene,
                    'other_total': other_total_participants,
                    'other_percentage': other_percentage,
                    'specificity': specificity,
                    'variant_info': variant_info_str
                })
    
    # Create dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by specificity
    if not results_df.empty:
        results_df = results_df.sort_values(['cluster', 'specificity'], ascending=[True, False])
    
    return results_df

def compare_clusters_pairwise(long_df, variant_types=None, categories=None):
    """
    Perform pairwise comparisons between clusters for variant burden.
    
    Args:
        long_df (pd.DataFrame): Long-format variant data
        variant_types (list): List of variant types to analyze
        categories (list): List of variant categories to analyze
        
    Returns:
        pd.DataFrame: Results of pairwise statistical tests
    """
    # Default all variant types and categories if none specified
    if variant_types is None:
        variant_types = long_df['variant_type'].unique()
    
    if categories is None:
        categories = long_df['category'].unique()
    
    # Get list of all clusters
    clusters = sorted(long_df['cluster'].unique())
    
    # Initialize results list
    results = []
    
    # Get participant counts by cluster
    participant_counts = long_df.groupby('cluster')['participant_id'].nunique()
    
    # Process each variant type
    for variant_type in variant_types:
        # Filter for this variant type
        var_df = long_df[long_df['variant_type'] == variant_type]
        
        # Skip if no data for this variant type
        if var_df.empty:
            continue
        
        # Process each category
        for category in categories:
            # Filter for this category
            cat_df = var_df[var_df['category'] == category]
            
            # Skip if no data for this category
            if cat_df.empty:
                continue
            
            # Process each impact bin
            for impact in cat_df['impact_score_bin'].unique():
                # Filter for this impact
                impact_df = cat_df[cat_df['impact_score_bin'] == impact]
                
                # Skip if no data for this impact
                if impact_df.empty:
                    continue
                
                # Focus on aggregated data (non-gene-specific)
                agg_df = impact_df[impact_df['gene_name'] == '--']
                
                # Process each cluster pair
                for i, cluster1 in enumerate(clusters):
                    for cluster2 in clusters[i+1:]:  # Only compare each pair once
                        # Get data for this pair
                        cluster1_df = agg_df[agg_df['cluster'] == cluster1]
                        cluster2_df = agg_df[agg_df['cluster'] == cluster2]
                        
                        # Skip if no data in either group
                        if cluster1_df.empty or cluster2_df.empty:
                            continue
                        
                        # Calculate per-participant rates
                        cluster1_sum = cluster1_df['count'].sum()
                        cluster2_sum = cluster2_df['count'].sum()
                        
                        # Get participant counts
                        cluster1_participants = participant_counts.get(cluster1, 0)
                        cluster2_participants = participant_counts.get(cluster2, 0)
                        
                        # Calculate rates
                        cluster1_rate = cluster1_sum / cluster1_participants if cluster1_participants > 0 else 0
                        cluster2_rate = cluster2_sum / cluster2_participants if cluster2_participants > 0 else 0
                        
                        # Calculate log2 fold change
                        log2fc = np.log2(cluster1_rate / cluster2_rate) if cluster1_rate > 0 and cluster2_rate > 0 else np.nan
                        
                        # Perform Fisher's exact test - fix contingency table
                        non_cluster1_variants = max(0, cluster1_participants - cluster1_sum)
                        non_cluster2_variants = max(0, cluster2_participants - cluster2_sum)
                        
                        contingency = np.array([
                            [cluster1_sum, non_cluster1_variants],
                            [cluster2_sum, non_cluster2_variants]
                        ])
                        
                        # Only perform test if all values are non-negative
                        if np.all(contingency >= 0):
                            oddsratio, pvalue = stats.fisher_exact(contingency)
                        else:
                            print(f"Warning: Negative values in contingency table for {variant_type} {category} {impact}, clusters {cluster1} vs {cluster2}")
                            continue
                        
                        # Add to results
                        results.append({
                            'cluster1': cluster1,
                            'cluster2': cluster2,
                            'variant_type': variant_type,
                            'category': category,
                            'impact': impact,
                            'cluster1_variants': cluster1_sum,
                            'cluster1_participants': cluster1_participants,
                            'cluster1_rate': cluster1_rate,
                            'cluster2_variants': cluster2_sum,
                            'cluster2_participants': cluster2_participants,
                            'cluster2_rate': cluster2_rate,
                            'log2fc': log2fc,
                            'oddsratio': oddsratio,
                            'pvalue': pvalue
                        })
    
    # Create dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate multiple testing correction
    if not results_df.empty and 'pvalue' in results_df.columns:
        # Add adjusted p-values (FDR)
        results_df['padj'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
    
    return results_df

##############################################
# Helper functions for secondary analyses
##############################################
"""
Helper functions for rare/common variant ratio analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

def calculate_rare_common_ratios(coding_df, cadd_threshold=None, sfari_only=False):
    """
    Calculate rare/common variant ratios for each participant and cluster.
    
    Args:
        coding_df (pd.DataFrame): Long-format coding variant data
        cadd_threshold (float, optional): CADD score threshold to filter variants (e.g., 20)
        sfari_only (bool): Whether to include only SFARI genes
    
    Returns:
        pd.DataFrame: Dataframe with rare/common ratios by participant and cluster
    """
    # Filter non-aggregated entries (with specific genes) if looking at SFARI genes
    if sfari_only:
        # Keep only SFARI genes (filter out gene_name=='--' and non-SFARI genes)
        df = coding_df[(coding_df['gene_name'] != '--') & 
                       (coding_df['sfari_score'].isin(['1', '2', '3', 'S']))]
    else:
        # For overall or high-impact analysis, use the aggregated data rows
        df = coding_df[coding_df['gene_name'] == '--']
    
    # Apply CADD threshold if specified
    if cadd_threshold is not None:
        if 'cadd_bin' in df.columns:
            # Filter based on cadd_bin
            cadd_masks = [
                df['cadd_bin'] == 'PTV',  # Always include PTVs
                df['cadd_bin'] == 'CADD>30',
                (df['cadd_bin'] == 'CADD20-30')
            ]
            df = df[np.logical_or.reduce(cadd_masks)]
        elif 'CADD_phred_hg19' in df.columns:
            # Filter based on numeric CADD score
            df = df[(df['CADD_phred_hg19'] >= cadd_threshold) | 
                    (df['Variant_Category'] == 'PTV')]  # Always include PTVs
    
    # Prepare the ratio calculation
    rare_df = df[df['variant_type'] == 'rare_coding']
    common_df = df[df['variant_type'] == 'common_coding']
    
    # Group by participant and cluster, sum variant counts
    rare_counts = rare_df.groupby(['participant_id', 'cluster'])['count'].sum().reset_index()
    common_counts = common_df.groupby(['participant_id', 'cluster'])['count'].sum().reset_index()
    
    # Ensure each participant appears in both rare and common
    all_participants = pd.DataFrame({
        'participant_id': pd.concat([rare_counts['participant_id'], common_counts['participant_id']]).unique()
    })
    
    # Add cluster information
    all_participants = all_participants.merge(
        pd.concat([rare_counts[['participant_id', 'cluster']], 
                  common_counts[['participant_id', 'cluster']]]).drop_duplicates(),
        on='participant_id',
        how='left'
    )
    
    # Merge rare and common counts
    ratio_df = all_participants.merge(
        rare_counts[['participant_id', 'count']].rename(columns={'count': 'rare_count'}),
        on='participant_id',
        how='left'
    ).merge(
        common_counts[['participant_id', 'count']].rename(columns={'count': 'common_count'}),
        on='participant_id',
        how='left'
    )
    
    # Fill NAs with 0 for participants who don't have one type
    ratio_df['rare_count'] = ratio_df['rare_count'].fillna(0)
    ratio_df['common_count'] = ratio_df['common_count'].fillna(0)
    
    # Calculate ratio (add small pseudo-count to avoid division by zero)
    ratio_df['ratio'] = (ratio_df['rare_count'] + 0.1) / (ratio_df['common_count'] + 0.1)
    
    # Add log ratio for more normal distribution in statistical tests
    ratio_df['log_ratio'] = np.log2(ratio_df['ratio'])
    
    # Add descriptive columns based on the filters applied
    ratio_df['analysis_type'] = 'overall'
    if cadd_threshold is not None:
        ratio_df['analysis_type'] = f'high_impact_cadd{cadd_threshold}'
    if sfari_only:
        ratio_df['analysis_type'] = ratio_df['analysis_type'] + '_sfari'
    
    return ratio_df

def compare_ratios_across_clusters(ratio_df, analysis_name):
    """
    Perform statistical tests to compare ratios across clusters.
    
    Args:
        ratio_df (pd.DataFrame): DataFrame with rare/common ratios
        analysis_name (str): Name to identify this analysis
    
    Returns:
        pd.DataFrame: Results of statistical tests
    """
    # Initialize results list
    results = []
    
    # 1. Kruskal-Wallis test across all clusters
    clusters = sorted(ratio_df['cluster'].unique())
    if len(clusters) > 1:  # Only if multiple clusters
        # Create list of samples for KW test
        samples = [ratio_df[ratio_df['cluster'] == c]['log_ratio'].values for c in clusters]
        
        # Remove any empty samples
        samples = [s for s in samples if len(s) > 0]
        
        if len(samples) > 1:  # Need at least 2 samples
            kw_stat, kw_pvalue = stats.kruskal(*samples)
            
            results.append({
                'analysis': analysis_name,
                'test': 'kruskal_wallis',
                'cluster1': 'all',
                'cluster2': 'all',
                'statistic': kw_stat,
                'pvalue': kw_pvalue,
                'adjusted_pvalue': kw_pvalue,  # Will be adjusted after all tests
                'significant': kw_pvalue < 0.05
            })
    
    # 2. Pairwise Mann-Whitney U tests
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:
            # Get ratio data for each cluster
            c1_ratios = ratio_df[ratio_df['cluster'] == c1]['log_ratio'].values
            c2_ratios = ratio_df[ratio_df['cluster'] == c2]['log_ratio'].values
            
            # Skip if either is empty
            if len(c1_ratios) == 0 or len(c2_ratios) == 0:
                continue
            
            # Mann-Whitney U test
            mw_stat, mw_pvalue = stats.mannwhitneyu(c1_ratios, c2_ratios, alternative='two-sided')
            
            # Calculate medians for easier interpretation - CHANGED FROM MEANS TO MEDIANS
            c1_median = ratio_df[ratio_df['cluster'] == c1]['ratio'].median()
            c2_median = ratio_df[ratio_df['cluster'] == c2]['ratio'].median()

            results.append({
                'analysis': analysis_name,
                'test': 'pairwise',
                'cluster1': c1,
                'cluster2': c2,
                'statistic': mw_stat,
                'pvalue': mw_pvalue,
                'c1_median_ratio': c1_median,  # Changed from mean to median
                'c2_median_ratio': c2_median,  # Changed from mean to median
                'ratio_fold_change': c1_median / c2_median if c2_median > 0 else np.nan  # Using medians
            })

            # # Calculate means for easier interpretation
            # c1_mean = ratio_df[ratio_df['cluster'] == c1]['ratio'].mean()
            # c2_mean = ratio_df[ratio_df['cluster'] == c2]['ratio'].mean()
            
            # results.append({
            #     'analysis': analysis_name,
            #     'test': 'pairwise',
            #     'cluster1': c1,
            #     'cluster2': c2,
            #     'statistic': mw_stat,
            #     'pvalue': mw_pvalue,
            #     'c1_mean_ratio': c1_mean,
            #     'c2_mean_ratio': c2_mean,
            #     'ratio_fold_change': c1_mean / c2_mean if c2_mean > 0 else np.nan
            # })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing correction if we have pairwise tests
    pairwise_tests = results_df[results_df['test'] == 'pairwise']
    if not pairwise_tests.empty:
        # Adjust p-values for pairwise tests
        pairwise_indices = pairwise_tests.index
        adjusted_pvalues = multipletests(pairwise_tests['pvalue'].values, method='fdr_bh')[1]
        
        # Add to results
        results_df.loc[pairwise_indices, 'adjusted_pvalue'] = adjusted_pvalues
        results_df.loc[pairwise_indices, 'significant'] = adjusted_pvalues < 0.05
    
    return results_df


# For gene main analysis # 
def extract_ptv_variant_gene_counts(coding_data, variant_type="rare", pli_bins=["High_high_pLI", "High_low_pLI"]):
    """
    Extracts PTV variant counts and gene counts per cluster for each pLI bin.
    Returns two DataFrames: one for variant counts, one for gene counts.
    """
    variant_rows = []
    gene_rows = []
    
    for cluster, cluster_data in coding_data.items():
        variant_counts = cluster_data["variant_counts"].get(variant_type, {}).get("PTV", {})
        gene_counts = cluster_data["gene_counts"].get(variant_type, {}).get("PTV", {})
        
        for pli_bin in pli_bins:
            v_count = variant_counts.get(pli_bin, 0)
            g_count = gene_counts.get(pli_bin, 0)
            variant_rows.append({"cluster": int(cluster), "pli_bin": pli_bin, "count": v_count})
            gene_rows.append({"cluster": int(cluster), "pli_bin": pli_bin, "count": g_count})
    
    df_variants = pd.DataFrame(variant_rows)
    df_genes = pd.DataFrame(gene_rows)
    return df_variants, df_genes


def extract_missense_variant_gene_counts(coding_data, variant_type="rare"):
    """
    Extract missense variant and gene counts by CADD x pLI bins across clusters.
    Returns two DataFrames: one for variants, one for genes.
    """
    cadd_bins = ["High", "Moderate", "Low"]
    pli_bins = ["High_high_pLI", "High_low_pLI", "Moderate_high_pLI", "Moderate_low_pLI", "Low_high_pLI", "Low_low_pLI"]
    
    variant_rows = []
    gene_rows = []
    
    for cluster, cluster_data in coding_data.items():
        var_counts = cluster_data["variant_counts"].get(variant_type, {}).get("Missense", {})
        gene_counts = cluster_data["gene_counts"].get(variant_type, {}).get("Missense", {})
        
        for bin_label in pli_bins:
            count = var_counts.get(bin_label, 0)
            variant_rows.append({"cluster": int(cluster), "impact_bin": bin_label, "count": count})
            
            g_count = gene_counts.get(bin_label, 0)
            gene_rows.append({"cluster": int(cluster), "impact_bin": bin_label, "count": g_count})
    
    return pd.DataFrame(variant_rows), pd.DataFrame(gene_rows)


def extract_synonymous_variant_gene_counts(coding_data, variant_type="rare"):
    """
    Extract synonymous variant and gene counts by pLI bins across clusters.
    """
    pli_bins = ["Low_high_pLI", "Low_low_pLI"]
    
    variant_rows = []
    gene_rows = []
    
    for cluster, cluster_data in coding_data.items():
        var_counts = cluster_data["variant_counts"].get(variant_type, {}).get("Synonymous", {})
        gene_counts = cluster_data["gene_counts"].get(variant_type, {}).get("Synonymous", {})
        
        for bin_label in pli_bins:
            count = var_counts.get(bin_label, 0)
            variant_rows.append({"cluster": int(cluster), "impact_bin": bin_label, "count": count})
            
            g_count = gene_counts.get(bin_label, 0)
            gene_rows.append({"cluster": int(cluster), "impact_bin": bin_label, "count": g_count})
    
    return pd.DataFrame(variant_rows), pd.DataFrame(gene_rows)

def compare_variant_burden_by_cluster(df, variant_type, value_type="variant"):
    """
    Performs Kruskal-Wallis and pairwise tests across clusters.
    
    Args:
        df (pd.DataFrame): DataFrame with 'cluster', 'impact_bin', 'count'
        variant_type (str): e.g., PTV, Missense
        value_type (str): 'variant' or 'gene'
    
    Returns:
        pd.DataFrame: Stats with p-values and comparisons
    """
    from scipy.stats import kruskal, mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    results = []
    for impact_bin in df['impact_bin'].unique():
        sub_df = df[df['impact_bin'] == impact_bin]
        grouped = [sub_df[sub_df['cluster'] == c]['count'].values for c in sorted(sub_df['cluster'].unique())]
        
        if len(grouped) >= 2:
            kw_stat, kw_p = kruskal(*grouped)
            results.append({
                "variant_type": variant_type,
                "impact_bin": impact_bin,
                "test": "kruskal",
                "statistic": kw_stat,
                "pvalue": kw_p
            })
            
            # Pairwise
            clusters = sorted(sub_df['cluster'].unique())
            for i, c1 in enumerate(clusters):
                for c2 in clusters[i+1:]:
                    x = sub_df[sub_df['cluster'] == c1]['count']
                    y = sub_df[sub_df['cluster'] == c2]['count']
                    if len(x) > 0 and len(y) > 0:
                        stat, p = mannwhitneyu(x, y, alternative='two-sided')
                        results.append({
                            "variant_type": variant_type,
                            "impact_bin": impact_bin,
                            "test": "pairwise",
                            "cluster1": c1,
                            "cluster2": c2,
                            "statistic": stat,
                            "pvalue": p
                        })
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['adjusted_pvalue'] = multipletests(df_results['pvalue'], method='fdr_bh')[1]
        df_results['significant'] = df_results['adjusted_pvalue'] < 0.05
    
    return df_results


"""
Additional statistical functions for genetic variant analysis following Huang et al. approach.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import List, Dict, Tuple, Any, Union, Optional

def calculate_cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cliff's Delta effect size between two groups.
    
    Cliff's Delta is a non-parametric effect size measure that quantifies the amount of
    difference between two groups. It's appropriate for non-normally distributed data
    and small sample sizes.
    
    Args:
        group1 (np.ndarray): First group values
        group2 (np.ndarray): Second group values
        
    Returns:
        float: Cliff's Delta (-1 to 1, with 0 indicating no difference)
    """
    # Ensure inputs are numpy arrays for efficient calculation
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    # Calculate the number of comparisons where group1 > group2, group1 < group2, and group1 == group2
    n1, n2 = len(group1), len(group2)
    
    if n1 == 0 or n2 == 0:
        return np.nan
    
    # Use numpy broadcasting for efficient calculation
    greater = sum(g1 > g2 for g1 in group1 for g2 in group2)
    less = sum(g1 < g2 for g1 in group1 for g2 in group2)
    
    # Calculate Cliff's Delta
    delta = (greater - less) / (n1 * n2)
    
    return delta

def interpret_cliffs_delta(delta: float) -> str:
    """
    Interpret Cliff's Delta effect size according to conventional thresholds.
    
    Args:
        delta (float): Cliff's Delta value
        
    Returns:
        str: Interpretation ("negligible", "small", "medium", or "large")
    """
    delta_abs = abs(delta)
    
    if delta_abs < 0.147:
        return "negligible"
    elif delta_abs < 0.33:
        return "small"
    elif delta_abs < 0.474:
        return "medium"
    else:
        return "large"

def hierarchical_coding_analysis(coding_data: Dict, variant_type: str = "rare") -> pd.DataFrame:
    """
    Perform hierarchical analysis of coding variants following the Huang structure.
    
    This follows the hierarchical structure:
    1. PTV Analysis (stratified by pLI constraint)
    2. Missense Analysis (stratified by CADD score and pLI)
    3. Synonymous Analysis (stratified by pLI)
    
    Args:
        coding_data (dict): The loaded coding variant data by cluster
        variant_type (str): Type of variant to analyze ('rare' or 'common')
        
    Returns:
        pd.DataFrame: Results organized by hierarchy level with statistics
    """
    results = []
    
    # 1. PTV Analysis
    # Get variant counts and gene counts for PTV, stratified by pLI
    ptv_variant_df, ptv_gene_df = extract_ptv_variant_gene_counts(
        coding_data, variant_type=variant_type, 
        pli_bins=["High_high_pLI", "High_low_pLI"]
    )
    
    # Rename columns for consistency
    ptv_variant_df = ptv_variant_df.rename(columns={'pli_bin': 'impact_bin'})
    ptv_gene_df = ptv_gene_df.rename(columns={'pli_bin': 'impact_bin'})
    
    # Test for cluster differences in PTV variants
    ptv_variant_stats = compare_variant_burden_by_cluster(
        ptv_variant_df, variant_type="PTV", value_type="variant"
    )
    
    # Test for cluster differences in PTV genes
    ptv_gene_stats = compare_variant_burden_by_cluster(
        ptv_gene_df, variant_type="PTV", value_type="gene"
    )
    
    # Add statistics to results
    for _, row in ptv_variant_stats.iterrows():
        if row['test'] == 'pairwise':
            # Get data for effect size calculation
            c1_data = ptv_variant_df[(ptv_variant_df['cluster'] == row['cluster1']) & 
                                   (ptv_variant_df['impact_bin'] == row['impact_bin'])]['count'].values
            c2_data = ptv_variant_df[(ptv_variant_df['cluster'] == row['cluster2']) & 
                                   (ptv_variant_df['impact_bin'] == row['impact_bin'])]['count'].values
            
            # Calculate effect size
            effect_size = calculate_cliffs_delta(c1_data, c2_data)
            effect_interp = interpret_cliffs_delta(effect_size)
            
            results.append({
                'hierarchy': 'PTV',
                'category': 'PTV',
                'impact_bin': row['impact_bin'],
                'test': row['test'],
                'cluster1': row['cluster1'],
                'cluster2': row['cluster2'],
                'statistic': row['statistic'],
                'pvalue': row['pvalue'],
                'adjusted_pvalue': row['adjusted_pvalue'],
                'significant': row['significant'],
                'effect_size': effect_size,
                'effect_interpretation': effect_interp,
                'value_type': 'variant',
                'variant_type': variant_type
            })
    
    # 2. Missense Analysis with CADD stratification
    missense_variant_df, missense_gene_df = extract_missense_variant_gene_counts(
        coding_data, variant_type=variant_type
    )
    
    # Test for cluster differences in Missense variants
    missense_variant_stats = compare_variant_burden_by_cluster(
        missense_variant_df, variant_type="Missense", value_type="variant"
    )
    
    # Test for cluster differences in Missense genes
    missense_gene_stats = compare_variant_burden_by_cluster(
        missense_gene_df, variant_type="Missense", value_type="gene"
    )
    
    # Map impact bins to CADD categories (for easier interpretation)
    cadd_mapping = {
        'High_high_pLI': 'CADD>30_pLI≥0.9',
        'High_low_pLI': 'CADD>30_pLI<0.9',
        'Moderate_high_pLI': 'CADD20-30_pLI≥0.9',
        'Moderate_low_pLI': 'CADD20-30_pLI<0.9',
        'Low_high_pLI': 'CADD0-20_pLI≥0.9',
        'Low_low_pLI': 'CADD0-20_pLI<0.9'
    }
    
    # Map impact bins to hierarchy categories
    hierarchy_mapping = {
        'High_high_pLI': 'Missense (CADD>30)',
        'High_low_pLI': 'Missense (CADD>30)',
        'Moderate_high_pLI': 'Missense (CADD20-30)',
        'Moderate_low_pLI': 'Missense (CADD20-30)',
        'Low_high_pLI': 'Missense (CADD0-20)',
        'Low_low_pLI': 'Missense (CADD0-20)'
    }
    
    # Add statistics to results
    for _, row in missense_variant_stats.iterrows():
        if row['test'] == 'pairwise':
            # Get data for effect size calculation
            c1_data = missense_variant_df[(missense_variant_df['cluster'] == row['cluster1']) & 
                                        (missense_variant_df['impact_bin'] == row['impact_bin'])]['count'].values
            c2_data = missense_variant_df[(missense_variant_df['cluster'] == row['cluster2']) & 
                                        (missense_variant_df['impact_bin'] == row['impact_bin'])]['count'].values
            
            # Calculate effect size
            effect_size = calculate_cliffs_delta(c1_data, c2_data)
            effect_interp = interpret_cliffs_delta(effect_size)
            
            results.append({
                'hierarchy': hierarchy_mapping.get(row['impact_bin'], 'Missense'),
                'category': 'Missense',
                'impact_bin': cadd_mapping.get(row['impact_bin'], row['impact_bin']),
                'test': row['test'],
                'cluster1': row['cluster1'],
                'cluster2': row['cluster2'],
                'statistic': row['statistic'],
                'pvalue': row['pvalue'],
                'adjusted_pvalue': row['adjusted_pvalue'],
                'significant': row['significant'],
                'effect_size': effect_size,
                'effect_interpretation': effect_interp,
                'value_type': 'variant',
                'variant_type': variant_type
            })
    
    # 3. Synonymous Analysis
    syn_variant_df, syn_gene_df = extract_synonymous_variant_gene_counts(
        coding_data, variant_type=variant_type
    )
    
    # Test for cluster differences in Synonymous variants
    syn_variant_stats = compare_variant_burden_by_cluster(
        syn_variant_df, variant_type="Synonymous", value_type="variant"
    )
    
    # Test for cluster differences in Synonymous genes
    syn_gene_stats = compare_variant_burden_by_cluster(
        syn_gene_df, variant_type="Synonymous", value_type="gene"
    )
    
    # Map impact bins for synonymous
    syn_mapping = {
        'Low_high_pLI': 'pLI≥0.9',
        'Low_low_pLI': 'pLI<0.9'
    }
    
    # Add statistics to results
    for _, row in syn_variant_stats.iterrows():
        if row['test'] == 'pairwise':
            # Get data for effect size calculation
            c1_data = syn_variant_df[(syn_variant_df['cluster'] == row['cluster1']) & 
                                   (syn_variant_df['impact_bin'] == row['impact_bin'])]['count'].values
            c2_data = syn_variant_df[(syn_variant_df['cluster'] == row['cluster2']) & 
                                   (syn_variant_df['impact_bin'] == row['impact_bin'])]['count'].values
            
            # Calculate effect size
            effect_size = calculate_cliffs_delta(c1_data, c2_data)
            effect_interp = interpret_cliffs_delta(effect_size)
            
            results.append({
                'hierarchy': 'Synonymous',
                'category': 'Synonymous',
                'impact_bin': syn_mapping.get(row['impact_bin'], row['impact_bin']),
                'test': row['test'],
                'cluster1': row['cluster1'],
                'cluster2': row['cluster2'],
                'statistic': row['statistic'],
                'pvalue': row['pvalue'],
                'adjusted_pvalue': row['adjusted_pvalue'],
                'significant': row['significant'],
                'effect_size': effect_size,
                'effect_interpretation': effect_interp,
                'value_type': 'variant',
                'variant_type': variant_type
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply additional FDR correction across all tests if needed
    if not results_df.empty and 'pvalue' in results_df.columns:
        results_df['global_adjusted_pvalue'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
        results_df['globally_significant'] = results_df['global_adjusted_pvalue'] < 0.05
    
    return results_df

def hierarchical_noncoding_analysis(noncoding_data: Dict, variant_type: str = "rare") -> pd.DataFrame:
    """
    Perform hierarchical analysis of non-coding variants.
    
    This follows the hierarchical structure:
    1. Regulatory Elements (Promoter, Enhancer)
    2. Other Non-coding Regions (Intronic, Intergenic, Other)
    
    Each stratified by impact (High_impact, Low_impact based on Eigen score)
    
    Args:
        noncoding_data (dict): The loaded non-coding variant data by cluster
        variant_type (str): Type of variant to analyze ('rare' or 'common')
        
    Returns:
        pd.DataFrame: Results organized by hierarchy level with statistics
    """
    results = []
    
    # Extract and organize non-coding variant data
    noncoding_df = extract_noncoding_variant_counts(noncoding_data)
    
    # 1. Analyze Regulatory Elements
    regulatory_categories = ['Promoter', 'Enhancer']
    for category in regulatory_categories:
        # Filter for this category
        category_df = noncoding_df[noncoding_df['category'] == category]
        
        # Test for cluster differences
        category_stats = compare_noncoding_burden_by_cluster(
            category_df, category=category
        )
        
        # Add statistics to results
        for _, row in category_stats.iterrows():
            if row['test'] == 'pairwise':
                # Get data for effect size calculation
                c1_data = category_df[(category_df['cluster'] == row['cluster1']) & 
                                    (category_df['impact'] == row['impact'])]['count'].values
                c2_data = category_df[(category_df['cluster'] == row['cluster2']) & 
                                    (category_df['impact'] == row['impact'])]['count'].values
                
                # Calculate effect size
                effect_size = calculate_cliffs_delta(c1_data, c2_data)
                effect_interp = interpret_cliffs_delta(effect_size)
                
                results.append({
                    'hierarchy': 'Regulatory',
                    'category': category,
                    'impact': row['impact'],
                    'test': row['test'],
                    'cluster1': row['cluster1'],
                    'cluster2': row['cluster2'],
                    'statistic': row['statistic'],
                    'pvalue': row['pvalue'],
                    'adjusted_pvalue': row['adjusted_pvalue'],
                    'significant': row['significant'],
                    'effect_size': effect_size,
                    'effect_interpretation': effect_interp,
                    'variant_type': variant_type
                })
    
    # 2. Analyze Other Non-coding Regions
    other_categories = ['Intronic', 'Intergenic', 'Other']
    for category in other_categories:
        # Filter for this category
        category_df = noncoding_df[noncoding_df['category'] == category]
        
        # Test for cluster differences
        category_stats = compare_noncoding_burden_by_cluster(
            category_df, category=category
        )
        
        # Add statistics to results
        for _, row in category_stats.iterrows():
            if row['test'] == 'pairwise':
                # Get data for effect size calculation
                c1_data = category_df[(category_df['cluster'] == row['cluster1']) & 
                                    (category_df['impact'] == row['impact'])]['count'].values
                c2_data = category_df[(category_df['cluster'] == row['cluster2']) & 
                                    (category_df['impact'] == row['impact'])]['count'].values
                
                # Calculate effect size
                effect_size = calculate_cliffs_delta(c1_data, c2_data)
                effect_interp = interpret_cliffs_delta(effect_size)
                
                results.append({
                    'hierarchy': 'Other Non-coding',
                    'category': category,
                    'impact': row['impact'],
                    'test': row['test'],
                    'cluster1': row['cluster1'],
                    'cluster2': row['cluster2'],
                    'statistic': row['statistic'],
                    'pvalue': row['pvalue'],
                    'adjusted_pvalue': row['adjusted_pvalue'],
                    'significant': row['significant'],
                    'effect_size': effect_size,
                    'effect_interpretation': effect_interp,
                    'variant_type': variant_type
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply additional FDR correction across all tests if needed
    if not results_df.empty and 'pvalue' in results_df.columns:
        results_df['global_adjusted_pvalue'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
        results_df['globally_significant'] = results_df['global_adjusted_pvalue'] < 0.05
    
    return results_df

def extract_noncoding_variant_counts(noncoding_data: Dict) -> pd.DataFrame:
    """
    Extract and format non-coding variant counts from nested dictionary structure.
    
    Args:
        noncoding_data (dict): Dictionary of non-coding variant data by cluster
        
    Returns:
        pd.DataFrame: Long-format DataFrame with non-coding variant counts
    """
    rows = []
    
    # Process regulatory elements
    regulatory_categories = ['Promoter', 'Enhancer']
    other_categories = ['Intronic', 'Intergenic', 'Other']
    impact_types = ['High_impact', 'Low_impact']
    
    for cluster, cluster_data in noncoding_data.items():
        # Get variant counts
        for category in regulatory_categories + other_categories:
            for impact in impact_types:
                count = cluster_data.get('variant_counts', {}).get(category, {}).get(impact, 0)
                
                rows.append({
                    'cluster': int(cluster),
                    'category': category,
                    'impact': impact,
                    'count': count,
                    'region_type': 'Regulatory' if category in regulatory_categories else 'Other'
                })
    
    return pd.DataFrame(rows)

def compare_noncoding_burden_by_cluster(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Compare non-coding variant burden across clusters using Mann-Whitney and Kruskal-Wallis tests.
    
    Args:
        df (pd.DataFrame): DataFrame with non-coding variant counts
        category (str): Category to analyze (e.g., 'Promoter', 'Intronic')
        
    Returns:
        pd.DataFrame: Statistical test results
    """
    from scipy.stats import kruskal, mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    
    results = []
    
    # Filter for the specified category
    cat_df = df[df['category'] == category]
    
    # For each impact level
    for impact in cat_df['impact'].unique():
        impact_df = cat_df[cat_df['impact'] == impact]
        
        # Kruskal-Wallis test across all clusters
        clusters = sorted(impact_df['cluster'].unique())
        grouped = [impact_df[impact_df['cluster'] == c]['count'].values for c in clusters]
        
        if len(grouped) >= 2 and all(len(g) > 0 for g in grouped):
            try:
                kw_stat, kw_p = kruskal(*grouped)
                results.append({
                    "category": category,
                    "impact": impact,
                    "test": "kruskal",
                    "statistic": kw_stat,
                    "pvalue": kw_p
                })
            except Exception as e:
                print(f"Kruskal-Wallis test failed for {category}, {impact}: {e}")
        
        # Pairwise Mann-Whitney U tests
        for i, c1 in enumerate(clusters):
            for c2 in clusters[i+1:]:
                x = impact_df[impact_df['cluster'] == c1]['count'].values
                y = impact_df[impact_df['cluster'] == c2]['count'].values
                
                if len(x) > 0 and len(y) > 0:
                    try:
                        stat, p = mannwhitneyu(x, y, alternative='two-sided')
                        results.append({
                            "category": category,
                            "impact": impact,
                            "test": "pairwise",
                            "cluster1": c1,
                            "cluster2": c2,
                            "statistic": stat,
                            "pvalue": p
                        })
                    except Exception as e:
                        print(f"Mann-Whitney test failed for {category}, {impact}, clusters {c1} vs {c2}: {e}")
    
    # Create DataFrame from results
    df_results = pd.DataFrame(results)
    
    # Calculate multiple testing correction if we have results
    if not df_results.empty and 'pvalue' in df_results.columns:
        df_results['adjusted_pvalue'] = multipletests(df_results['pvalue'], method='fdr_bh')[1]
        df_results['significant'] = df_results['adjusted_pvalue'] < 0.05
    
    return df_results

def one_vs_all_hierarchical_analysis(coding_data: Dict, noncoding_data: Dict, 
                                   variant_type: str = "rare") -> pd.DataFrame:
    """
    Perform one-vs-all hierarchical analysis for coding and non-coding variants.
    
    This is a high-level function that combines both coding and non-coding analyses
    in a one-vs-all comparison framework.
    
    Args:
        coding_data (dict): The loaded coding variant data by cluster
        noncoding_data (dict): The loaded non-coding variant data by cluster
        variant_type (str): Type of variant to analyze ('rare' or 'common')
        
    Returns:
        pd.DataFrame: Combined results from all analyses
    """
    # Get participant counts per cluster
    participant_counts = get_participant_counts(coding_data)
    
    # Transform coding data to cluster-level for one-vs-all
    coding_cluster_df = transform_coding_to_cluster_level(coding_data, variant_type, participant_counts)
    
    # Transform non-coding data to cluster-level for one-vs-all
    noncoding_cluster_df = transform_noncoding_to_cluster_level(noncoding_data, variant_type, participant_counts)
    
    # Perform one-vs-all analysis for each cluster
    results = []
    clusters = sorted(participant_counts.keys())
    
    # For coding variants
    for cluster in clusters:
        # Compare this cluster to all others
        cluster_results = compare_cluster_to_all(
            coding_cluster_df, cluster, "coding", variant_type
        )
        results.extend(cluster_results)
    
    # For non-coding variants
    for cluster in clusters:
        # Compare this cluster to all others
        cluster_results = compare_cluster_to_all(
            noncoding_cluster_df, cluster, "noncoding", variant_type
        )
        results.extend(cluster_results)
    
    # Convert to DataFrame and apply FDR correction
    results_df = pd.DataFrame(results)
    
    if not results_df.empty and 'pvalue' in results_df.columns:
        results_df['adjusted_pvalue'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
        results_df['significant'] = results_df['adjusted_pvalue'] < 0.05
    
    return results_df

def get_participant_counts(coding_data: Dict) -> Dict[int, int]:
    """
    Estimate the number of participants in each cluster.
    
    Args:
        coding_data (dict): The loaded coding variant data by cluster
        
    Returns:
        dict: Dictionary mapping cluster ID to participant count
    """
    # In a real scenario, you might need to adjust this based on how your data is structured
    # This is just a placeholder implementation
    participant_counts = {}
    
    for cluster_id, cluster_data in coding_data.items():
        # Estimate participants based on common variants (likely present in all)
        # This is approximate and should be adjusted for your specific data structure
        participant_counts[int(cluster_id)] = 10  # Assuming ~10 participants per cluster as mentioned
    
    return participant_counts

def transform_coding_to_cluster_level(coding_data: Dict, variant_type: str, 
                                     participant_counts: Dict[int, int]) -> pd.DataFrame:
    """
    Transform coding data to cluster-level format for one-vs-all analysis.
    
    Args:
        coding_data (dict): The loaded coding variant data by cluster
        variant_type (str): Type of variant to analyze ('rare' or 'common')
        participant_counts (dict): Dictionary mapping cluster ID to participant count
        
    Returns:
        pd.DataFrame: Long-format DataFrame suitable for one-vs-all comparisons
    """
    rows = []
    
    # Process PTVs
    for cluster_id, cluster_data in coding_data.items():
        cluster_id = int(cluster_id)
        
        # PTV analysis
        ptv_counts = cluster_data.get('variant_counts', {}).get(variant_type, {}).get('PTV', {})
        for pli_bin, count in ptv_counts.items():
            # Normalize by participant count
            rate = count / participant_counts.get(cluster_id, 1)
            
            rows.append({
                'cluster': cluster_id,
                'hierarchy': 'PTV',
                'category': 'PTV',
                'impact_bin': pli_bin,
                'count': count,
                'rate': rate,
                'participants': participant_counts.get(cluster_id, 0),
                'variant_type': variant_type
            })
        
        # Missense analysis
        missense_counts = cluster_data.get('variant_counts', {}).get(variant_type, {}).get('Missense', {})
        
        # Map impact bins to hierarchy categories
        hierarchy_mapping = {
            'High_high_pLI': 'Missense (CADD>30)',
            'High_low_pLI': 'Missense (CADD>30)',
            'Moderate_high_pLI': 'Missense (CADD20-30)',
            'Moderate_low_pLI': 'Missense (CADD20-30)',
            'Low_high_pLI': 'Missense (CADD0-20)',
            'Low_low_pLI': 'Missense (CADD0-20)'
        }
        
        for impact_bin, count in missense_counts.items():
            # Normalize by participant count
            rate = count / participant_counts.get(cluster_id, 1)
            
            rows.append({
                'cluster': cluster_id,
                'hierarchy': hierarchy_mapping.get(impact_bin, 'Missense'),
                'category': 'Missense',
                'impact_bin': impact_bin,
                'count': count,
                'rate': rate,
                'participants': participant_counts.get(cluster_id, 0),
                'variant_type': variant_type
            })
        
        # Synonymous analysis
        syn_counts = cluster_data.get('variant_counts', {}).get(variant_type, {}).get('Synonymous', {})
        for pli_bin, count in syn_counts.items():
            # Normalize by participant count
            rate = count / participant_counts.get(cluster_id, 1)
            
            rows.append({
                'cluster': cluster_id,
                'hierarchy': 'Synonymous',
                'category': 'Synonymous',
                'impact_bin': pli_bin,
                'count': count,
                'rate': rate,
                'participants': participant_counts.get(cluster_id, 0),
                'variant_type': variant_type
            })
    
    return pd.DataFrame(rows)

def transform_noncoding_to_cluster_level(noncoding_data: Dict, variant_type: str, 
                                       participant_counts: Dict[int, int]) -> pd.DataFrame:
    """
    Transform non-coding data to cluster-level format for one-vs-all analysis.
    
    Args:
        noncoding_data (dict): The loaded non-coding variant data by cluster
        variant_type (str): Type of variant to analyze ('rare' or 'common')
        participant_counts (dict): Dictionary mapping cluster ID to participant count
        
    Returns:
        pd.DataFrame: Long-format DataFrame suitable for one-vs-all comparisons
    """
    rows = []
    
    # Process non-coding variants
    regulatory_categories = ['Promoter', 'Enhancer']
    other_categories = ['Intronic', 'Intergenic', 'Other']
    impact_types = ['High_impact', 'Low_impact']
    
    for cluster_id, cluster_data in noncoding_data.items():
        cluster_id = int(cluster_id)
        
        # Regulatory elements
        for category in regulatory_categories:
            for impact in impact_types:
                count = cluster_data.get('variant_counts', {}).get(category, {}).get(impact, 0)
                # Normalize by participant count
                rate = count / participant_counts.get(cluster_id, 1)
                
                rows.append({
                    'cluster': cluster_id,
                    'hierarchy': 'Regulatory',
                    'category': category,
                    'impact': impact,
                    'count': count,
                    'rate': rate,
                    'participants': participant_counts.get(cluster_id, 0),
                    'variant_type': variant_type
                })
        
        # Other non-coding regions
        for category in other_categories:
            for impact in impact_types:
                count = cluster_data.get('variant_counts', {}).get(category, {}).get(impact, 0)
                # Normalize by participant count
                rate = count / participant_counts.get(cluster_id, 1)
                
                rows.append({
                    'cluster': cluster_id,
                    'hierarchy': 'Other Non-coding',
                    'category': category,
                    'impact': impact,
                    'count': count,
                    'rate': rate,
                    'participants': participant_counts.get(cluster_id, 0),
                    'variant_type': variant_type
                })
    
    return pd.DataFrame(rows)

def compare_cluster_to_all(df: pd.DataFrame, target_cluster: int, 
                         variant_domain: str, variant_type: str) -> List[Dict]:
    """
    Compare a specific cluster to all other clusters for each category.
    
    Args:
        df (pd.DataFrame): DataFrame with cluster-level data
        target_cluster (int): Cluster ID to compare against all others
        variant_domain (str): Domain of variants ('coding' or 'noncoding')
        variant_type (str): Type of variant ('rare' or 'common')
        
    Returns:
        list: List of dictionaries with comparison results
    """
    from scipy.stats import mannwhitneyu
    
    results = []
    
    # Get all hierarchies in the data
    hierarchies = df['hierarchy'].unique()
    
    for hierarchy in hierarchies:
        # Filter for this hierarchy
        hierarchy_df = df[df['hierarchy'] == hierarchy]
        
        # Get all categories in this hierarchy
        categories = hierarchy_df['category'].unique()
        
        for category in categories:
            # Filter for this category
            category_df = hierarchy_df[hierarchy_df['category'] == category]
            
            # Get all impact bins or levels
            if 'impact_bin' in category_df.columns:
                impact_levels = category_df['impact_bin'].unique()
                impact_col = 'impact_bin'
            elif 'impact' in category_df.columns:
                impact_levels = category_df['impact'].unique()
                impact_col = 'impact'
            else:
                continue  # Skip if no impact column
            
            for impact in impact_levels:
                # Filter for this impact
                impact_df = category_df[category_df[impact_col] == impact]
                
                # Get data for target cluster
                target_df = impact_df[impact_df['cluster'] == target_cluster]
                
                # Get data for all other clusters
                other_df = impact_df[impact_df['cluster'] != target_cluster]
                
                # Skip if no data for either group
                if target_df.empty or other_df.empty:
                    continue
                
                # Calculate statistics
                target_count = target_df['count'].sum()
                other_count = other_df['count'].sum()
                
                target_participants = target_df['participants'].iloc[0]
                other_participants = other_df['participants'].sum()
                
                target_rate = target_count / target_participants if target_participants > 0 else 0
                other_rate = other_count / other_participants if other_participants > 0 else 0
                
                # Calculate log2 fold change
                log2fc = np.log2(target_rate / other_rate) if target_rate > 0 and other_rate > 0 else np.nan
                
                # For statistical test, we need individual data points per cluster
                # This is where the Mann-Whitney U test is useful
                try:
                    # For target cluster, we use the single value repeated if needed
                    target_values = target_df['rate'].values
                    
                    # For other clusters, we should have individual rates
                    other_values = other_df['rate'].values
                    
                    # Perform Mann-Whitney U test
                    stat, pvalue = mannwhitneyu(target_values, other_values, alternative='two-sided')
                    
                    # Calculate effect size
                    effect_size = calculate_cliffs_delta(target_values, other_values)
                    effect_interp = interpret_cliffs_delta(effect_size)
                    
                    # Add to results
                    results.append({
                        'variant_domain': variant_domain,
                        'variant_type': variant_type,
                        'hierarchy': hierarchy,
                        'category': category,
                        'impact': impact,
                        'cluster': target_cluster,
                        'other_clusters': 'all',
                        'target_count': target_count,
                        'target_participants': target_participants,
                        'target_rate': target_rate,
                        'other_count': other_count,
                        'other_participants': other_participants,
                        'other_rate': other_rate,
                        'log2fc': log2fc,
                        'statistic': stat,
                        'pvalue': pvalue,
                        'effect_size': effect_size,
                        'effect_interpretation': effect_interp
                    })
                    
                except Exception as e:
                    print(f"Statistical test failed for {hierarchy}, {category}, {impact}, cluster {target_cluster}: {e}")
    
    return results

def normalize_variant_counts_by_gene_size(variant_df, gene_length_df=None):
    """
    Normalize variant counts by gene length to account for size bias.
    
    Args:
        variant_df (pd.DataFrame): DataFrame with variant counts by gene
        gene_length_df (pd.DataFrame, optional): DataFrame with gene lengths
        
    Returns:
        pd.DataFrame: DataFrame with normalized variant counts
    """
    # If no gene length information is provided, return original data
    if gene_length_df is None:
        print("Warning: No gene length information provided. Using raw counts.")
        return variant_df.copy()
    
    # Merge variant data with gene length data
    merged_df = variant_df.merge(
        gene_length_df,
        on='gene_name',
        how='left'
    )
    
    # Fill missing gene lengths with median (for genes not in the reference)
    median_length = gene_length_df['gene_length'].median()
    merged_df['gene_length'] = merged_df['gene_length'].fillna(median_length)
    
    # Normalize counts by gene length (per kilobase)
    merged_df['normalized_count'] = merged_df['count'] / (merged_df['gene_length'] / 1000)
    
    return merged_df

def analyze_sfari_gene_enrichment(coding_data, sfari_gene_df, variant_type="rare"):
    """
    Analyze enrichment of variants in SFARI autism genes across clusters.
    
    Args:
        coding_data (dict): The loaded coding variant data by cluster
        sfari_gene_df (pd.DataFrame): DataFrame with SFARI gene information
        variant_type (str): Type of variant to analyze ('rare' or 'common')
        
    Returns:
        pd.DataFrame: Results of SFARI gene enrichment analysis
    """
    results = []
    
    # Extract gene-level data for the specified variant type
    gene_df = extract_gene_level_data(coding_data, variant_type)
    
    # Merge with SFARI gene information
    merged_df = gene_df.merge(
        sfari_gene_df[['gene', 'score']],
        left_on='gene_name',
        right_on='gene',
        how='left'
    )
    
    # Flag SFARI genes (those with a score)
    merged_df['is_sfari'] = ~merged_df['score'].isna()
    
    # Group SFARI scores into categories
    merged_df['sfari_category'] = 'non-SFARI'
    merged_df.loc[merged_df['score'].isin(['1', '2']), 'sfari_category'] = 'high_confidence'
    merged_df.loc[merged_df['score'].isin(['3', '4']), 'sfari_category'] = 'medium_confidence'
    merged_df.loc[merged_df['score'].isin(['5', 'S']), 'sfari_category'] = 'low_confidence'
    
    # Get all clusters
    clusters = sorted(merged_df['cluster'].unique())
    
    # Analyze enrichment for each cluster and SFARI category
    for cluster in clusters:
        cluster_df = merged_df[merged_df['cluster'] == cluster]
        
        # Get total genes in this cluster
        total_genes = cluster_df['gene_name'].nunique()
        
        # Analyze each SFARI category
        for sfari_cat in ['high_confidence', 'medium_confidence', 'low_confidence', 'all_sfari']:
            if sfari_cat == 'all_sfari':
                cat_df = cluster_df[cluster_df['is_sfari']]
            else:
                cat_df = cluster_df[cluster_df['sfari_category'] == sfari_cat]
            
            # Count SFARI genes in this category
            sfari_genes = cat_df['gene_name'].nunique()
            
            # Skip if no genes in this category
            if sfari_genes == 0:
                continue
            
            # Calculate percentage
            percentage = (sfari_genes / total_genes) * 100 if total_genes > 0 else 0
            
            # Compare to other clusters
            other_clusters = [c for c in clusters if c != cluster]
            other_df = merged_df[merged_df['cluster'].isin(other_clusters)]
            
            if sfari_cat == 'all_sfari':
                other_cat_df = other_df[other_df['is_sfari']]
            else:
                other_cat_df = other_df[other_df['sfari_category'] == sfari_cat]
            
            # Count SFARI genes in other clusters
            other_sfari_genes = other_cat_df['gene_name'].nunique()
            other_total_genes = other_df['gene_name'].nunique()
            
            # Calculate percentage in other clusters
            other_percentage = (other_sfari_genes / other_total_genes) * 100 if other_total_genes > 0 else 0
            
            # Calculate enrichment
            enrichment = percentage / other_percentage if other_percentage > 0 else np.inf
            log2fc = np.log2(enrichment) if enrichment != np.inf and enrichment > 0 else np.nan
            
            # Contingency table for Fisher's exact test
            contingency = np.array([
                [sfari_genes, total_genes - sfari_genes],
                [other_sfari_genes, other_total_genes - other_sfari_genes]
            ])
            
            # Perform Fisher's exact test
            oddsratio, pvalue = stats.fisher_exact(contingency)
            
            # Add to results
            results.append({
                'cluster': cluster,
                'sfari_category': sfari_cat,
                'sfari_genes': sfari_genes,
                'total_genes': total_genes,
                'percentage': percentage,
                'other_sfari_genes': other_sfari_genes,
                'other_total_genes': other_total_genes,
                'other_percentage': other_percentage,
                'enrichment': enrichment,
                'log2fc': log2fc,
                'oddsratio': oddsratio,
                'pvalue': pvalue
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate multiple testing correction
    if not results_df.empty and 'pvalue' in results_df.columns:
        results_df['adjusted_pvalue'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
        results_df['significant'] = results_df['adjusted_pvalue'] < 0.05
    
    return results_df

def extract_gene_level_data(coding_data, variant_type="rare"):
    """
    Extract gene-level data from the coding variant data structure.
    
    Args:
        coding_data (dict): The loaded coding variant data by cluster
        variant_type (str): Type of variant to analyze ('rare' or 'common')
        
    Returns:
        pd.DataFrame: Long-format DataFrame with gene-level data
    """
    rows = []
    
    for cluster, cluster_data in coding_data.items():
        cluster_id = int(cluster)
        
        # Get gene-level data
        genes = cluster_data.get('genes', {}).get(variant_type, {})
        
        # Process each variant category
        categories = ['PTV', 'Missense', 'Synonymous']
        for category in categories:
            if category in genes:
                # Process each impact bin
                for impact_bin, gene_dict in genes[category].items():
                    # Process each gene
                    for gene_name, count in gene_dict.items():
                        rows.append({
                            'cluster': cluster_id,
                            'gene_name': gene_name,
                            'category': category,
                            'impact_bin': impact_bin,
                            'count': count,
                            'variant_type': variant_type
                        })
    
    return pd.DataFrame(rows)

def analyze_pathway_enrichment(gene_df, pathway_df, min_genes=5):
    """
    Analyze pathway enrichment for genes with variants in each cluster.
    
    Args:
        gene_df (pd.DataFrame): DataFrame with gene-level variant data
        pathway_df (pd.DataFrame): DataFrame with pathway information
        min_genes (int): Minimum number of genes for pathway analysis
        
    Returns:
        pd.DataFrame: Results of pathway enrichment analysis
    """
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests
    
    results = []
    
    # Get all clusters
    clusters = sorted(gene_df['cluster'].unique())
    
    # Get total gene universe (all genes with variants in any cluster)
    all_genes = set(gene_df['gene_name'].unique())
    total_genes = len(all_genes)
    
    # Group pathways by pathway ID
    pathway_genes = {}
    for _, row in pathway_df.iterrows():
        pathway_id = row['pathway_id']
        gene = row['gene']
        
        if pathway_id not in pathway_genes:
            pathway_genes[pathway_id] = {
                'name': row['pathway_name'],
                'genes': set()
            }
        
        pathway_genes[pathway_id]['genes'].add(gene)
    
    # Filter pathways with too few genes
    pathway_genes = {
        pathway_id: info 
        for pathway_id, info in pathway_genes.items() 
        if len(info['genes']) >= min_genes
    }
    
    # Analyze each cluster
    for cluster in clusters:
        # Get genes with variants in this cluster
        cluster_genes = set(gene_df[gene_df['cluster'] == cluster]['gene_name'].unique())
        cluster_gene_count = len(cluster_genes)
        
        # Skip if too few genes
        if cluster_gene_count < min_genes:
            continue
        
        # Analyze each pathway
        for pathway_id, info in pathway_genes.items():
            pathway_name = info['name']
            pathway_gene_set = info['genes']
            
            # Get overlap with this cluster
            overlap_genes = cluster_genes.intersection(pathway_gene_set)
            overlap_count = len(overlap_genes)
            
            # Skip if too few genes in overlap
            if overlap_count < 3:
                continue
            
            # Calculate statistics for enrichment
            cluster_non_pathway = cluster_gene_count - overlap_count
            pathway_non_cluster = len(pathway_gene_set.intersection(all_genes)) - overlap_count
            other = total_genes - cluster_non_pathway - pathway_non_cluster - overlap_count
            
            # Create contingency table
            contingency = np.array([
                [overlap_count, cluster_non_pathway],
                [pathway_non_cluster, other]
            ])
            
            # Perform Fisher's exact test
            oddsratio, pvalue = fisher_exact(contingency)
            
            # Calculate enrichment
            expected = (cluster_gene_count * len(pathway_gene_set)) / total_genes
            enrichment = overlap_count / expected if expected > 0 else np.inf
            
            # Add to results
            results.append({
                'cluster': cluster,
                'pathway_id': pathway_id,
                'pathway_name': pathway_name,
                'overlap_genes': overlap_count,
                'cluster_genes': cluster_gene_count,
                'pathway_genes': len(pathway_gene_set.intersection(all_genes)),
                'expected_overlap': expected,
                'enrichment': enrichment,
                'oddsratio': oddsratio,
                'pvalue': pvalue,
                'gene_list': ','.join(sorted(overlap_genes))
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate multiple testing correction
    if not results_df.empty and 'pvalue' in results_df.columns:
        # Apply FDR correction per cluster
        clusters = results_df['cluster'].unique()
        
        for cluster in clusters:
            cluster_mask = results_df['cluster'] == cluster
            if sum(cluster_mask) > 1:  # Only correct if multiple pathways
                results_df.loc[cluster_mask, 'adjusted_pvalue'] = multipletests(
                    results_df.loc[cluster_mask, 'pvalue'], method='fdr_bh'
                )[1]
        
        # Flag significant results
        results_df['significant'] = results_df.get('adjusted_pvalue', results_df['pvalue']) < 0.05
        
        # Sort by p-value
        results_df = results_df.sort_values(['cluster', 'pvalue'])
    
    return results_df

def compare_functional_impact_distributions(coding_data, variant_type="rare"):
    """
    Compare the distribution of functional impacts across clusters.
    
    Args:
        coding_data (dict): The loaded coding variant data by cluster
        variant_type (str): Type of variant to analyze ('rare' or 'common')
        
    Returns:
        pd.DataFrame: Results with proportions of each functional category
    """
    # Define the functional impact categories
    categories = ['PTV', 'Missense', 'Synonymous']
    
    # Initialize results
    rows = []
    
    # Process each cluster
    for cluster, cluster_data in coding_data.items():
        cluster_id = int(cluster)
        
        # Get variant counts for this variant type
        variant_counts = cluster_data.get('variant_counts', {}).get(variant_type, {})
        
        # Calculate total count across all categories
        total_count = 0
        category_counts = {}
        
        for category in categories:
            if category in variant_counts:
                # Sum counts across all impact bins
                category_count = sum(variant_counts[category].values())
                category_counts[category] = category_count
                total_count += category_count
        
        # Calculate proportions
        for category in categories:
            count = category_counts.get(category, 0)
            proportion = count / total_count if total_count > 0 else 0
            
            rows.append({
                'cluster': cluster_id,
                'category': category,
                'count': count,
                'total': total_count,
                'proportion': proportion,
                'variant_type': variant_type
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(rows)
    
    return results_df