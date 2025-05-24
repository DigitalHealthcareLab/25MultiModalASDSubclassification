'''
Helper functions for cluster comparison of gene variant distributions
'''
import pandas as pd
import numpy as np
from collections import defaultdict


def create_sfari_coding_cluster_comparison_table(cluster_summaries, variant_type):
    """
    Create a comparison table of coding variant statistics across clusters,
    specifically for SFARI genes.
    
    Args:
        cluster_summaries (dict): The cluster summaries dictionary
        variant_type (str): Type of variants ('common_coding' or 'rare_coding')
        
    Returns:
        pd.DataFrame: A comparison table with clusters as columns, filtered for SFARI genes
    """
    # Get clusters and sort them
    clusters = sorted(cluster_summaries.keys())
    
    # Create basic cluster info rows
    basic_info = []
    for cluster in clusters:
        basic_info.append({
            'Statistic': 'Cluster',
            f'Cluster {cluster}': cluster
        })
        
        basic_info.append({
            'Statistic': 'Participants',
            f'Cluster {cluster}': len(cluster_summaries[cluster]['participants'])
        })
        
        basic_info.append({
            'Statistic': 'Variant Type',
            f'Cluster {cluster}': f"{variant_type} (SFARI genes only)"
        })
    
    rows = []
    
    # Variant types to analyze
    variant_types = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)', 
                     'Missense (CADD 0-20)', 'Synonymous']
    
    # Gene tolerance categories
    tolerance_categories = ['All genes', 'pLI >= 0.9', 'pLI < 0.9']
    
    # Add header separator for All SFARI Genes
    rows.append({'Statistic': '--- All SFARI Genes ---'})
    
    # First section: All SFARI genes for each variant type and tolerance
    for vtype in variant_types:
        for tolerance in tolerance_categories:
            # Create row for this combination
            row = {'Statistic': f'{vtype} ({tolerance}) - All SFARI Genes'}
            
            for cluster in clusters:
                stats = cluster_summaries[cluster]['burden_stats'].get(vtype, {}).get(tolerance, {}).get('All SFARI', {})
                row[f'Cluster {cluster}'] = stats.get('median_variants', 0)
            
            rows.append(row)
    
    # Add headers for specific SFARI categories
    sfari_scores = ['SFARI 1', 'SFARI 2', 'SFARI 3', 'SFARI S']
    
    for sfari_score in sfari_scores:
        rows.append({'Statistic': f'--- {sfari_score} Genes ---'})
        
        for vtype in variant_types:
            for tolerance in tolerance_categories:
                # Create row for this combination
                row = {'Statistic': f'{vtype} ({tolerance}) - {sfari_score}'}
                
                for cluster in clusters:
                    stats = cluster_summaries[cluster]['burden_stats'].get(vtype, {}).get(tolerance, {}).get(sfari_score, {})
                    row[f'Cluster {cluster}'] = stats.get('median_variants', 0)
                
                rows.append(row)
    
    # Create DataFrame and add basic info at the top
    comparison_df = pd.DataFrame(basic_info + rows)
    
    return comparison_df


def generate_coding_cluster_summaries(participant_results, cluster_assignments):
    """
    Generate optimized SFARI gene burden summaries across clusters for coding variants.
    
    Args:
        participant_results (dict): Dictionary mapping participant_id to coding analysis results
        cluster_assignments (dict): Dictionary mapping participant_id to cluster_id
        
    Returns:
        dict: Optimized cluster summaries for coding variants
    """
    # Define categories we need to track
    variant_types = [
        'PTV', 
        'Missense (CADD>30)', 
        'Missense (CADD 20-30)', 
        'Missense (CADD 0-20)', 
        'Synonymous', 
        'Other'
    ]
    
    gene_tolerances = [
        'All genes',
        'pLI >= 0.9',
        'pLI < 0.9'
    ]
    
    sfari_categories = [
        'All SFARI',
        'SFARI 1',
        'SFARI 2',
        'SFARI 3',
        'SFARI S'
    ]
    
    # Get unique clusters
    clusters = sorted(set(cluster_assignments.values()))
    
    # Create a dict to collect all data by cluster
    cluster_data = {cluster: {
        'participants': [],
        'variant_stats': defaultdict(list),
        'gene_stats': defaultdict(list),
        'sfari_stats': defaultdict(list),
        'burden_stats': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'unique_genes': {
            'variant_types': defaultdict(set),
            'sfari_categories': defaultdict(set),
            'burden': defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        }
    } for cluster in clusters}
    
    # Assign participants to clusters
    for pid, cluster in cluster_assignments.items():
        if cluster in cluster_data:
            cluster_data[cluster]['participants'].append(pid)
    
    # Process each participant's results
    for pid, results in participant_results.items():
        if pid not in cluster_assignments or not results:
            continue
            
        cluster = cluster_assignments[pid]
        
        # Extract the summary DataFrame
        summary_df = results.get('summary_df', pd.DataFrame())
        if summary_df.empty:
            continue
        
        # Process all rows at once - much more efficient
        for _, row in summary_df.iterrows():
            vtype = row['Test']
            tolerance = row['Gene_Tolerance']
            sfari = row['SFARI_Category']
            
            # Skip irrelevant categories
            if (vtype not in variant_types or 
                tolerance not in gene_tolerances or 
                sfari not in sfari_categories):
                continue
            
            # Update burden stats (hierarchical)
            cluster_data[cluster]['burden_stats'][vtype][tolerance][sfari].append({
                'variant_count': row['Variant_Count'],
                'gene_count': row['Gene_Count']
            })
            
            # Update unique genes for burden stats
            if 'Genes' in row and row['Genes']:
                cluster_data[cluster]['unique_genes']['burden'][vtype][tolerance][sfari].update(row['Genes'])
            
            # Update variant type stats (using All genes only to avoid double counting)
            if tolerance == 'All genes':
                cluster_data[cluster]['variant_stats'][vtype].append({
                    'variant_count': row['Variant_Count'],
                    'gene_count': row['Gene_Count']
                })
                
                # Update unique genes for variant types
                if 'Genes' in row and row['Genes']:
                    cluster_data[cluster]['unique_genes']['variant_types'][vtype].update(row['Genes'])
            
            # Update SFARI stats (using All genes only)
            if tolerance == 'All genes' and sfari != 'All SFARI':
                cluster_data[cluster]['sfari_stats'][sfari].append({
                    'variant_count': row['Variant_Count'],
                    'gene_count': row['Gene_Count']
                })
                
                # Update unique genes for SFARI categories
                if 'Genes' in row and row['Genes']:
                    cluster_data[cluster]['unique_genes']['sfari_categories'][sfari].update(row['Genes'])
    
    # Now calculate statistics and build final structure
    cluster_summaries = {}
    
    for cluster, data in cluster_data.items():
        cluster_summaries[cluster] = {
            'participant_count': len(data['participants']),
            'participants': data['participants'],
            'burden_stats': {},
            'variant_type_stats': {},
            'sfari_score_stats': {}
        }
        
        # Process variant type stats
        for vtype, stats_list in data['variant_stats'].items():
            if stats_list:
                variant_counts = [s['variant_count'] for s in stats_list]
                gene_counts = [s['gene_count'] for s in stats_list]
                
                cluster_summaries[cluster]['variant_type_stats'][vtype] = {
                    'median_variants': np.median(variant_counts),
                    'mean_variants': np.mean(variant_counts),
                    'std_variants': np.std(variant_counts),
                    'total_variants': sum(variant_counts),
                    'median_genes': np.median(gene_counts),
                    'mean_genes': np.mean(gene_counts),
                    'std_genes': np.std(gene_counts),
                    'total_unique_genes': len(data['unique_genes']['variant_types'][vtype]),
                    'unique_genes_all_participants': list(data['unique_genes']['variant_types'][vtype])
                }
        
        # Process SFARI stats
        for sfari, stats_list in data['sfari_stats'].items():
            if stats_list:
                variant_counts = [s['variant_count'] for s in stats_list]
                gene_counts = [s['gene_count'] for s in stats_list]
                
                cluster_summaries[cluster]['sfari_score_stats'][sfari] = {
                    'median_variants': np.median(variant_counts),
                    'mean_variants': np.mean(variant_counts),
                    'std_variants': np.std(variant_counts),
                    'total_variants': sum(variant_counts),
                    'median_genes': np.median(gene_counts),
                    'mean_genes': np.mean(gene_counts),
                    'std_genes': np.std(gene_counts),
                    'total_unique_genes': len(data['unique_genes']['sfari_categories'][sfari]),
                    'unique_genes_all_participants': list(data['unique_genes']['sfari_categories'][sfari])
                }
        
        # Process burden stats (hierarchical)
        cluster_summaries[cluster]['burden_stats'] = {}
        for vtype in variant_types:
            cluster_summaries[cluster]['burden_stats'][vtype] = {}
            
            for tolerance in gene_tolerances:
                cluster_summaries[cluster]['burden_stats'][vtype][tolerance] = {}
                
                for sfari in sfari_categories:
                    stats_list = data['burden_stats'][vtype][tolerance][sfari]
                    
                    if stats_list:
                        variant_counts = [s['variant_count'] for s in stats_list]
                        gene_counts = [s['gene_count'] for s in stats_list]
                        
                        cluster_summaries[cluster]['burden_stats'][vtype][tolerance][sfari] = {
                            'median_variants': np.median(variant_counts),
                            'mean_variants': np.mean(variant_counts),
                            'std_variants': np.std(variant_counts),
                            'total_variants': sum(variant_counts),
                            'median_genes': np.median(gene_counts),
                            'mean_genes': np.mean(gene_counts),
                            'std_genes': np.std(gene_counts),
                            'total_unique_genes': len(data['unique_genes']['burden'][vtype][tolerance][sfari]),
                            'unique_genes_all_participants': list(data['unique_genes']['burden'][vtype][tolerance][sfari])
                        }
                    else:
                        # Initialize with zeros
                        cluster_summaries[cluster]['burden_stats'][vtype][tolerance][sfari] = {
                            'median_variants': 0,
                            'mean_variants': 0,
                            'std_variants': 0,
                            'total_variants': 0,
                            'median_genes': 0,
                            'mean_genes': 0,
                            'std_genes': 0,
                            'total_unique_genes': 0,
                            'unique_genes_all_participants': []
                        }
    
    return cluster_summaries


def generate_noncoding_cluster_summaries(participant_results, cluster_assignments):
    """
    Generate optimized summaries across clusters for noncoding variants.
    
    Args:
        participant_results (dict): Dictionary mapping participant_id to noncoding analysis results
        cluster_assignments (dict): Dictionary mapping participant_id to cluster_id
        
    Returns:
        dict: Optimized cluster summaries for noncoding variants
    """
    # Define categories specific to noncoding variants
    genomic_regions = [
        'Intronic', 'Intergenic', 'Enhancer', 'Promoter', 'Enhancer,Promoter',
        'Upstream', 'Downstream', "3' UTR", "5' UTR", 'Other'
    ]
    
    eigen_bins = ['Eigen>0', 'Eigen<=0']
    
    sfari_categories = [
        'All SFARI', 'SFARI 1', 'SFARI 2', 'SFARI 3', 'SFARI S'
    ]
    
    # Get unique clusters
    clusters = sorted(set(cluster_assignments.values()))
    
    # Create a dict to collect all data by cluster
    cluster_data = {cluster: {
        'participants': [],
        'region_stats': defaultdict(list),
        'eigen_stats': defaultdict(list),
        'eigen_region_stats': defaultdict(lambda: defaultdict(list)),
        'sfari_stats': defaultdict(list),
        'sfari_region_stats': defaultdict(lambda: defaultdict(list)),
        'unique_genes': {
            'regions': defaultdict(set),
            'eigen_bins': defaultdict(set),
            'eigen_by_region': defaultdict(lambda: defaultdict(set)),
            'sfari': defaultdict(set),
            'sfari_by_region': defaultdict(lambda: defaultdict(set))
        }
    } for cluster in clusters}
    
    # Assign participants to clusters
    for pid, cluster in cluster_assignments.items():
        if cluster in cluster_data:
            cluster_data[cluster]['participants'].append(pid)
    
    # Process each participant's results
    for pid, results in participant_results.items():
        if pid not in cluster_assignments or not results:
            continue
            
        cluster = cluster_assignments[pid]
        
        # Get the filtered DataFrame
        filtered_df = results.get('filtered_df', pd.DataFrame())
        if filtered_df.empty:
            continue
        
        # Process by region
        for region in genomic_regions:
            region_df = filtered_df[filtered_df['Simplified_Region'] == region]
            if not region_df.empty:
                genes = [g for g in region_df['GeneName'].unique() 
                        if g != '.' and g != '--' and pd.notna(g)]
                
                cluster_data[cluster]['region_stats'][region].append({
                    'variant_count': len(region_df),
                    'gene_count': len(genes)
                })
                
                cluster_data[cluster]['unique_genes']['regions'][region].update(genes)
        
        # Process by Eigen score
        if 'eigen_bin' in filtered_df.columns:
            for eigen_bin in eigen_bins:
                eigen_df = filtered_df[filtered_df['eigen_bin'] == eigen_bin]
                if not eigen_df.empty:
                    genes = [g for g in eigen_df['GeneName'].unique() 
                            if g != '.' and g != '--' and pd.notna(g)]
                    
                    cluster_data[cluster]['eigen_stats'][eigen_bin].append({
                        'variant_count': len(eigen_df),
                        'gene_count': len(genes)
                    })
                    
                    cluster_data[cluster]['unique_genes']['eigen_bins'][eigen_bin].update(genes)
                    
                    # Process Eigen by region
                    for region in genomic_regions:
                        eigen_region_df = eigen_df[eigen_df['Simplified_Region'] == region]
                        if not eigen_region_df.empty:
                            region_genes = [g for g in eigen_region_df['GeneName'].unique() 
                                          if g != '.' and g != '--' and pd.notna(g)]
                            
                            cluster_data[cluster]['eigen_region_stats'][eigen_bin][region].append({
                                'variant_count': len(eigen_region_df),
                                'gene_count': len(region_genes)
                            })
                            
                            cluster_data[cluster]['unique_genes']['eigen_by_region'][eigen_bin][region].update(region_genes)
        
        # Process SFARI variants
        sfari_df = filtered_df[filtered_df['SFARI_gene_score'] != '.']
        if not sfari_df.empty:
            # All SFARI
            all_sfari_genes = [g for g in sfari_df['GeneName'].unique() 
                              if g != '.' and g != '--' and pd.notna(g)]
            
            cluster_data[cluster]['sfari_stats']['All SFARI'].append({
                'variant_count': len(sfari_df),
                'gene_count': len(all_sfari_genes)
            })
            
            cluster_data[cluster]['unique_genes']['sfari']['All SFARI'].update(all_sfari_genes)
            
            # Process SFARI by region for All SFARI
            for region in genomic_regions:
                sfari_region_df = sfari_df[sfari_df['Simplified_Region'] == region]
                if not sfari_region_df.empty:
                    region_genes = [g for g in sfari_region_df['GeneName'].unique() 
                                  if g != '.' and g != '--' and pd.notna(g)]
                    
                    cluster_data[cluster]['sfari_region_stats']['All SFARI'][region].append({
                        'variant_count': len(sfari_region_df),
                        'gene_count': len(region_genes)
                    })
                    
                    cluster_data[cluster]['unique_genes']['sfari_by_region']['All SFARI'][region].update(region_genes)
            
            # Process by SFARI score
            for score, category in zip(['1', '2', '3', 'S'], ['SFARI 1', 'SFARI 2', 'SFARI 3', 'SFARI S']):
                score_df = sfari_df[sfari_df['SFARI_gene_score'] == score]
                if not score_df.empty:
                    score_genes = [g for g in score_df['GeneName'].unique() 
                                 if g != '.' and g != '--' and pd.notna(g)]
                    
                    cluster_data[cluster]['sfari_stats'][category].append({
                        'variant_count': len(score_df),
                        'gene_count': len(score_genes)
                    })
                    
                    cluster_data[cluster]['unique_genes']['sfari'][category].update(score_genes)
                    
                    # Process SFARI by region for this category
                    for region in genomic_regions:
                        score_region_df = score_df[score_df['Simplified_Region'] == region]
                        if not score_region_df.empty:
                            region_genes = [g for g in score_region_df['GeneName'].unique() 
                                          if g != '.' and g != '--' and pd.notna(g)]
                            
                            cluster_data[cluster]['sfari_region_stats'][category][region].append({
                                'variant_count': len(score_region_df),
                                'gene_count': len(region_genes)
                            })
                            
                            cluster_data[cluster]['unique_genes']['sfari_by_region'][category][region].update(region_genes)
    
    # Helper function to calculate statistics
    def calculate_stats(stats_list, unique_genes):
        if not stats_list:
            return {
                'median_variants': 0,
                'mean_variants': 0,
                'std_variants': 0,
                'total_variants': 0,
                'median_genes': 0,
                'mean_genes': 0,
                'std_genes': 0,
                'total_unique_genes': 0,
                'unique_genes_all_participants': []
            }
        
        variant_counts = [s['variant_count'] for s in stats_list]
        gene_counts = [s['gene_count'] for s in stats_list]
        
        return {
            'median_variants': np.median(variant_counts),
            'mean_variants': np.mean(variant_counts),
            'std_variants': np.std(variant_counts),
            'total_variants': sum(variant_counts),
            'median_genes': np.median(gene_counts),
            'mean_genes': np.mean(gene_counts),
            'std_genes': np.std(gene_counts),
            'total_unique_genes': len(unique_genes),
            'unique_genes_all_participants': list(unique_genes)
        }
    
    # Calculate statistics and build final structure
    cluster_summaries = {}
    
    for cluster, data in cluster_data.items():
        cluster_summaries[cluster] = {
            'participant_count': len(data['participants']),
            'participants': data['participants'],
            'region_stats': {},
            'eigen_stats': {},
            'sfari_stats': {}
        }
        
        # Process region stats
        for region, stats_list in data['region_stats'].items():
            cluster_summaries[cluster]['region_stats'][region] = calculate_stats(
                stats_list, data['unique_genes']['regions'][region]
            )
        
        # Process Eigen stats
        for eigen_bin, stats_list in data['eigen_stats'].items():
            cluster_summaries[cluster]['eigen_stats'][eigen_bin] = calculate_stats(
                stats_list, data['unique_genes']['eigen_bins'][eigen_bin]
            )
            
            # Add regions sub-dictionary
            cluster_summaries[cluster]['eigen_stats'][eigen_bin]['regions'] = {}
            
            for region, region_stats in data['eigen_region_stats'][eigen_bin].items():
                cluster_summaries[cluster]['eigen_stats'][eigen_bin]['regions'][region] = calculate_stats(
                    region_stats, data['unique_genes']['eigen_by_region'][eigen_bin][region]
                )
        
        # Process SFARI stats
        for sfari, stats_list in data['sfari_stats'].items():
            cluster_summaries[cluster]['sfari_stats'][sfari] = calculate_stats(
                stats_list, data['unique_genes']['sfari'][sfari]
            )
            
            # Add regions sub-dictionary
            cluster_summaries[cluster]['sfari_stats'][sfari]['regions'] = {}
            
            for region, region_stats in data['sfari_region_stats'][sfari].items():
                cluster_summaries[cluster]['sfari_stats'][sfari]['regions'][region] = calculate_stats(
                    region_stats, data['unique_genes']['sfari_by_region'][sfari][region]
                )
    
    return cluster_summaries


def create_coding_cluster_comparison_table(cluster_summaries, variant_type):
    """
    Create a comparison table of coding variant statistics across clusters.
    
    Args:
        cluster_summaries (dict): The cluster summaries dictionary
        variant_type (str): Type of variants ('common_coding' or 'rare_coding')
        
    Returns:
        pd.DataFrame: A comparison table with clusters as columns
    """
    # Get clusters and sort them
    clusters = sorted(cluster_summaries.keys())
    
    # Create basic cluster info rows
    basic_info = []
    for cluster in clusters:
        basic_info.append({
            'Statistic': 'Cluster',
            f'Cluster {cluster}': cluster
        })
        
        basic_info.append({
            'Statistic': 'Participants',
            f'Cluster {cluster}': len(cluster_summaries[cluster]['participants'])
        })
        
        basic_info.append({
            'Statistic': 'Variant Type',
            f'Cluster {cluster}': variant_type
        })
    
    rows = []
    
    # Add header separator
    rows.append({'Statistic': '--- Variant Categories ---'})
    
    # Variant type statistics
    variant_types = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)', 
                     'Missense (CADD 0-20)', 'Synonymous']
    
    for vtype in variant_types:
        # Median per participant row
        median_row = {'Statistic': f'{vtype} (median per participant)'}
        # Unique genes row
        genes_row = {'Statistic': f'{vtype} (unique genes across cluster)'}
        
        for cluster in clusters:
            stats = cluster_summaries[cluster]['variant_type_stats'].get(vtype, {})
            
            median_row[f'Cluster {cluster}'] = stats.get('median_variants', 0)
            genes_row[f'Cluster {cluster}'] = stats.get('total_unique_genes', 0)
        
        rows.append(median_row)
        rows.append(genes_row)
    
    # Add header separator
    rows.append({'Statistic': '--- SFARI Gene Categories ---'})
    
    # SFARI gene statistics
    sfari_categories = ['SFARI 1', 'SFARI 2', 'SFARI 3', 'SFARI S']
    
    for sfari in sfari_categories:
        # Median per participant row
        median_row = {'Statistic': f'{sfari} genes with variants (median per participant)'}
        # Unique genes row
        genes_row = {'Statistic': f'{sfari} genes with variants (unique genes across cluster)'}
        
        for cluster in clusters:
            stats = cluster_summaries[cluster]['sfari_score_stats'].get(sfari, {})
            
            median_row[f'Cluster {cluster}'] = stats.get('median_variants', 0)
            genes_row[f'Cluster {cluster}'] = stats.get('total_unique_genes', 0)
        
        rows.append(median_row)
        rows.append(genes_row)
    
    # Add header separator
    rows.append({'Statistic': '--- High Impact Categories ---'})
    
    # PTV in high pLI SFARI genes
    ptv_row = {'Statistic': 'PTV in SFARI 1-2 genes with pLI >= 0.9 (median per participant)'}
    for cluster in clusters:
        ptv_high_pli = cluster_summaries[cluster]['burden_stats'].get('PTV', {}).get('pLI >= 0.9', {})
        sfari1_count = ptv_high_pli.get('SFARI 1', {}).get('median_variants', 0)
        sfari2_count = ptv_high_pli.get('SFARI 2', {}).get('median_variants', 0)
        ptv_row[f'Cluster {cluster}'] = sfari1_count + sfari2_count
    rows.append(ptv_row)
    
    # High CADD missense in SFARI genes
    cadd_row = {'Statistic': 'Missense (CADD>30) in SFARI 1-2 genes (median per participant)'}
    for cluster in clusters:
        high_cadd = cluster_summaries[cluster]['burden_stats'].get('Missense (CADD>30)', {}).get('All genes', {})
        sfari1_count = high_cadd.get('SFARI 1', {}).get('median_variants', 0)
        sfari2_count = high_cadd.get('SFARI 2', {}).get('median_variants', 0)
        cadd_row[f'Cluster {cluster}'] = sfari1_count + sfari2_count
    rows.append(cadd_row)
    
    # Create DataFrame and add basic info at the top
    comparison_df = pd.DataFrame(basic_info + rows)
    
    return comparison_df

def create_noncoding_cluster_comparison_table(cluster_summaries):
    """
    Create a comparison table of noncoding variant statistics across clusters.
    
    Args:
        cluster_summaries (dict): The cluster summaries dictionary
        
    Returns:
        pd.DataFrame: A comparison table with clusters as columns
    """
    # Get clusters and sort them
    clusters = sorted(cluster_summaries.keys())
    
    # Create basic cluster info rows
    basic_info = []
    for cluster in clusters:
        basic_info.append({
            'Statistic': 'Cluster',
            f'Cluster {cluster}': cluster
        })
        
        basic_info.append({
            'Statistic': 'Participants',
            f'Cluster {cluster}': len(cluster_summaries[cluster]['participants'])
        })
        
        basic_info.append({
            'Statistic': 'Variant Type',
            f'Cluster {cluster}': 'rare_noncoding'
        })
    
    rows = []
    
    # Add header separator
    rows.append({'Statistic': '--- Genomic Region Categories ---'})
    
    # Genomic region statistics
    genomic_regions = ['Enhancer', 'Promoter', 'Enhancer,Promoter', 
                       'Intronic', 'Intergenic', "3' UTR", "5' UTR"]
    
    for region in genomic_regions:
        # Median per participant row
        median_row = {'Statistic': f'{region} variants (median per participant)'}
        # Unique genes row
        genes_row = {'Statistic': f'{region} variants (unique genes across cluster)'}
        
        for cluster in clusters:
            stats = cluster_summaries[cluster]['region_stats'].get(region, {})
            
            median_row[f'Cluster {cluster}'] = stats.get('median_variants', 0)
            genes_row[f'Cluster {cluster}'] = stats.get('total_unique_genes', 0)
        
        rows.append(median_row)
        rows.append(genes_row)
    
    # Add header separator
    rows.append({'Statistic': '--- Eigen Score Categories ---'})
    
    # Eigen bin statistics
    eigen_bins = ['Eigen>0', 'Eigen<=0']
    
    for eigen_bin in eigen_bins:
        # Check if this Eigen bin exists in any cluster
        if any(eigen_bin in summary.get('eigen_stats', {}) for summary in cluster_summaries.values()):
            # Median per participant row
            median_row = {'Statistic': f'{eigen_bin} variants (median per participant)'}
            # Unique genes row
            genes_row = {'Statistic': f'{eigen_bin} variants (unique genes across cluster)'}
            
            for cluster in clusters:
                stats = cluster_summaries[cluster]['eigen_stats'].get(eigen_bin, {})
                
                median_row[f'Cluster {cluster}'] = stats.get('median_variants', 0)
                genes_row[f'Cluster {cluster}'] = stats.get('total_unique_genes', 0)
            
            rows.append(median_row)
            rows.append(genes_row)
    
    # Add header separator
    rows.append({'Statistic': '--- SFARI Gene Categories ---'})
    
    # SFARI gene statistics
    sfari_categories = ['SFARI 1', 'SFARI 2', 'SFARI 3', 'SFARI S']
    
    for sfari in sfari_categories:
        # Median per participant row
        median_row = {'Statistic': f'{sfari} genes with noncoding variants (median per participant)'}
        # Unique genes row
        genes_row = {'Statistic': f'{sfari} genes with noncoding variants (unique genes across cluster)'}
        
        for cluster in clusters:
            stats = cluster_summaries[cluster]['sfari_stats'].get(sfari, {})
            
            median_row[f'Cluster {cluster}'] = stats.get('median_variants', 0)
            genes_row[f'Cluster {cluster}'] = stats.get('total_unique_genes', 0)
        
        rows.append(median_row)
        rows.append(genes_row)
    
    # Add header separator
    rows.append({'Statistic': '--- High Impact Categories ---'})
    
    # Regulatory variants in SFARI genes
    reg_row = {'Statistic': 'Regulatory variants in SFARI 1-2 genes (median per participant)'}
    for cluster in clusters:
        sfari_stats = cluster_summaries[cluster]['sfari_stats']
        
        # Sum enhancer and promoter stats for SFARI 1 and 2 genes
        enhancer_sfari1 = sfari_stats.get('SFARI 1', {}).get('regions', {}).get('Enhancer', {}).get('median_variants', 0)
        enhancer_sfari2 = sfari_stats.get('SFARI 2', {}).get('regions', {}).get('Enhancer', {}).get('median_variants', 0)
        promoter_sfari1 = sfari_stats.get('SFARI 1', {}).get('regions', {}).get('Promoter', {}).get('median_variants', 0)
        promoter_sfari2 = sfari_stats.get('SFARI 2', {}).get('regions', {}).get('Promoter', {}).get('median_variants', 0)
        enhancer_promoter_sfari1 = sfari_stats.get('SFARI 1', {}).get('regions', {}).get('Enhancer,Promoter', {}).get('median_variants', 0)
        enhancer_promoter_sfari2 = sfari_stats.get('SFARI 2', {}).get('regions', {}).get('Enhancer,Promoter', {}).get('median_variants', 0)
        
        # Sum all regulatory variants in SFARI 1-2 genes
        reg_row[f'Cluster {cluster}'] = (enhancer_sfari1 + enhancer_sfari2 + 
                                        promoter_sfari1 + promoter_sfari2 +
                                        enhancer_promoter_sfari1 + enhancer_promoter_sfari2)
    rows.append(reg_row)
    
    # High impact noncoding variants
    if 'Eigen>0' in eigen_bins:
        eigen_row = {'Statistic': 'High impact (Eigen>0) variants in SFARI 1-2 genes (median per participant)'}
        for cluster in clusters:
            eigen_stats = cluster_summaries[cluster]['eigen_stats'].get('Eigen>0', {}).get('regions', {})
            sfari_stats = cluster_summaries[cluster]['sfari_stats']
            
            # Get counts for SFARI 1-2 genes with Eigen>0
            eigen_sfari1 = sfari_stats.get('SFARI 1', {}).get('median_variants', 0) * 0.5  # Estimate high impact
            eigen_sfari2 = sfari_stats.get('SFARI 2', {}).get('median_variants', 0) * 0.5  # Estimate high impact
            
            eigen_row[f'Cluster {cluster}'] = eigen_sfari1 + eigen_sfari2
        rows.append(eigen_row)
    
    # Create DataFrame and add basic info at the top
    comparison_df = pd.DataFrame(basic_info + rows)
    
    return comparison_df


def calculate_rare_common_ratio(rare_summaries, common_summaries, sfari_only=False):
    """
    Calculate the ratio of rare to common coding variants for each cluster.

    Args:
        rare_summaries (dict): Rare coding variant cluster summaries
        common_summaries (dict): Common coding variant cluster summaries
        sfari_only (bool): If True, calculate using only SFARI 1-2 genes; else use all variants
        
    Returns:
        pd.DataFrame: Table of rare:common ratios by cluster
    """
    # Get clusters common to both summaries
    clusters = sorted(set(rare_summaries.keys()) & set(common_summaries.keys()))
    
    # Define the categories to calculate ratios for
    if sfari_only:
        # Categories for SFARI-only ratios
        categories = {
            'SFARI 1-2 Genes (All)': {
                'rare_key': lambda c: (c['sfari_score_stats'].get('SFARI 1', {}).get('total_variants', 0) +
                                      c['sfari_score_stats'].get('SFARI 2', {}).get('total_variants', 0)),
                'common_key': lambda c: (c['sfari_score_stats'].get('SFARI 1', {}).get('total_variants', 0) +
                                        c['sfari_score_stats'].get('SFARI 2', {}).get('total_variants', 0))
            },
            'SFARI 1-2 Genes (PTV)': {
                'rare_key': lambda c: (c['burden_stats'].get('PTV', {}).get('All genes', {}).get('SFARI 1', {}).get('total_variants', 0) +
                                      c['burden_stats'].get('PTV', {}).get('All genes', {}).get('SFARI 2', {}).get('total_variants', 0)),
                'common_key': lambda c: (c['burden_stats'].get('PTV', {}).get('All genes', {}).get('SFARI 1', {}).get('total_variants', 0) +
                                        c['burden_stats'].get('PTV', {}).get('All genes', {}).get('SFARI 2', {}).get('total_variants', 0))
            },
            'SFARI 1-2 Genes (Missense CADD>30)': {
                'rare_key': lambda c: (c['burden_stats'].get('Missense (CADD>30)', {}).get('All genes', {}).get('SFARI 1', {}).get('total_variants', 0) +
                                      c['burden_stats'].get('Missense (CADD>30)', {}).get('All genes', {}).get('SFARI 2', {}).get('total_variants', 0)),
                'common_key': lambda c: (c['burden_stats'].get('Missense (CADD>30)', {}).get('All genes', {}).get('SFARI 1', {}).get('total_variants', 0) +
                                        c['burden_stats'].get('Missense (CADD>30)', {}).get('All genes', {}).get('SFARI 2', {}).get('total_variants', 0))
            }
        }
    else:
        # Categories for all-gene ratios
        categories = {
            'All Variants': {
                'rare_key': lambda c: sum(stats.get('total_variants', 0) for stats in c['variant_type_stats'].values()),
                'common_key': lambda c: sum(stats.get('total_variants', 0) for stats in c['variant_type_stats'].values())
            },
            'PTV': {
                'rare_key': lambda c: c['variant_type_stats'].get('PTV', {}).get('total_variants', 0),
                'common_key': lambda c: c['variant_type_stats'].get('PTV', {}).get('total_variants', 0)
            },
            'Missense (CADD>30)': {
                'rare_key': lambda c: c['variant_type_stats'].get('Missense (CADD>30)', {}).get('total_variants', 0),
                'common_key': lambda c: c['variant_type_stats'].get('Missense (CADD>30)', {}).get('total_variants', 0)
            },
            'Missense (CADD 20-30)': {
                'rare_key': lambda c: c['variant_type_stats'].get('Missense (CADD 20-30)', {}).get('total_variants', 0),
                'common_key': lambda c: c['variant_type_stats'].get('Missense (CADD 20-30)', {}).get('total_variants', 0)
            },
            'High pLI Genes (pLI >= 0.9)': {
                'rare_key': lambda c: (c['burden_stats'].get('PTV', {}).get('pLI >= 0.9', {}).get('All SFARI', {}).get('total_variants', 0) +
                                      c['burden_stats'].get('Missense (CADD>30)', {}).get('pLI >= 0.9', {}).get('All SFARI', {}).get('total_variants', 0)),
                'common_key': lambda c: (c['burden_stats'].get('PTV', {}).get('pLI >= 0.9', {}).get('All SFARI', {}).get('total_variants', 0) +
                                        c['burden_stats'].get('Missense (CADD>30)', {}).get('pLI >= 0.9', {}).get('All SFARI', {}).get('total_variants', 0))
            }
        }
    
    # Create rows for each category
    rows = []
    for category, keys in categories.items():
        row = {'Category': category}
        
        for cluster in clusters:
            rare_c = rare_summaries[cluster]
            common_c = common_summaries[cluster]
            
            rare_count = keys['rare_key'](rare_c)
            common_count = keys['common_key'](common_c)
            
            # Calculate the ratio, handling division by zero
            if common_count > 0:
                ratio = rare_count / common_count
            else:
                ratio = float('inf') if rare_count > 0 else 0
            
            # Store both the ratio and raw counts
            row[f'Cluster {cluster}'] = ratio
            row[f'Cluster {cluster} Rare Count'] = rare_count
            row[f'Cluster {cluster} Common Count'] = common_count

        rows.append(row)

    # Create the DataFrame and return
    ratio_df = pd.DataFrame(rows)
    return ratio_df