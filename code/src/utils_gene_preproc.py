'''
Helper utilities for gene variant preprocessing and SFARI burden analysis
'''
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def analyze_sfari_variants_by_burden(sample_id, variant_type="3_1_Common_Coding", categorized_df=None, pli_threshold=0.9):
    """
    Analyzes variants with a focus on SFARI genes, using the burden categories 
    (PTV, Missense CADD>30, etc.) and pLI scores.
    
    Optimized version with reduced dataframe operations and improved memory usage.
    """
    if categorized_df is None:
        raise ValueError("Pre-categorized dataframe must be provided")

    # Make a copy to avoid modifying original
    df = categorized_df.copy()
    
    # Convert PLI to numeric once
    df['PLI'] = pd.to_numeric(df['PLI'], errors='coerce')
    df['SFARI_Numeric'] = df['SFARI_gene_score'].map({'1': 1, '2': 2, '3': 3, 'S': 4, '.': None})

    # Add gene tolerance column
    df['Gene_Tolerance'] = np.where(df['PLI'] >= pli_threshold, 
                                     f'pLI >= {pli_threshold}', 
                                     f'pLI < {pli_threshold}')

    # Create test column using vectorized operations where possible
    conditions = [
        (df['Variant_Category'] == 'Missense') & (df['cadd_bin'] == 'CADD>30'),
        (df['Variant_Category'] == 'Missense') & (df['cadd_bin'] == 'CADD20-30'),
        (df['Variant_Category'] == 'Missense') & (df['cadd_bin'] == 'CADD<20')
    ]
    choices = [
        'Missense (CADD>30)',
        'Missense (CADD 20-30)',
        'Missense (CADD 0-20)'
    ]
    df['Test'] = np.select(conditions, choices, default=df['Variant_Category'])

    # Filter for SFARI variants once
    sfari_variants = df[df['SFARI_gene_score'] != '.']
    
    # Define categories
    test_categories = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)', 
                       'Missense (CADD 0-20)', 'Synonymous', 'Other']
    tolerance_categories = ['All genes', f'pLI >= {pli_threshold}', f'pLI < {pli_threshold}']
    sfari_categories = {'All SFARI': None, 'SFARI 1': '1', 'SFARI 2': '2', 'SFARI 3': '3', 'SFARI S': 'S'}
    
    # Create SFARI map for grouping
    sfari_map = {s: 'All SFARI' if s in ['1', '2', '3', 'S'] else None for s in sfari_variants['SFARI_gene_score'].unique()}
    for cat, val in sfari_categories.items():
        if val:
            sfari_map[val] = cat
    
    # Add mapped categories for groupby
    sfari_variants['SFARI_Category'] = sfari_variants['SFARI_gene_score'].map(sfari_map)
    
    # Create a flattened results list with a single groupby operation
    flattened_results = []
    genes_by_group = {}
    
    # Add "All genes" group
    for test, test_group in sfari_variants.groupby('Test'):
        if test not in test_categories:
            continue
            
        for sfari_cat, sfari_group in test_group.groupby('SFARI_Category'):
            if sfari_cat not in sfari_categories:
                continue
                
            # Get unique genes (excluding invalid values)
            genes = [g for g in sfari_group['GeneName'].unique() 
                     if g not in ['.', '--'] and pd.notna(g)]
            
            # Store for "All genes" tolerance
            group_key = (test, 'All genes', sfari_cat)
            genes_by_group[group_key] = genes
            
            flattened_results.append({
                'Test': test,
                'Gene_Tolerance': 'All genes',
                'SFARI_Category': sfari_cat,
                'Variant_Count': len(sfari_group),
                'Gene_Count': len(genes),
                'Genes': genes
            })
            
            # Now add pLI-specific tolerance groups
            for tol in [f'pLI >= {pli_threshold}', f'pLI < {pli_threshold}']:
                if tol == f'pLI >= {pli_threshold}':
                    tol_group = sfari_group[sfari_group['PLI'] >= pli_threshold]
                else:
                    tol_group = sfari_group[sfari_group['PLI'] < pli_threshold]
                
                genes = [g for g in tol_group['GeneName'].unique() 
                         if g not in ['.', '--'] and pd.notna(g)]
                
                group_key = (test, tol, sfari_cat)
                genes_by_group[group_key] = genes
                
                flattened_results.append({
                    'Test': test,
                    'Gene_Tolerance': tol,
                    'SFARI_Category': sfari_cat,
                    'Variant_Count': len(tol_group),
                    'Gene_Count': len(genes),
                    'Genes': genes
                })
    
    summary_df = pd.DataFrame(flattened_results)
    
    # Process high-priority genes more efficiently
    high_priority_genes = defaultdict(list)
    
    # Create index lookup for faster filtering
    high_priority_keys = []
    
    # PTV or Missense (CADD>30) in SFARI 1/2
    high_priority_keys.extend([
        (test, 'All genes', sfari_cat)
        for test in ['PTV', 'Missense (CADD>30)']
        for sfari_cat in ['SFARI 1', 'SFARI 2']
    ])
    
    # Any variant in SFARI 1/2 with high pLI
    high_priority_keys.extend([
        (test, f'pLI >= {pli_threshold}', sfari_cat)
        for test in test_categories
        for sfari_cat in ['SFARI 1', 'SFARI 2']
    ])
    
    # Process high priority genes
    for key in high_priority_keys:
        test, tol, sfari_cat = key
        if key in genes_by_group:
            for gene in genes_by_group[key]:
                if tol == f'pLI >= {pli_threshold}':
                    high_priority_genes[gene].append(f"{test} in {sfari_cat} with high pLI")
                else:
                    high_priority_genes[gene].append(f"{test} in {sfari_cat}")
    
    # Process gene details for high-priority genes efficiently
    gene_variants_map = {}
    if high_priority_genes:
        # Filter once for all high priority genes
        high_priority_gene_set = set(high_priority_genes.keys())
        all_variants = sfari_variants[sfari_variants['GeneName'].isin(high_priority_gene_set)]
        
        # Group by gene name
        for gene, group in all_variants.groupby('GeneName'):
            gene_variants_map[gene] = group
    
    gene_details = {}
    for gene in high_priority_genes.keys():
        if gene in gene_variants_map:
            gene_group = gene_variants_map[gene]
            gene_details[gene] = {
                'sfari_score': gene_group['SFARI_gene_score'].iloc[0] if not gene_group.empty else 'Unknown',
                'pli': gene_group['PLI'].iloc[0] if not gene_group.empty else 'Unknown',
                'variants': [{
                    'rsID': var['rsID'],
                    'REF:ALT': var['REF:ALT'],
                    'Effect': var['Variant_Effect'],
                    'Category': var['Variant_Category'],
                    'CADD': var['CADD_phred_hg19']
                } for _, var in gene_group.iterrows()]
            }
    
    # Build burden_results from summary_df for backward compatibility
    burden_results = defaultdict(lambda: defaultdict(dict))
    for _, row in summary_df.iterrows():
        burden_results[row['Test']][row['Gene_Tolerance']][row['SFARI_Category']] = {
            'variant_count': row['Variant_Count'],
            'gene_count': row['Gene_Count'],
            'genes': row['Genes']
        }
    
    # Convert to regular dict
    burden_results = {
        test: {
            tol: dict(sfari)
            for tol, sfari in tols.items()
        } for test, tols in burden_results.items()
    }
    
    return {
        'sample_id': sample_id,
        'total_variants': len(categorized_df),
        'sfari_variants': len(sfari_variants),
        'percent_sfari': (len(sfari_variants) / len(categorized_df) * 100) if len(categorized_df) > 0 else 0,
        'burden_results': burden_results,
        'summary_df': summary_df,
        'high_priority_genes': dict(high_priority_genes),
        'gene_details': gene_details
    }

def extract_sfari_variant_counts(df):
    """
    Extract counts of variants affecting SFARI genes by variant type
    """
    # Filter for SFARI genes only
    sfari_df = df[df['SFARI_gene_score'] != '.'].copy()
    
    # Initialize counts dictionary
    sfari_counts = {}
    
    # Count main variant types
    for variant_type in ['PTV', 'Missense', 'Synonymous']:
        var_df = sfari_df[sfari_df['Variant_Category'] == variant_type]
        sfari_counts[f'SFARI_{variant_type}'] = len(var_df)
        
        # Further subdivide by pLI
        pli_high = var_df[var_df['pli_constraint'] == 'pLI>=0.9']
        pli_low = var_df[var_df['pli_constraint'] == 'pLI<0.9']
        sfari_counts[f'SFARI_{variant_type} (pLI >= 0.9)'] = len(pli_high)
        sfari_counts[f'SFARI_{variant_type} (pLI < 0.9)'] = len(pli_low)
        
        # For Missense, further subdivide by CADD score
        if variant_type == 'Missense':
            for cadd_bin in ['CADD>30', 'CADD20-30', 'CADD<20']:
                cadd_df = var_df[var_df['cadd_bin'] == cadd_bin]
                sfari_counts[f'SFARI_Missense ({cadd_bin})'] = len(cadd_df)
                
                # And by pLI within CADD bins
                pli_high_cadd = cadd_df[cadd_df['pli_constraint'] == 'pLI>=0.9']
                pli_low_cadd = cadd_df[cadd_df['pli_constraint'] == 'pLI<0.9']
                sfari_counts[f'SFARI_Missense ({cadd_bin}) (pLI >= 0.9)'] = len(pli_high_cadd)
                sfari_counts[f'SFARI_Missense ({cadd_bin}) (pLI < 0.9)'] = len(pli_low_cadd)
    
    # Further subdivide by SFARI categories
    for sfari_cat in ['1', '2', '3', 'S']:
        cat_df = sfari_df[sfari_df['SFARI_gene_score'] == sfari_cat]
        sfari_counts[f'SFARI_{sfari_cat}_total'] = len(cat_df)
        
        # Count by variant type within each SFARI category
        for variant_type in ['PTV', 'Missense', 'Synonymous']:
            var_df = cat_df[cat_df['Variant_Category'] == variant_type]
            sfari_counts[f'SFARI_{sfari_cat}_{variant_type}'] = len(var_df)
            
            # By pLI
            pli_high = var_df[var_df['pli_constraint'] == 'pLI>=0.9']
            pli_low = var_df[var_df['pli_constraint'] == 'pLI<0.9']
            sfari_counts[f'SFARI_{sfari_cat}_{variant_type} (pLI >= 0.9)'] = len(pli_high)
            sfari_counts[f'SFARI_{sfari_cat}_{variant_type} (pLI < 0.9)'] = len(pli_low)
            
            # For Missense, further subdivide by CADD score
            if variant_type == 'Missense':
                for cadd_bin in ['CADD>30', 'CADD20-30', 'CADD<20']:
                    cadd_df = var_df[var_df['cadd_bin'] == cadd_bin]
                    sfari_counts[f'SFARI_{sfari_cat}_Missense ({cadd_bin})'] = len(cadd_df)
                    
                    # And by pLI within CADD bins
                    pli_high_cadd = cadd_df[cadd_df['pli_constraint'] == 'pLI>=0.9']
                    pli_low_cadd = cadd_df[cadd_df['pli_constraint'] == 'pLI<0.9']
                    sfari_counts[f'SFARI_{sfari_cat}_Missense ({cadd_bin}) (pLI >= 0.9)'] = len(pli_high_cadd)
                    sfari_counts[f'SFARI_{sfari_cat}_Missense ({cadd_bin}) (pLI < 0.9)'] = len(pli_low_cadd)
    
    return sfari_counts