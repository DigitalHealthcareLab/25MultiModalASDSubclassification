'''
Date Created: 04-28-2025 
Author: Chanyoung Ko
Objective : New approach for analyze genetic variant differences between your autism clusters.
'''
#%%
from pathlib import Path
import os 
import pickle
import pandas as pd

# Set path to individual analysis files
individual_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/individual_analyses')

# Select one participant file
participant_id = "asd001"  # You can replace this with any specific ID
coding_file = individual_dir / f"{participant_id}_coding_analysis.pkl"
noncoding_file = individual_dir / f"{participant_id}_noncoding_analysis.pkl"

# Load coding data
if coding_file.exists():
    with open(coding_file, 'rb') as f:
        coding_data = pickle.load(f)
    
    print(f"Coding data keys: {coding_data.keys()}")
    
    # --- Check common coding data ---
    if 'common_coding' in coding_data:
        print(f"\nCommon coding keys: {coding_data['common_coding'].keys()}")
        
        if 'full_df' in coding_data['common_coding']:
            full_df = coding_data['common_coding']['full_df']
            print(f"\nCommon coding dataframe shape: {full_df.shape}")
            print(f"Common coding dataframe columns: {full_df.columns.tolist()}")
            
            if 'SFARI_gene_score' in full_df.columns:
                sfari_counts = full_df['SFARI_gene_score'].value_counts()
                print(f"\nSFARI gene score distribution (common coding): {sfari_counts.to_dict()}")
                
                sfari_df = full_df[full_df['SFARI_gene_score'] != '.']
                
                if not sfari_df.empty:
                    print(f"\nSFARI variants count (common coding): {len(sfari_df)}")
                    print(f"SFARI variant categories (common coding): {sfari_df['Variant_Category'].value_counts().to_dict()}")
                    
                    ptv_sfari = sfari_df[sfari_df['Variant_Category'] == 'PTV']
                    print(f"PTV variants in SFARI genes (common coding): {len(ptv_sfari)}")
                    
                    if 'cadd_bin' in sfari_df.columns:
                        high_cadd_sfari = sfari_df[(sfari_df['Variant_Category'] == 'Missense') & 
                                                   (sfari_df['cadd_bin'] == 'CADD>30')]
                        print(f"High CADD missense variants in SFARI genes (common coding): {len(high_cadd_sfari)}")
    
    # --- Check rare coding data ---
    if 'rare_coding' in coding_data:
        print(f"\nRare coding keys: {coding_data['rare_coding'].keys()}")
        
        if 'full_df' in coding_data['rare_coding']:
            full_df = coding_data['rare_coding']['full_df']
            print(f"\nRare coding dataframe shape: {full_df.shape}")
            print(f"Rare coding dataframe columns: {full_df.columns.tolist()}")
            
            if 'SFARI_gene_score' in full_df.columns:
                sfari_counts = full_df['SFARI_gene_score'].value_counts()
                print(f"\nSFARI gene score distribution (rare coding): {sfari_counts.to_dict()}")
                
                sfari_df = full_df[full_df['SFARI_gene_score'] != '.']
                
                if not sfari_df.empty:
                    print(f"\nSFARI variants count (rare coding): {len(sfari_df)}")
                    print(f"SFARI variant categories (rare coding): {sfari_df['Variant_Category'].value_counts().to_dict()}")
                    
                    ptv_sfari = sfari_df[sfari_df['Variant_Category'] == 'PTV']
                    print(f"PTV variants in SFARI genes (rare coding): {len(ptv_sfari)}")
                    
                    if 'cadd_bin' in sfari_df.columns:
                        high_cadd_sfari = sfari_df[(sfari_df['Variant_Category'] == 'Missense') & 
                                                   (sfari_df['cadd_bin'] == 'CADD>30')]
                        print(f"High CADD missense variants in SFARI genes (rare coding): {len(high_cadd_sfari)}")

# Load noncoding data
if noncoding_file.exists():
    with open(noncoding_file, 'rb') as f:
        noncoding_data = pickle.load(f)
    
    print(f"\nNoncoding data keys: {noncoding_data.keys()}")
    
    if 'filtered_df' in noncoding_data:
        filtered_df = noncoding_data['filtered_df']
        print(f"\nNoncoding dataframe shape: {filtered_df.shape}")
        print(f"Noncoding dataframe columns: {filtered_df.columns.tolist()}")
        
        if 'SFARI_gene_score' in filtered_df.columns:
            sfari_counts = filtered_df['SFARI_gene_score'].value_counts()
            print(f"\nSFARI gene score distribution (noncoding): {sfari_counts.to_dict()}")
            
            sfari_df = filtered_df[filtered_df['SFARI_gene_score'] != '.']
            
            if not sfari_df.empty:
                print(f"\nSFARI noncoding variants count: {len(sfari_df)}")
                
                if 'Simplified_Region' in sfari_df.columns:
                    print(f"SFARI region distribution (noncoding): {sfari_df['Simplified_Region'].value_counts().to_dict()}")


#%%
### Keep common & rare coding variants separate ###           
def create_coding_burden_tables(participant_id, coding_data, variant_type, sfari_only=False):
    """
    Create burden analysis tables for a participant.
    
    Args:
        participant_id (str): ID of the participant
        coding_data (dict): Coding data dictionary for the participant
        variant_type (str): Either 'common_coding' or 'rare_coding'
        sfari_only (bool): If True, filter for SFARI genes only
        
    Returns:
        pd.DataFrame: Burden analysis table
    """
    # Select the specific variant type dataframe
    if variant_type in coding_data and 'full_df' in coding_data[variant_type]:
        df = coding_data[variant_type]['full_df']
    else:
        return None
    
    if df.empty:
        return None
    
    # Filter for SFARI genes if requested
    if sfari_only:
        filtered_df = df[df['SFARI_gene_score'] != '.']
        label = "SFARI genes only"
    else:
        filtered_df = df
        label = "All genes (including non-SFARI)"
    
    # Basic stats
    total_variants = len(df)
    filtered_variants = len(filtered_df)
    percent = (filtered_variants / total_variants * 100) if total_variants > 0 else 0
    
    print(f"Total {variant_type} variants: {total_variants}")
    print(f"{label}: {filtered_variants} ({percent:.2f}%)")
    
    # Define categories
    variant_types = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)', 
                    'Missense (CADD 0-20)', 'Synonymous', 'Other']
    
    tolerance_categories = ['All genes', 'pLI >= 0.9', 'pLI < 0.9']
    
    # For SFARI-specific analysis, include SFARI subcategories
    if sfari_only:
        categories = ['All SFARI', 'SFARI 1', 'SFARI 2', 'SFARI 3', 'SFARI S']
    else:
        categories = ['All']  # Just one category for all genes
    
    # Initialize results table
    results = []
    
    # Process for primary category first (All SFARI or All)
    primary_category = categories[0]
    for vtype in variant_types:
        for tolerance in tolerance_categories:
            # Filter by variant type
            if vtype == 'PTV':
                type_filter = filtered_df['Variant_Category'] == 'PTV'
            elif vtype == 'Missense (CADD>30)':
                type_filter = (filtered_df['Variant_Category'] == 'Missense') & (filtered_df['cadd_bin'] == 'CADD>30')
            elif vtype == 'Missense (CADD 20-30)':
                type_filter = (filtered_df['Variant_Category'] == 'Missense') & (filtered_df['cadd_bin'] == 'CADD20-30')
            elif vtype == 'Missense (CADD 0-20)':
                type_filter = (filtered_df['Variant_Category'] == 'Missense') & (filtered_df['cadd_bin'] == 'CADD<20')
            elif vtype == 'Synonymous':
                type_filter = filtered_df['Variant_Category'] == 'Synonymous'
            else:  # 'Other'
                type_filter = filtered_df['Variant_Category'] == 'Other'
            
            # Filter by tolerance
            if tolerance == 'All genes':
                tol_filter = True
            elif tolerance == 'pLI >= 0.9':
                tol_filter = filtered_df['pli_constraint'] == 'pLI>=0.9'
            else:  # 'pLI < 0.9'
                tol_filter = filtered_df['pli_constraint'] == 'pLI<0.9'
            
            # Apply filters
            result_df = filtered_df[type_filter & tol_filter]
            
            # Count variants and genes
            variant_count = len(result_df)
            genes = result_df['GeneName'].unique()
            gene_count = len(genes)
            
            # Add to results
            results.append({
                'Test': vtype,
                'Gene_Tolerance': tolerance,
                'Category': primary_category,
                'No_of_genes': gene_count,
                'Variant_Count': variant_count
            })
    
    # For SFARI analysis, also process specific SFARI scores
    if sfari_only and len(categories) > 1:
        # Now process for specific SFARI scores
        for category, score in zip(categories[1:], ['1', '2', '3', 'S']):
            # Filter for this SFARI score
            score_df = filtered_df[filtered_df['SFARI_gene_score'] == score]
            
            for vtype in variant_types:
                for tolerance in tolerance_categories:
                    # Filter by variant type
                    if vtype == 'PTV':
                        type_filter = score_df['Variant_Category'] == 'PTV'
                    elif vtype == 'Missense (CADD>30)':
                        type_filter = (score_df['Variant_Category'] == 'Missense') & (score_df['cadd_bin'] == 'CADD>30')
                    elif vtype == 'Missense (CADD 20-30)':
                        type_filter = (score_df['Variant_Category'] == 'Missense') & (score_df['cadd_bin'] == 'CADD20-30')
                    elif vtype == 'Missense (CADD 0-20)':
                        type_filter = (score_df['Variant_Category'] == 'Missense') & (score_df['cadd_bin'] == 'CADD<20')
                    elif vtype == 'Synonymous':
                        type_filter = score_df['Variant_Category'] == 'Synonymous'
                    else:  # 'Other'
                        type_filter = score_df['Variant_Category'] == 'Other'
                    
                    # Filter by tolerance
                    if tolerance == 'All genes':
                        tol_filter = True
                    elif tolerance == 'pLI >= 0.9':
                        tol_filter = score_df['pli_constraint'] == 'pLI>=0.9'
                    else:  # 'pLI < 0.9'
                        tol_filter = score_df['pli_constraint'] == 'pLI<0.9'
                    
                    # Apply filters
                    result_df = score_df[type_filter & tol_filter]
                    
                    # Count variants and genes
                    variant_count = len(result_df)
                    genes = result_df['GeneName'].unique()
                    gene_count = len(genes)
                    
                    # Add to results
                    results.append({
                        'Test': vtype,
                        'Gene_Tolerance': tolerance,
                        'Category': category,
                        'No_of_genes': gene_count,
                        'Variant_Count': variant_count
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort appropriately
    results_df = results_df.sort_values(['Test', 'Gene_Tolerance', 'Category'])
    
    return results_df

def test_variant_separation(participant_id, individual_dir):
    """
    Test separate analyses for common and rare variants on a single participant.
    
    Args:
        participant_id (str): ID of the participant to analyze
        individual_dir (Path): Directory containing participant data files
    """
    # Load coding data
    coding_file = individual_dir / f"{participant_id}_coding_analysis.pkl"
    
    if not coding_file.exists():
        print(f"File not found: {coding_file}")
        return
    
    with open(coding_file, 'rb') as f:
        coding_data = pickle.load(f)
    
    print(f"\n===== Processing {participant_id} =====")
    
    # Process common variants - all genes
    print("\n===== All Genes Burden Analysis for {participant_id} - Common Variants =====")
    common_all_genes = create_coding_burden_tables(
        participant_id, coding_data, 'common_coding', sfari_only=False
    )
    
    if common_all_genes is not None:
        print("\nTest                      Gene_Tolerance  Category      No_of_genes  Variant_Count  ")
        print("------------------------- --------------- ------------- ------------ ---------------")
        
        for _, row in common_all_genes.iterrows():
            print(f"{row['Test']:<25} {row['Gene_Tolerance']:<15} {row['Category']:<13} {row['No_of_genes']:<12} {row['Variant_Count']:<15}")
    
    # Process common variants - SFARI genes
    print("\n===== SFARI Gene Burden Analysis for {participant_id} - Common Variants =====")
    common_sfari = create_coding_burden_tables(
        participant_id, coding_data, 'common_coding', sfari_only=True
    )
    
    if common_sfari is not None:
        # Print All SFARI section
        print("\nTest                      Gene_Tolerance  Category      No_of_genes  Variant_Count  ")
        print("------------------------- --------------- ------------- ------------ ---------------")
        
        all_sfari = common_sfari[common_sfari['Category'] == 'All SFARI']
        for _, row in all_sfari.iterrows():
            print(f"{row['Test']:<25} {row['Gene_Tolerance']:<15} {row['Category']:<13} {row['No_of_genes']:<12} {row['Variant_Count']:<15}")
        
        # Print Details by SFARI Score section
        print("\n--- Details by SFARI Score ---")
        detailed = common_sfari[common_sfari['Category'] != 'All SFARI']
        for _, row in detailed.iterrows():
            print(f"{row['Test']:<25} {row['Gene_Tolerance']:<15} {row['Category']:<13} {row['No_of_genes']:<12} {row['Variant_Count']:<15}")
    
    # Process rare variants - all genes
    print("\n===== All Genes Burden Analysis for {participant_id} - Rare Variants =====")
    rare_all_genes = create_coding_burden_tables(
        participant_id, coding_data, 'rare_coding', sfari_only=False
    )
    
    if rare_all_genes is not None:
        print("\nTest                      Gene_Tolerance  Category      No_of_genes  Variant_Count  ")
        print("------------------------- --------------- ------------- ------------ ---------------")
        
        for _, row in rare_all_genes.iterrows():
            print(f"{row['Test']:<25} {row['Gene_Tolerance']:<15} {row['Category']:<13} {row['No_of_genes']:<12} {row['Variant_Count']:<15}")
    
    # Process rare variants - SFARI genes
    print("\n===== SFARI Gene Burden Analysis for {participant_id} - Rare Variants =====")
    rare_sfari = create_coding_burden_tables(
        participant_id, coding_data, 'rare_coding', sfari_only=True
    )
    
    if rare_sfari is not None:
        # Print All SFARI section
        print("\nTest                      Gene_Tolerance  Category      No_of_genes  Variant_Count  ")
        print("------------------------- --------------- ------------- ------------ ---------------")
        
        all_sfari = rare_sfari[rare_sfari['Category'] == 'All SFARI']
        for _, row in all_sfari.iterrows():
            print(f"{row['Test']:<25} {row['Gene_Tolerance']:<15} {row['Category']:<13} {row['No_of_genes']:<12} {row['Variant_Count']:<15}")
        
        # Print Details by SFARI Score section
        print("\n--- Details by SFARI Score ---")
        detailed = rare_sfari[rare_sfari['Category'] != 'All SFARI']
        for _, row in detailed.iterrows():
            print(f"{row['Test']:<25} {row['Gene_Tolerance']:<15} {row['Category']:<13} {row['No_of_genes']:<12} {row['Variant_Count']:<15}")
    
    # Return the DataFrames for further analysis if needed
    return common_all_genes, common_sfari, rare_all_genes, rare_sfari


# Test on participant "asd001"
test_participant = "asd001"
common_all, common_sfari, rare_all, rare_sfari = test_variant_separation(test_participant, individual_dir)

#%%
### Extracting rare noncoding####
def create_noncoding_burden_tables(participant_id, noncoding_data, sfari_only=False):
    """
    Create burden analysis tables for noncoding variants of a participant.
    
    Args:
        participant_id (str): ID of the participant
        noncoding_data (dict): Noncoding data dictionary for the participant
        sfari_only (bool): If True, filter for SFARI genes only
        
    Returns:
        pd.DataFrame: Burden analysis table
    """
    # Get the filtered dataframe
    if 'filtered_df' in noncoding_data:
        filtered_df = noncoding_data['filtered_df']
    else:
        return None
    
    if filtered_df.empty:
        return None
    
    # Filter for SFARI genes if requested
    if sfari_only:
        df = filtered_df[filtered_df['SFARI_gene_score'] != '.']
        label = "SFARI genes only"
    else:
        df = filtered_df
        label = "All genes (including non-SFARI)"
    
    # Basic stats
    total_variants = len(filtered_df)
    filtered_variants = len(df)
    percent = (filtered_variants / total_variants * 100) if total_variants > 0 else 0
    
    print(f"Total noncoding variants: {total_variants}")
    print(f"{label}: {filtered_variants} ({percent:.2f}%)")
    
    # Define categories
    region_types = ['Enhancer', 'Promoter', 'Enhancer,Promoter', 'Other']
    
    # Define Eigen score categories (using eigen_bin from the data)
    # Assuming eigen_bin categorizes as 'Eigen <= 0' and 'Eigen > 0'
    eigen_categories = ['All genes', 'Eigen <= 0', 'Eigen > 0']
    
    # For SFARI-specific analysis, include SFARI subcategories
    if sfari_only:
        categories = ['All SFARI', 'SFARI 1', 'SFARI 2', 'SFARI 3', 'SFARI S']
    else:
        categories = ['All']  # Just one category for all genes
    
    # Initialize results table
    results = []
    
    # Process for primary category first (All SFARI or All)
    primary_category = categories[0]
    for region in region_types:
        for eigen_cat in eigen_categories:
            # Filter by region type
            if region == 'Other':
                # If it's not in any of the defined regions
                region_filter = ~(
                    (df['Simplified_Region'] == 'Enhancer') | 
                    (df['Simplified_Region'] == 'Promoter') | 
                    (df['Simplified_Region'] == 'Enhancer,Promoter')
                )
            else:
                region_filter = df['Simplified_Region'] == region
            
            # Filter by Eigen score
            if eigen_cat == 'All genes':
                eigen_filter = True
            elif eigen_cat == 'Eigen <= 0':
                eigen_filter = df['eigen_bin'] == 'Eigen<=0'
            else:  # 'Eigen > 0'
                eigen_filter = df['eigen_bin'] == 'Eigen>0'
            
            # Apply filters
            result_df = df[region_filter & eigen_filter]
            
            # Count variants and genes
            variant_count = len(result_df)
            genes = result_df['GeneName'].unique()
            gene_count = len(genes)
            
            # Add to results
            results.append({
                'Test': region,
                'Gene_Tolerance': eigen_cat,
                'Category': primary_category,
                'No_of_genes': gene_count,
                'Variant_Count': variant_count
            })
    
    # For SFARI analysis, also process specific SFARI scores
    if sfari_only and len(categories) > 1:
        # Now process for specific SFARI scores
        for category, score in zip(categories[1:], ['1', '2', '3', 'S']):
            # Filter for this SFARI score
            score_df = df[df['SFARI_gene_score'] == score]
            
            for region in region_types:
                for eigen_cat in eigen_categories:
                    # Filter by region type
                    if region == 'Other':
                        # If it's not in any of the defined regions
                        region_filter = ~(
                            (score_df['Simplified_Region'] == 'Enhancer') | 
                            (score_df['Simplified_Region'] == 'Promoter') | 
                            (score_df['Simplified_Region'] == 'Enhancer,Promoter')
                        )
                    else:
                        region_filter = score_df['Simplified_Region'] == region
                    
                    # Filter by Eigen score
                    if eigen_cat == 'All genes':
                        eigen_filter = True
                    elif eigen_cat == 'Eigen <= 0':
                        eigen_filter = score_df['eigen_bin'] == 'Eigen<=0'
                    else:  # 'Eigen > 0'
                        eigen_filter = score_df['eigen_bin'] == 'Eigen>0'
                    
                    # Apply filters
                    result_df = score_df[region_filter & eigen_filter]
                    
                    # Count variants and genes
                    variant_count = len(result_df)
                    genes = result_df['GeneName'].unique()
                    gene_count = len(genes)
                    
                    # Add to results
                    results.append({
                        'Test': region,
                        'Gene_Tolerance': eigen_cat,
                        'Category': category,
                        'No_of_genes': gene_count,
                        'Variant_Count': variant_count
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort appropriately
    results_df = results_df.sort_values(['Test', 'Gene_Tolerance', 'Category'])
    
    return results_df

def test_noncoding_analysis(participant_id, individual_dir):
    """
    Test noncoding variant analysis on a single participant.
    
    Args:
        participant_id (str): ID of the participant to analyze
        individual_dir (Path): Directory containing participant data files
    """
    # Load noncoding data
    noncoding_file = individual_dir / f"{participant_id}_noncoding_analysis.pkl"
    
    if not noncoding_file.exists():
        print(f"File not found: {noncoding_file}")
        return
    
    with open(noncoding_file, 'rb') as f:
        noncoding_data = pickle.load(f)
    
    print(f"\n===== Processing {participant_id} Noncoding Variants =====")
    
    # Process all genes
    print(f"\n===== All Genes Burden Analysis for {participant_id} - Rare Noncoding Variants =====")
    all_genes = create_noncoding_burden_tables(
        participant_id, noncoding_data, sfari_only=False
    )
    
    if all_genes is not None:
        print("\nTest                      Gene_Tolerance  Category      No_of_genes  Variant_Count  ")
        print("------------------------- --------------- ------------- ------------ ---------------")
        
        for _, row in all_genes.iterrows():
            print(f"{row['Test']:<25} {row['Gene_Tolerance']:<15} {row['Category']:<13} {row['No_of_genes']:<12} {row['Variant_Count']:<15}")
    
    # Process SFARI genes
    print(f"\n===== SFARI Gene Burden Analysis for {participant_id} - Rare Noncoding Variants =====")
    sfari_genes = create_noncoding_burden_tables(
        participant_id, noncoding_data, sfari_only=True
    )
    
    if sfari_genes is not None:
        # Print All SFARI section
        print("\nTest                      Gene_Tolerance  Category      No_of_genes  Variant_Count  ")
        print("------------------------- --------------- ------------- ------------ ---------------")
        
        all_sfari = sfari_genes[sfari_genes['Category'] == 'All SFARI']
        for _, row in all_sfari.iterrows():
            print(f"{row['Test']:<25} {row['Gene_Tolerance']:<15} {row['Category']:<13} {row['No_of_genes']:<12} {row['Variant_Count']:<15}")
        
        # Print Details by SFARI Score section
        print("\n--- Details by SFARI Score ---")
        detailed = sfari_genes[sfari_genes['Category'] != 'All SFARI']
        for _, row in detailed.iterrows():
            print(f"{row['Test']:<25} {row['Gene_Tolerance']:<15} {row['Category']:<13} {row['No_of_genes']:<12} {row['Variant_Count']:<15}")
    
    # Return the DataFrames for further analysis if needed
    return all_genes, sfari_genes


# Test on participant "asd001"
test_participant = "asd001"
noncoding_all, noncoding_sfari = test_noncoding_analysis(test_participant, individual_dir)


#%%
######################################################
## Process across all participants ## 
######################################################
import pickle
import pandas as pd
from pathlib import Path
import sys

def process_all_participants(individual_dir, cluster_assignments):
    """
    Process all participants and create combined burden tables that preserve
    the original structure including gene tolerance categories.
    
    Args:
        individual_dir (Path): Directory containing individual analysis files
        cluster_assignments (dict): Mapping of participant IDs to clusters
        
    Returns:
        dict: Dictionary containing combined burden tables for all analyses
    """
    # Initialize dictionaries to store combined tables
    combined_tables = {
        'common_coding_all': [],
        'common_coding_sfari': [],
        'rare_coding_all': [],
        'rare_coding_sfari': [],
        'rare_noncoding_all': [],
        'rare_noncoding_sfari': []
    }
    
    # Track processing stats
    total_participants = len(cluster_assignments)
    processed_participants = 0
    failed_participants = 0
    
    print(f"Starting to process {total_participants} participants...")
    
    # Process each participant
    for participant_id, cluster in cluster_assignments.items():
        print(f"\nProcessing participant: {participant_id} (Cluster {cluster}) - [{processed_participants + failed_participants + 1}/{total_participants}]")
        
        # Process coding variants
        coding_file = individual_dir / f"{participant_id}_coding_analysis.pkl"
        
        if coding_file.exists():
            try:
                with open(coding_file, 'rb') as f:
                    coding_data = pickle.load(f)
                
                print(f"  ✓ Loaded coding data for {participant_id}")
                
                # Process common coding variants - all genes
                common_all_table = create_coding_burden_tables(
                    participant_id, coding_data, 'common_coding', sfari_only=False
                )
                
                if common_all_table is not None and not common_all_table.empty:
                    # Add participant ID and cluster columns
                    common_all_table['participant_id'] = participant_id
                    common_all_table['cluster'] = cluster
                    print(f"  ✓ Created common coding (all genes) table: {len(common_all_table)} rows")
                    combined_tables['common_coding_all'].append(common_all_table)
                
                # Process common coding variants - SFARI genes
                common_sfari_table = create_coding_burden_tables(
                    participant_id, coding_data, 'common_coding', sfari_only=True
                )
                
                if common_sfari_table is not None and not common_sfari_table.empty:
                    # Add participant ID and cluster columns
                    common_sfari_table['participant_id'] = participant_id
                    common_sfari_table['cluster'] = cluster
                    print(f"  ✓ Created common coding (SFARI genes) table: {len(common_sfari_table)} rows")
                    combined_tables['common_coding_sfari'].append(common_sfari_table)
                
                # Process rare coding variants - all genes
                rare_all_table = create_coding_burden_tables(
                    participant_id, coding_data, 'rare_coding', sfari_only=False
                )
                
                if rare_all_table is not None and not rare_all_table.empty:
                    # Add participant ID and cluster columns
                    rare_all_table['participant_id'] = participant_id
                    rare_all_table['cluster'] = cluster
                    print(f"  ✓ Created rare coding (all genes) table: {len(rare_all_table)} rows")
                    combined_tables['rare_coding_all'].append(rare_all_table)
                
                # Process rare coding variants - SFARI genes
                rare_sfari_table = create_coding_burden_tables(
                    participant_id, coding_data, 'rare_coding', sfari_only=True
                )
                
                if rare_sfari_table is not None and not rare_sfari_table.empty:
                    # Add participant ID and cluster columns
                    rare_sfari_table['participant_id'] = participant_id
                    rare_sfari_table['cluster'] = cluster
                    print(f"  ✓ Created rare coding (SFARI genes) table: {len(rare_sfari_table)} rows")
                    combined_tables['rare_coding_sfari'].append(rare_sfari_table)
                
                processed_participants += 1
            
            except Exception as e:
                print(f"  ✗ Error processing coding file for {participant_id}: {e}")
                failed_participants += 1
        else:
            print(f"  ✗ Coding file not found: {coding_file}")
        
        # Process noncoding variants
        noncoding_file = individual_dir / f"{participant_id}_noncoding_analysis.pkl"
        
        if noncoding_file.exists():
            try:
                with open(noncoding_file, 'rb') as f:
                    noncoding_data = pickle.load(f)
                
                print(f"  ✓ Loaded noncoding data for {participant_id}")
                
                # Process noncoding variants - all genes
                noncoding_all_table = create_noncoding_burden_tables(
                    participant_id, noncoding_data, sfari_only=False
                )
                
                if noncoding_all_table is not None and not noncoding_all_table.empty:
                    # Add participant ID and cluster columns
                    noncoding_all_table['participant_id'] = participant_id
                    noncoding_all_table['cluster'] = cluster
                    print(f"  ✓ Created noncoding (all genes) table: {len(noncoding_all_table)} rows")
                    combined_tables['rare_noncoding_all'].append(noncoding_all_table)
                
                # Process noncoding variants - SFARI genes
                noncoding_sfari_table = create_noncoding_burden_tables(
                    participant_id, noncoding_data, sfari_only=True
                )
                
                if noncoding_sfari_table is not None and not noncoding_sfari_table.empty:
                    # Add participant ID and cluster columns
                    noncoding_sfari_table['participant_id'] = participant_id
                    noncoding_sfari_table['cluster'] = cluster
                    print(f"  ✓ Created noncoding (SFARI genes) table: {len(noncoding_sfari_table)} rows")
                    combined_tables['rare_noncoding_sfari'].append(noncoding_sfari_table)
            
            except Exception as e:
                print(f"  ✗ Error processing noncoding file for {participant_id}: {e}")
                failed_participants += 1
        else:
            print(f"  ✗ Noncoding file not found: {noncoding_file}")
    
    # Combine all tables into dataframes
    combined_dfs = {}
    for key, tables in combined_tables.items():
        if tables:
            combined_dfs[key] = pd.concat(tables, ignore_index=True)
            print(f"\nCombined {key} dataframe: {combined_dfs[key].shape}")
        else:
            combined_dfs[key] = pd.DataFrame()
            print(f"\nNo data for {key}")
    
    # Print statistics
    print(f"\n===== Processing Summary =====")
    print(f"Total participants: {total_participants}")
    print(f"Successfully processed: {processed_participants}")
    print(f"Failed to process: {failed_participants}")
    
    # Return the combined dataframes
    return combined_dfs

def create_cluster_summaries(df, value_columns, groupby_columns):
    """
    Create summary statistics by cluster and other grouping variables.
    
    Args:
        df (pd.DataFrame): DataFrame with variant data
        value_columns (list): List of columns to calculate statistics for
        groupby_columns (list): Columns to group by (e.g., ['cluster', 'variant_type'])
        
    Returns:
        pd.DataFrame: Summary statistics
    """
    if df.empty:
        print("Warning: Empty DataFrame provided to create_cluster_summaries")
        # Return empty DataFrame with expected columns for consistency
        expected_columns = groupby_columns + ['participant_id_count'] + [f"{col}_{stat}" for col in value_columns for stat in ['mean', 'median', 'sum', 'std']]
        return pd.DataFrame(columns=expected_columns)
    
    try:
        # Group by the specified columns
        grouped = df.groupby(groupby_columns)
        
        # Calculate statistics
        summaries = grouped.agg({
            'participant_id': 'count',  # Count of participants
            **{col: ['mean', 'median', 'sum', 'std'] for col in value_columns}
        }).reset_index()
        
        # Flatten multi-level columns
        summaries.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in summaries.columns]
        
        return summaries
    except Exception as e:
        print(f"Error creating cluster summaries: {e}")
        # Return empty DataFrame with expected columns for consistency
        expected_columns = groupby_columns + ['participant_id_count'] + [f"{col}_{stat}" for col in value_columns for stat in ['mean', 'median', 'sum', 'std']]
        return pd.DataFrame(columns=expected_columns)

def analyze_clusters(combined_dfs, results_dir):
    """
    Save combined dataframes and create summary statistics.
    
    Args:
        combined_dfs (dict): Dictionary containing combined burden tables
        results_dir (Path): Directory to save results
    """
    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n===== Saving Combined Data =====")
    # Save combined data
    for key, df in combined_dfs.items():
        if not df.empty:
            output_file = results_dir / f"{key}_combined.csv"
            df.to_csv(output_file, index=False)
            print(f"✓ Saved {output_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
        else:
            print(f"✗ No data for {key}")
    
    print("\n===== Creating Summary Statistics =====")
    # Create and save cluster summaries
    for key, df in combined_dfs.items():
        if df.empty:
            continue
            
        try:
            # Define grouping columns based on dataframe type
            if 'sfari' in key:
                groupby_columns = ['cluster', 'Test', 'Gene_Tolerance', 'Category']
            else:
                groupby_columns = ['cluster', 'Test', 'Gene_Tolerance']
                
            # Create summary with count, mean, and sum of variant counts
            summary = df.groupby(groupby_columns).agg({
                'participant_id': 'nunique',  # Count unique participants
                'Variant_Count': ['mean', 'median', 'sum', 'std'],
                'No_of_genes': ['mean', 'median', 'sum', 'std']
            }).reset_index()
            
            # Flatten column names
            summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in summary.columns]
            
            # Save summary
            output_file = results_dir / f"{key}_summary.csv"
            summary.to_csv(output_file, index=False)
            print(f"✓ Created and saved {output_file.name}")
            
        except Exception as e:
            print(f"✗ Error creating summary for {key}: {e}")
    
    print(f"\n===== All data and summaries saved to {results_dir} =====")
    return True

def load_cluster_info():
    """
    Load cluster assignments for participants.

    Returns:
        tuple: (cluster_df, participant_to_cluster) where:
            - cluster_df is a DataFrame with participant info and clusters
            - participant_to_cluster is a dict mapping participant IDs to cluster IDs
    """
    try:
        cluster_file = wgs_dir / 'dataframes' / 'participant_clinical_brain_cluster_df.csv'
        
        if not cluster_file.exists():
            print(f"Error: Cluster file not found at {cluster_file}")
            return pd.DataFrame(), {}
            
        cluster_df = pd.read_csv(cluster_file)
        
        # Check for required columns
        if 'cluster' not in cluster_df.columns or 'id' not in cluster_df.columns:
            print(f"Error: Cluster file missing required columns. Found: {cluster_df.columns.tolist()}")
            return pd.DataFrame(), {}
            
        # Drop rows with missing cluster values
        original_count = len(cluster_df)
        cluster_df = cluster_df.dropna(subset=['cluster'])
        if len(cluster_df) < original_count:
            print(f"Warning: Dropped {original_count - len(cluster_df)} rows with missing cluster values")
            
        # Convert data types
        cluster_df['cluster'] = cluster_df['cluster'].astype(int)
        cluster_df['id'] = cluster_df['id'].astype(str)
        
        # Create mapping
        participant_to_cluster = dict(zip(cluster_df['id'], cluster_df['cluster']))
        
        print(f"Successfully loaded cluster assignments for {len(participant_to_cluster)} participants")
        print(f"Clusters found: {sorted(cluster_df['cluster'].unique())}")
        
        return cluster_df, participant_to_cluster
        
    except Exception as e:
        print(f"Error loading cluster information: {e}")
        return pd.DataFrame(), {}

def main():
    """
    Main function to run the entire analysis pipeline.
    """
    print("="*80)
    print("Starting DNA Variant Analysis Across Participants")
    print("="*80)
    
    # Define the paths - using your existing paths
    base_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis')
    individual_dir = base_dir / 'individual_analyses'
    results_dir = base_dir / 'cluster_summaries'
    global wgs_dir
    wgs_dir = Path('/home/data/2025_asd_wgs')  # Define this based on your system structure
        
    # Verify paths exist
    if not individual_dir.exists():
        print(f"Error: Individual directory not found at {individual_dir}")
        return
        
    if not wgs_dir.exists():
        print(f"Error: WGS directory not found at {wgs_dir}")
        return
        
    print(f"Base directory: {base_dir}")
    print(f"Individual analyses directory: {individual_dir}")
    print(f"Results directory: {results_dir}")
    print(f"WGS directory: {wgs_dir}")
    
    # Load cluster assignments
    print("\nLoading cluster assignments...")
    cluster_df, participant_to_cluster = load_cluster_info()
    
    if not participant_to_cluster:
        print("Error: Failed to load cluster assignments. Aborting.")
        return
        
    # Print a sample of cluster assignments
    print(f"\nSample of cluster assignments:")
    for participant_id, cluster in list(participant_to_cluster.items())[:5]:
        print(f"{participant_id}: Cluster {cluster}")
    
    # Process all participants
    print("\nProcessing all participants...")
    try:
        combined_dfs = process_all_participants(individual_dir, participant_to_cluster)
    except Exception as e:
        print(f"Error during participant processing: {e}")
        print("Aborting analysis.")
        return
    
    # Analyze clusters and generate summary statistics
    print("\nAnalyzing clusters and generating summaries...")
    try:
        analyze_clusters(combined_dfs, results_dir)
    except Exception as e:
        print(f"Error during cluster analysis: {e}")
        print("Failed to complete analysis.")
        return
        
    print("\nAnalysis complete!")


# This allows you to run the script directly or import functions from it
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled error in main execution: {e}")
        sys.exit(1)

#%%
## Check saved results before analyzing for all 42 partcipants ## 
import pandas as pd
from pathlib import Path

# Set path to cluster summaries
cluster_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/cluster_summaries')

# Check common_coding_all_combined.csv
common_coding_all = pd.read_csv(cluster_dir / 'common_coding_all_combined.csv')
print("\n--- common_coding_all_combined.csv ---")
print(f"Shape: {common_coding_all.shape}")
print(f"Columns: {list(common_coding_all.columns)}")
print("\nFirst few rows:")
print(common_coding_all.head(3))

# Check Gene_Tolerance categories
if 'Gene_Tolerance' in common_coding_all.columns:
    print("\nGene_Tolerance categories:")
    print(common_coding_all['Gene_Tolerance'].value_counts())

# Check common_coding_sfari_combined.csv
common_coding_sfari = pd.read_csv(cluster_dir / 'common_coding_sfari_combined.csv')
print("\n--- common_coding_sfari_combined.csv ---")
print(f"Shape: {common_coding_sfari.shape}")
print(f"Columns: {list(common_coding_sfari.columns)}")
print("\nFirst few rows:")
print(common_coding_sfari.head(3))

# Check SFARI categories
if 'Category' in common_coding_sfari.columns:
    print("\nSFARI categories:")
    print(common_coding_sfari['Category'].value_counts())

# Check rare_noncoding_all_combined.csv
rare_noncoding_all = pd.read_csv(cluster_dir / 'rare_noncoding_all_combined.csv')
print("\n--- rare_noncoding_all_combined.csv ---")
print(f"Shape: {rare_noncoding_all.shape}")
print(f"Columns: {list(rare_noncoding_all.columns)}")
print("\nFirst few rows:")
print(rare_noncoding_all.head(3))

# Check noncoding categories (Test = region type, Gene_Tolerance = Eigen score)
if 'Test' in rare_noncoding_all.columns:
    print("\nNoncoding region types:")
    print(rare_noncoding_all['Test'].value_counts())

if 'Gene_Tolerance' in rare_noncoding_all.columns:
    print("\nEigen score categories:")
    print(rare_noncoding_all['Gene_Tolerance'].value_counts())

print("\nInspection complete.")

# Check common_coding_sfari_combined.csv
rare_noncoding_sfari = pd.read_csv(cluster_dir / 'rare_noncoding_sfari_combined.csv')
print("\n--- rare_noncoding_sfari_combined.csv ---")
print(f"Shape: {rare_noncoding_sfari.shape}")
print(f"Columns: {list(rare_noncoding_sfari.columns)}")
print("\nFirst few rows:")
print(rare_noncoding_sfari.head(3))

# Check SFARI categories
if 'Category' in rare_noncoding_sfari.columns:
    print("\nSFARI categories:")
    print(rare_noncoding_sfari['Category'].value_counts())


#%%
#####################################################
# Step 2: Statistical Analysis of Variant Distribution (updated)
######################################################
# Load the functions to perform statistical analysis & execute
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Define a global seed manager class
class SeedManager:
    """
    A class to manage random seeds for reproducible results.
    This ensures consistent seeding across multiple function calls.
    """
    def __init__(self, master_seed=42):
        self.master_seed = master_seed
        self.feature_counter = 0
        self.bootstrap_counter = 0
        self.set_master_seed()
    
    def set_master_seed(self):
        """Set the master seed for the entire analysis."""
        np.random.seed(self.master_seed)
        
    def get_feature_seed(self):
        """Get a deterministic seed for each feature based on feature counter."""
        self.feature_counter += 1
        return self.master_seed + self.feature_counter
    
    def get_bootstrap_seed(self):
        """Get a deterministic seed for each bootstrap iteration."""
        self.bootstrap_counter += 1
        return self.master_seed + 1000 + self.bootstrap_counter
        
    def reset_feature_counter(self):
        """Reset feature counter for a new analysis."""
        self.feature_counter = 0
        
    def reset_bootstrap_counter(self):
        """Reset bootstrap counter for a new bootstrap analysis."""
        self.bootstrap_counter = 0


####### Create a global seed manager instance ####### 
seed_manager = SeedManager(master_seed=14) #42, 14

# Function to load combined data
def load_combined_data(base_dir, file_prefix):
    """Load combined data files for statistical analysis."""
    file_path = base_dir / f"{file_prefix}_combined.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Ensure numeric columns are correctly typed
    numeric_cols = ['No_of_genes', 'Variant_Count']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure cluster is treated as a categorical variable
    if 'cluster' in df.columns:
        df['cluster'] = df['cluster'].astype('category')
    
    # Create a combined feature column for easier analysis
    # Ensure all required columns exist before trying to combine them
    required_cols = ['Test', 'Gene_Tolerance', 'Category']
    if all(col in df.columns for col in required_cols):
        df['feature'] = df.apply(lambda x: f"{x['Test']} - {x['Gene_Tolerance']} - {x['Category']}", axis=1)
    else:
        # For dataframes that might be missing some columns, use what's available
        feature_cols = []
        if 'Test' in df.columns:
            feature_cols.append('Test')
        if 'Gene_Tolerance' in df.columns:
            feature_cols.append('Gene_Tolerance')
        if 'Category' in df.columns:
            feature_cols.append('Category')
        
        if feature_cols:
            df['feature'] = df[feature_cols].apply(lambda x: ' - '.join(x.astype(str)), axis=1)
        else:
            print(f"Warning: No feature columns found in {file_prefix}")
    
    print(f"Loaded {len(df)} rows with {df['participant_id'].nunique()} unique participants")
    if 'feature' in df.columns:
        print(f"Found {len(df['feature'].unique())} unique features")
    
    return df

# Function to perform Kruskal-Wallis test on variant distributions
def perform_kruskal_wallis(df, variant_column):
    """
    Perform Kruskal-Wallis test with improved error handling and debugging.
    """
    clusters = sorted(df['cluster'].unique())
    
    # Create groups and print diagnostics
    groups = []
    print(f"Checking data for feature: {df['feature'].iloc[0] if 'feature' in df.columns else 'Unknown'}")
    
    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster][variant_column].values
        print(f"  Cluster {cluster}: {len(cluster_data)} values, min={np.min(cluster_data) if len(cluster_data) > 0 else 'N/A'}, max={np.max(cluster_data) if len(cluster_data) > 0 else 'N/A'}")
        
        # Only add non-empty groups
        if len(cluster_data) > 0:
            groups.append(cluster_data)
    
    # Check if we have enough groups
    if len(groups) < 2:
        print("  Not enough groups for Kruskal-Wallis test")
        return {'statistic': np.nan, 'pvalue': np.nan}
    
    # Check if all values are identical
    all_values = np.concatenate(groups)
    if len(np.unique(all_values)) == 1:
        print(f"  All values are identical: {all_values[0]}")
        return {'statistic': np.nan, 'pvalue': np.nan}
    
    # Perform Kruskal-Wallis test
    try:
        statistic, pvalue = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis test successful: statistic={statistic:.4f}, p-value={pvalue:.6f}")
        return {'statistic': statistic, 'pvalue': pvalue}
    except Exception as e:
        print(f"  Error in Kruskal-Wallis test: {e}")
        return {'statistic': np.nan, 'pvalue': np.nan}

# Function to calculate effect sizes for each cluster vs. others
def calculate_effect_sizes(df, variant_column, use_median=True):
    """Calculate effect size (log2 fold change) for each cluster vs. all others combined."""
    clusters = sorted(df['cluster'].unique())
    results = []
    
    # Set seed for this feature
    feature_seed = seed_manager.get_feature_seed()
    np.random.seed(feature_seed)
    
    # For each cluster, calculate effect size vs all others combined
    for cluster in clusters:
        # Split data into current cluster and all others
        cluster_data = df[df['cluster'] == cluster][variant_column].values
        other_data = df[df['cluster'] != cluster][variant_column].values
        
        # Skip if either group is empty
        if len(cluster_data) == 0 or len(other_data) == 0:
            continue
            
        # Calculate central tendency
        if use_median:
            cluster_central = np.median(cluster_data)
            other_central = np.median(other_data)
        else:
            cluster_central = np.mean(cluster_data)
            other_central = np.mean(other_data)
            
        # Handle zeros to avoid log(0) issues
        if cluster_central == 0:
            cluster_central = 0.1
        if other_central == 0:
            other_central = 0.1
            
        # Calculate log2 fold change
        log2fc = np.log2(cluster_central / other_central)
        
        # Perform Mann-Whitney U test for this cluster vs others
        try:
            u_stat, p_value = stats.mannwhitneyu(cluster_data, other_data, alternative='two-sided')
            
            # Calculate standardized effect size (r = Z / sqrt(N))
            # First convert U to Z score
            n1, n2 = len(cluster_data), len(other_data)
            z_score = (u_stat - (n1 * n2 / 2)) / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
            r_effect = z_score / np.sqrt(n1 + n2)
            
        except Exception as e:
            print(f"Error in Mann-Whitney test for cluster {cluster}: {e}")
            p_value = np.nan
            r_effect = np.nan
        
        # Calculate confidence intervals for log2fc using bootstrapping
        try:
            # Reset bootstrap counter before bootstrapping
            seed_manager.reset_bootstrap_counter()
            
            n_resamples = 1000
            bootstrap_log2fcs = []
            
            for _ in range(n_resamples):
                # Set a new seed for each bootstrap sample
                bs_seed = seed_manager.get_bootstrap_seed()
                np.random.seed(bs_seed)
                
                # Resample with replacement
                cluster_resample = np.random.choice(cluster_data, size=len(cluster_data), replace=True)
                other_resample = np.random.choice(other_data, size=len(other_data), replace=True)
                
                # Calculate central tendency for resamples
                if use_median:
                    cluster_central_bs = np.median(cluster_resample)
                    other_central_bs = np.median(other_resample)
                else:
                    cluster_central_bs = np.mean(cluster_resample)
                    other_central_bs = np.mean(other_resample)
                
                # Handle zeros
                if cluster_central_bs == 0:
                    cluster_central_bs = 0.1
                if other_central_bs == 0:
                    other_central_bs = 0.1
                
                # Calculate log2fc for this resample
                log2fc_bs = np.log2(cluster_central_bs / other_central_bs)
                bootstrap_log2fcs.append(log2fc_bs)
            
            # Calculate 95% confidence interval
            ci_lower = np.percentile(bootstrap_log2fcs, 2.5)
            ci_upper = np.percentile(bootstrap_log2fcs, 97.5)
            
        except Exception as e:
            print(f"Error in bootstrapping for cluster {cluster}: {e}")
            ci_lower, ci_upper = np.nan, np.nan
        
        # Store results
        results.append({
            'cluster': cluster,
            'feature': df['feature'].iloc[0] if 'feature' in df.columns else variant_column,
            'log2fc': log2fc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'effect_size': r_effect,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False,
            'trending': (p_value >= 0.05) & (p_value < 0.1) if not np.isnan(p_value) else False
        })
    
    return pd.DataFrame(results)

# Function to analyze feature-based variant distributions
def analyze_feature_distributions(df, use_median=True):
    """Analyze distribution of variant features across clusters."""
    kw_results = []
    effect_sizes_list = []
    
    # Reset feature seed counter at the start of a new analysis
    seed_manager.reset_feature_counter()
    
    # Get unique features
    features = df['feature'].unique()
    
    # Perform analysis for each feature
    for feature in features:
        # Filter data for this feature
        feature_df = df[df['feature'] == feature].copy()
        
        # Skip if too few data points
        if feature_df['participant_id'].nunique() < 5:
            print(f"Skipping {feature}: too few participants")
            continue
            
        # Perform Kruskal-Wallis test
        kw_result = perform_kruskal_wallis(feature_df, 'Variant_Count')
        kw_results.append({
            'feature': feature,
            'statistic': kw_result['statistic'],
            'pvalue': kw_result['pvalue'],
            'significant': kw_result['pvalue'] < 0.05 if not np.isnan(kw_result['pvalue']) else False,
            'trending': (kw_result['pvalue'] >= 0.05) & (kw_result['pvalue'] < 0.1) if not np.isnan(kw_result['pvalue']) else False
        })
        
        # Calculate effect sizes - seed is managed inside the function
        effect_sizes = calculate_effect_sizes(feature_df, 'Variant_Count', use_median=use_median)
        if not effect_sizes.empty:
            effect_sizes_list.append(effect_sizes)
    
    # Combine all effect sizes into single DataFrame if any exist
    if effect_sizes_list:
        all_effect_sizes = pd.concat(effect_sizes_list, ignore_index=True)
    else:
        all_effect_sizes = pd.DataFrame()
    
    # Create DataFrame with Kruskal-Wallis results
    kw_df = pd.DataFrame(kw_results)
    
    return {
        'kruskal_wallis': kw_df,
        'effect_sizes': all_effect_sizes
    }

# Function to display results with formatting
def display_results(results_df, title, show_all=True):
    """Display results with optional filtering for significant and trending results"""
    print(f"\n{title}:")
    if results_df.empty:
        print("No results found")
        return
    
    # Sort by p-value
    sorted_df = results_df.sort_values('pvalue')
    
    # Add significance markers
    sorted_df['significance'] = ''
    sorted_df.loc[(sorted_df['pvalue'] < 0.1) & (sorted_df['pvalue'] >= 0.05), 'significance'] = '*'  # Trending
    sorted_df.loc[sorted_df['pvalue'] < 0.05, 'significance'] = '**'  # Significant
    
    # Add trending column
    sorted_df['trending'] = (sorted_df['pvalue'] < 0.1) & (sorted_df['pvalue'] >= 0.05)
    
    # Format p-values
    sorted_df['pvalue_formatted'] = sorted_df['pvalue'].apply(lambda x: f"{x:.6f}" if not np.isnan(x) else "N/A")
    
    if show_all:
        display_cols = ['feature', 'statistic', 'pvalue_formatted', 'significance']
        print(sorted_df[display_cols].to_string(index=False))
    else:
        # Show both significant and trending results
        filtered_df = sorted_df[sorted_df['pvalue'] < 0.1]  # Include both significant (<0.05) and trending (<0.1)
        if filtered_df.empty:
            print("No significant or trending results found")
        else:
            display_cols = ['feature', 'statistic', 'pvalue_formatted', 'significance', 'trending']
            print(filtered_df[display_cols].to_string(index=False))


#%%
# Load the combined data files
# Set paths
results_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/cluster_summaries')
plots_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/plots')
os.makedirs(plots_dir, exist_ok=True)

try:
    # All genes combined data
    common_coding_all = load_combined_data(results_dir, 'common_coding_all')
    rare_coding_all = load_combined_data(results_dir, 'rare_coding_all')
    rare_noncoding_all = load_combined_data(results_dir, 'rare_noncoding_all')
    
    # SFARI genes combined data
    common_coding_sfari = load_combined_data(results_dir, 'common_coding_sfari')
    rare_coding_sfari = load_combined_data(results_dir, 'rare_coding_sfari')
    rare_noncoding_sfari = load_combined_data(results_dir, 'rare_noncoding_sfari')
    
    print("Data loading complete.")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Define dataset groups for cleaner processing
all_genes_datasets = [
    ('common_coding', common_coding_all),
    ('rare_coding', rare_coding_all),
    ('rare_noncoding', rare_noncoding_all)
]

sfari_datasets = [
    ('common_coding', common_coding_sfari),
    ('rare_coding', rare_coding_sfari),
    ('rare_noncoding', rare_noncoding_sfari)
]

# Analysis results will be stored in these dictionaries
all_genes_results = {}
sfari_results = {}

# PART 1: Analysis using all genes
print("\n==== ANALYSIS PART 1: ALL GENES ====")

# Process all genes datasets
for name, dataset in all_genes_datasets:
    print(f"Analyzing {name} variants (all genes)...")

    results = analyze_feature_distributions(dataset)
    all_genes_results[name] = results
    
    # Save results
    results['kruskal_wallis'].to_csv(results_dir / f"all_genes_{name}_kruskal_wallis.csv", index=False)
    results['effect_sizes'].to_csv(results_dir / f"all_genes_{name}_effect_sizes.csv", index=False)

# PART 2: Analysis focusing specifically on SFARI genes
print("\n==== ANALYSIS PART 2: SFARI GENES ====")

# Process SFARI genes datasets
for name, dataset in sfari_datasets:
    print(f"Analyzing {name} variants (SFARI genes)...")
    
    results = analyze_feature_distributions(dataset)
    sfari_results[name] = results
    
    # Save results
    results['kruskal_wallis'].to_csv(results_dir / f"sfari_{name}_kruskal_wallis.csv", index=False)
    results['effect_sizes'].to_csv(results_dir / f"sfari_{name}_effect_sizes.csv", index=False)

# Combine all effect sizes for the forest plot (all genes)
all_genes_effect_sizes = pd.concat([
    all_genes_results['common_coding']['effect_sizes'],
    all_genes_results['rare_coding']['effect_sizes'],
    all_genes_results['rare_noncoding']['effect_sizes']
], ignore_index=True)

all_genes_effect_sizes.to_csv(results_dir / "all_genes_variant_effect_sizes.csv", index=False)

# Combine all SFARI-specific effect sizes
sfari_effect_sizes = pd.concat([
    sfari_results['common_coding']['effect_sizes'],
    sfari_results['rare_coding']['effect_sizes'],
    sfari_results['rare_noncoding']['effect_sizes']
], ignore_index=True)

sfari_effect_sizes.to_csv(results_dir / "sfari_variant_effect_sizes.csv", index=False)

# Print summary of findings - ALL results, sorted by p-value
print("\n==== ANALYSIS RESULTS (ALL GENES) ====")
for name in all_genes_results:
    display_results(all_genes_results[name]['kruskal_wallis'], f"{name.replace('_', ' ').title()} variants (all genes)", show_all=True)

print("\n==== ANALYSIS RESULTS (SFARI GENES) ====")
for name in sfari_results:
    display_results(sfari_results[name]['kruskal_wallis'], f"{name.replace('_', ' ').title()} variants (SFARI genes)", show_all=True)

# Print just the significant findings for easy reference
print("\n==== SIGNIFICANT FINDINGS SUMMARY ====")
for name in all_genes_results:
    display_results(all_genes_results[name]['kruskal_wallis'], f"{name.replace('_', ' ').title()} significant variants (all genes)", show_all=False)

for name in sfari_results:
    display_results(sfari_results[name]['kruskal_wallis'], f"{name.replace('_', ' ').title()} significant variants (SFARI genes)", show_all=False)


#%%
# Check a sample of the 'all_genes_variant_effect_sizes.csv' file 
import pandas as pd
from pathlib import Path

# Path to the results directory
results_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/cluster_summaries')

# Load the effect sizes file
effect_sizes_file = results_dir / "all_genes_variant_effect_sizes.csv"

# Check if the file exists
if effect_sizes_file.exists():
    # Load the file
    effect_sizes_df = pd.read_csv(effect_sizes_file)
    
    # Print basic information
    print(f"File shape: {effect_sizes_df.shape}")
    print(f"Columns: {list(effect_sizes_df.columns)}")
    
    # Print first 10 rows
    print("\nFirst 10 rows:")
    print(effect_sizes_df.head(10))
    
    # Print some statistics about effect sizes
    print("\nSummary statistics for log2fc:")
    print(effect_sizes_df['log2fc'].describe())
    
    # Print some unique features to see the structure
    print("\nSample of unique features:")
    print(effect_sizes_df['feature'].sample(10).values)
    
    # Count significant findings
    sig_count = effect_sizes_df['significant'].sum()
    print(f"\nNumber of significant findings: {sig_count} out of {len(effect_sizes_df)}")
else:
    print(f"File not found: {effect_sizes_file}")

# Load the SFARI effect sizes file
sfari_effect_sizes_file = results_dir / "sfari_variant_effect_sizes.csv"

if sfari_effect_sizes_file.exists():
    sfari_df = pd.read_csv(sfari_effect_sizes_file)
    
    print(f"\nSFARI file shape: {sfari_df.shape}")
    print(f"Columns: {list(sfari_df.columns)}")
    
    # Print first 5 rows
    print("\nFirst 5 rows of SFARI data:")
    print(sfari_df.head(5))
    
    # Print some unique features to see the structure
    print("\nSample of unique SFARI features:")
    print(sfari_df['feature'].sample(5).values)
    
    # Count significant findings
    sig_count = sfari_df['significant'].sum()
    print(f"\nNumber of significant SFARI findings: {sig_count} out of {len(sfari_df)}")
else:
    print(f"File not found: {sfari_effect_sizes_file}")


#%%
######################################################
# Step 3: Create Forest Plot of Variant Effects
###################################################### 
# Load libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D

def create_cluster_association_plot(effect_sizes_df,
                                    variant_subset='rare_coding',
                                    gene_type='all',
                                    display_plot=True,
                                    output_file=None):
    """
    Create a forest plot similar to Huang et al. showing gene variant categories and their
    association with clusters.  Any CI that would extend beyond the plotted x-axis
    is automatically trimmed for readability.
    """
    df = effect_sizes_df.copy()

    # Extract components from feature string
    df[['Test','Gene_Tolerance','Category']] = df['feature'].str.split(' - ', expand=True)
    
    # Filter by gene_type
    if gene_type == 'sfari':
        df = df[df['Category'].str.contains('SFARI')]

    # Filter by variant_subset
    if variant_subset == 'common_coding':
        types = ['PTV','Missense (CADD>30)','Missense (CADD 20-30)',
                 'Missense (CADD 0-20)','Synonymous']
        title_prefix = "Common Coding Variants"
    elif variant_subset == 'rare_coding':
        types = ['PTV','Missense (CADD>30)','Missense (CADD 20-30)',
                 'Missense (CADD 0-20)','Synonymous']
        title_prefix = "Rare Coding Variants"
    elif variant_subset == 'rare_noncoding':
        types = ['Enhancer','Promoter','Enhancer,Promoter']
        df = df[df['Gene_Tolerance'].isin(['All genes','Eigen > 0','Eigen <= 0'])]
        title_prefix = "Rare Noncoding Variants"
    elif variant_subset == 'rare_all':
        types = ['PTV','Missense (CADD>30)','Missense (CADD 20-30)',
                 'Missense (CADD 0-20)','Synonymous',
                 'Enhancer','Promoter','Enhancer,Promoter']
        title_prefix = "All Rare Variants"
    else:
        raise ValueError(f"Invalid variant_subset: {variant_subset}")
    df = df[df['Test'].isin(types)]
    
    # Build feature_label
    df['feature_label'] = df['Test'] + ' - ' + df['Gene_Tolerance']
    
    # For each feature, pick strongest cluster (abs log2fc)
    feature_clusters, feature_p_values, feature_log2fc = {}, {}, {}
    feature_ci_lower, feature_ci_upper, feature_significant = {}, {}, {}
    for flab in df['feature_label'].unique():
        sub = df[df['feature_label']==flab]
        idx = sub['log2fc'].abs().idxmax()
        row = sub.loc[idx]
        feature_clusters[flab]   = row['cluster']
        feature_p_values[flab]   = row['p_value']
        feature_log2fc[flab]     = row['log2fc']
        feature_ci_lower[flab]   = row['ci_lower']
        feature_ci_upper[flab]   = row['ci_upper']
        feature_significant[flab] = bool(row['significant'])
    
    # Determine axis limits and clamp thresholds
    default_min, default_max = -0.3, 0.05
    data_min = min(v for v in feature_ci_lower.values() if not np.isnan(v))
    data_max = max(v for v in feature_ci_upper.values() if not np.isnan(v))
    x_min = min(default_min, data_min)
    x_max = max(default_max, data_max)

    # Define ordering of features
    coding_order = ['PTV','Missense (CADD>30)','Missense (CADD 20-30)',
                    'Missense (CADD 0-20)','Synonymous']
    noncoding_order = ['Enhancer','Promoter','Enhancer,Promoter']
    if variant_subset in ['common_coding','rare_coding']:
        test_order = coding_order
        tol_order = ['All genes','pLI >= 0.9','pLI < 0.9']
    elif variant_subset == 'rare_noncoding':
        test_order = noncoding_order
        tol_order = ['All genes','Eigen > 0','Eigen <= 0']
    else:
        test_order = coding_order + noncoding_order
        tol_order = ['All genes','pLI >= 0.9','pLI < 0.9','Eigen > 0','Eigen <= 0']
    
    unique_features = []
    for t in test_order:
        for tol in tol_order:
            lab = f"{t} - {tol}"
            if lab in feature_clusters:
                unique_features.append(lab)
    if not unique_features:
        raise ValueError("No features to plot after filtering.")
    
    # Map y positions (reverse so PTV at top)
    y_positions = {f: len(unique_features)-1-i for i,f in enumerate(unique_features)}
    
    # Start figure
    fig, ax = plt.subplots(figsize=(10, len(unique_features)*0.5 + 3))
    
    # Colors & markers
    cols = {1:'royalblue',2:'orange',3:'teal',4:'crimson'}
    mks  = {1:'o',       2:'s',     3:'D',   4:'^'}
    
    # Plot each feature
    for feat in unique_features:
        cl = feature_clusters[feat]
        y  = y_positions[feat]
        lo = feature_ci_lower[feat]
        hi = feature_ci_upper[feat]
        # clamp
        lo = max(lo, x_min)
        hi = min(hi, x_max)
        ax.plot([lo,hi],[y,y],color=cols[cl],alpha=0.5,lw=2)
        
        log2fc = feature_log2fc[feat]
        sig    = feature_significant[feat]
        sz     = 8 if sig else 6
        face   = cols[cl] if sig else 'white'
        edge   = cols[cl]
        ax.plot(log2fc, y,
                marker=mks[cl], markersize=sz,
                markerfacecolor=face,
                markeredgecolor=edge,
                alpha=0.8)
    
    # Zero line
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # Y-ticks
    rev = list(reversed(unique_features))
    ax.set_yticks([y_positions[f] for f in rev])
    labels = [f"{f}{' *' if feature_significant[f] else ''}" for f in rev]
    ax.set_yticklabels(labels)
    
    # Horizontal separators
    current = None
    for i,f in enumerate(rev):
        t = f.split(' - ')[0]
        if t!=current:
            current=t
            if i>0:
                ax.axhline(i-0.5, color='gray', linestyle='-', alpha=0.3, lw=0.5)
    
    # Axes & legend
    ax.set_xlabel('Beta for Cluster Assignment', fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.text(x_min, -0.5, '← Depleted', ha='left', va='center')
    ax.text(x_max, -0.5, 'Enriched →', ha='right', va='center')
    
    legend_elems = [ Line2D([0],[0],marker=mks[i], color=cols[i], markersize=8, label=f'Cluster {i}')
                     for i in sorted(cols) ]
    ax.legend(handles=legend_elems, loc='upper right', frameon=True)
    
    gene_str = "SFARI Genes" if gene_type=='sfari' else "All Genes"
    ax.set_title(f"{title_prefix} ({gene_str})", fontsize=14)
    plt.figtext(0.1, 0.02, "* p < 0.05", va='bottom')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if display_plot:
        plt.show()
    
    return fig, ax


# Load the required data
results_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/cluster_summaries')
plots_dir   = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/plots')

# Read CSVs
common_coding_all   = pd.read_csv(results_dir / "all_genes_common_coding_effect_sizes.csv")
rare_coding_all     = pd.read_csv(results_dir / "all_genes_rare_coding_effect_sizes.csv")
rare_noncoding_all  = pd.read_csv(results_dir / "all_genes_rare_noncoding_effect_sizes.csv")
sfari_common_coding = pd.read_csv(results_dir / "sfari_common_coding_effect_sizes.csv")
sfari_rare_coding   = pd.read_csv(results_dir / "sfari_rare_coding_effect_sizes.csv")
sfari_rare_noncoding= pd.read_csv(results_dir / "sfari_rare_noncoding_effect_sizes.csv")

# Combine rare sets
rare_all_genes  = pd.concat([rare_coding_all, rare_noncoding_all], ignore_index=True)
rare_sfari_genes= pd.concat([sfari_rare_coding, sfari_rare_noncoding], ignore_index=True)

# 1. Common Coding - All Genes
create_cluster_association_plot(
    common_coding_all,
    variant_subset='common_coding',
    gene_type='all',
    output_file=plots_dir/"cluster_assoc_common_coding_all.png"
)

# 2. Rare Coding - All Genes
create_cluster_association_plot(
    rare_coding_all,
    variant_subset='rare_coding',
    gene_type='all',
    output_file=plots_dir/"cluster_assoc_rare_coding_all.png"
)

# 3. Rare Noncoding - All Genes
create_cluster_association_plot(
    rare_noncoding_all,
    variant_subset='rare_noncoding',
    gene_type='all',
    output_file=plots_dir/"cluster_assoc_rare_noncoding_all.png"
)

# 4. All Rare Variants - All Genes
create_cluster_association_plot(
    rare_all_genes,
    variant_subset='rare_all',
    gene_type='all',
    output_file=plots_dir/"cluster_assoc_rare_all_genes.png"
)

# 5. Common Coding - SFARI Genes
create_cluster_association_plot(
    sfari_common_coding,
    variant_subset='common_coding',
    gene_type='sfari',
    output_file=plots_dir/"cluster_assoc_common_coding_sfari.png"
)

# 6. Rare Coding - SFARI Genes
create_cluster_association_plot(
    sfari_rare_coding,
    variant_subset='rare_coding',
    gene_type='sfari',
    output_file=plots_dir/"cluster_assoc_rare_coding_sfari.png"
)

# 7. Rare Noncoding - SFARI Genes
create_cluster_association_plot(
    sfari_rare_noncoding,
    variant_subset='rare_noncoding',
    gene_type='sfari',
    output_file=plots_dir/"cluster_assoc_rare_noncoding_sfari.png"
)

# 8. All Rare Variants - SFARI Genes
create_cluster_association_plot(
    rare_sfari_genes,
    variant_subset='rare_all',
    gene_type='sfari',
    output_file=plots_dir/"cluster_assoc_rare_sfari_genes.png"
)

print("All plots created successfully!")


#%% 
# Option 2: Plotting with all 4 clusters  
def create_improved_forest_plot(effect_sizes_df, variant_subset='rare_coding', gene_type='all', 
                               display_plot=True, output_file=None):
    """
    Create a forest plot similar to Huang et al. with proper hierarchical ordering.
    
    Args:
        effect_sizes_df: DataFrame with effect sizes
        variant_subset: 'common_coding', 'rare_coding', 'rare_noncoding', or 'rare_all'
        gene_type: 'all' or 'sfari'
        display_plot: Whether to display the plot (for interactive mode)
        output_file: Path to save the plot
    """
    # Make a copy to avoid modifying the original
    df = effect_sizes_df.copy()
    
    # Extract components from feature string
    df[['Test', 'Gene_Tolerance', 'Category']] = df['feature'].str.split(' - ', expand=True)
    
    # Filter by gene type if specified
    if gene_type == 'sfari':
        # Filter for any SFARI category
        df = df[df['Category'].str.contains('SFARI')]
    
    # Filter by variant subset and set title_prefix
    if variant_subset == 'common_coding':
        # Filter for common coding variants
        coding_types = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)', 
                       'Missense (CADD 0-20)', 'Synonymous']
        df = df[df['Test'].isin(coding_types)]
        title_prefix = "Common Coding Variants"
    elif variant_subset == 'rare_coding':
        # Filter for rare coding variants
        coding_types = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)', 
                       'Missense (CADD 0-20)', 'Synonymous']
        df = df[df['Test'].isin(coding_types)]
        title_prefix = "Rare Coding Variants"
    elif variant_subset == 'rare_noncoding':
        # Filter for rare noncoding variants
        noncoding_types = ['Enhancer', 'Promoter', 'Enhancer,Promoter']
        df = df[df['Test'].isin(noncoding_types) & 
               (df['Gene_Tolerance'].isin(['All genes', 'Eigen > 0', 'Eigen <= 0']))]
        title_prefix = "Rare Noncoding Variants"
    elif variant_subset == 'rare_all':
        # Include all rare variants (coding and noncoding)
        # Exclude "Other" category
        all_types = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)', 
                   'Missense (CADD 0-20)', 'Synonymous', 'Enhancer', 'Promoter', 'Enhancer,Promoter']
        df = df[df['Test'].isin(all_types)]
        title_prefix = "All Rare Variants"
    else:
        raise ValueError(f"Invalid variant_subset: {variant_subset}")
    
    # Define ordering for tests (in reverse order to match Huang et al.)
    coding_order = [
        'PTV',
        'Missense (CADD>30)',
        'Missense (CADD 20-30)',
        'Missense (CADD 0-20)',
        'Synonymous'
    ]
    
    noncoding_order = [
        'Enhancer',
        'Promoter',
        'Enhancer,Promoter'
    ]
    
    # Choose the right test order based on variant subset
    if variant_subset in ['common_coding', 'rare_coding']:
        test_order = coding_order
    elif variant_subset == 'rare_noncoding':
        test_order = noncoding_order
    else:  # rare_all
        test_order = coding_order + noncoding_order
    
    # Define ordering for gene tolerance
    if variant_subset in ['common_coding', 'rare_coding']:
        tolerance_order = ['All genes', 'pLI >= 0.9', 'pLI < 0.9']
    elif variant_subset == 'rare_noncoding':
        tolerance_order = ['All genes', 'Eigen > 0', 'Eigen <= 0']
    else:  # rare_all - include all tolerance types
        tolerance_order = ['All genes', 'pLI >= 0.9', 'pLI < 0.9', 'Eigen > 0', 'Eigen <= 0']
    
    # Create a unique feature label that combines Test and Gene_Tolerance
    df['feature_label'] = df['Test'] + ' - ' + df['Gene_Tolerance']
    
    # Get unique feature labels in the correct hierarchical order
    unique_features = []
    for test in test_order:
        for tolerance in tolerance_order:
            feature_label = f"{test} - {tolerance}"
            if feature_label in df['feature_label'].values:
                unique_features.append(feature_label)
    
    # Check if we have any data left after filtering
    if not unique_features:
        print(f"No data found for variant_subset={variant_subset}, gene_type={gene_type}")
        return None, None
    
    # Create a mapping for y positions (reverse order to match Huang et al. with PTV at top)
    y_positions = {feature: len(unique_features) - 1 - i for i, feature in enumerate(unique_features)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(unique_features) * 0.5 + 3))
    
    # Set up colors and markers for clusters
    cluster_colors = {
        1: 'royalblue',
        2: 'orange',
        3: 'teal',
        4: 'crimson'
    }
    
    cluster_markers = {
        1: 'o',
        2: 's',
        3: 'D',
        4: '^'
    }
    
    # Create a dictionary to track significant features
    significant_features = {}
    
    # Plot the data points and error bars for each cluster
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        
        for _, row in cluster_df.iterrows():
            if row['feature_label'] in y_positions:
                y_pos = y_positions[row['feature_label']]
                
                # Add small offset for each cluster to avoid overlap
                offset = (cluster - 2.5) * 0.15
                
                # Plot confidence interval
                ax.plot([row['ci_lower'], row['ci_upper']], [y_pos + offset, y_pos + offset], 
                       color=cluster_colors[cluster], alpha=0.5, linewidth=1.5)
                
                # Plot point estimate
                is_significant = row['significant']
                marker_size = 8 if is_significant else 6
                ax.plot(row['log2fc'], y_pos + offset, 
                       marker=cluster_markers[cluster], color=cluster_colors[cluster], 
                       markersize=marker_size, alpha=0.8)
                
                # Track significant findings
                if is_significant:
                    if row['feature_label'] not in significant_features:
                        significant_features[row['feature_label']] = 0
                    significant_features[row['feature_label']] += 1
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Customize the plot
    reversed_unique_features = list(reversed(unique_features))
    ax.set_yticks(list(reversed(list(y_positions.values()))))
    
    # Create y-tick labels with significance indicators
    y_labels = []
    for feature in reversed_unique_features:
        label = feature
        if feature in significant_features:
            label = f"{label} *"  # Just use a single asterisk for significance
        y_labels.append(label)
    
    ax.set_yticklabels(y_labels)
    
    # Add horizontal lines between gene categories
    current_test = None
    for i, feature in enumerate(reversed_unique_features):
        test = feature.split(' - ')[0]
        if current_test is None or test != current_test:
            current_test = test
            if i > 0:
                ax.axhline(y=i-0.5, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Set x-axis label
    ax.set_xlabel('Beta for Cluster Assignment', fontsize=12)
    
    # Adjust x-axis limits to match scale similar to Huang et al
    # Detect min/max values but limit to reasonable range
    x_min = max(-3, df['ci_lower'].min())
    x_max = min(3, df['ci_upper'].max())
    ax.set_xlim([x_min, x_max])
    
    # Add labels for "Depleted" and "Enriched"
    ax.text(ax.get_xlim()[0], -0.5, '← Depleted', ha='left', va='center', fontsize=10)
    ax.text(ax.get_xlim()[1], -0.5, 'Enriched →', ha='right', va='center', fontsize=10)
    
    # Create a legend for clusters
    legend_elements = [
        Line2D([0], [0], marker=cluster_markers[i], color='w', markerfacecolor=cluster_colors[i], 
              markersize=8, label=f'Cluster {i}') for i in sorted(cluster_colors.keys())
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    # Set title based on variant and gene type
    gene_str = "SFARI Genes" if gene_type == 'sfari' else "All Genes"
    plot_title = f"{title_prefix} ({gene_str})"
    plt.title(plot_title, fontsize=14)
    
    # Add subtitle explaining significance (further to the left)
    plt.figtext(0.1, 0.02, "* p < 0.05", ha='left', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Display the plot if requested
    if display_plot:
        plt.show()
    
    return fig, ax


# Load the required data
results_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/cluster_summaries')
plots_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/plots')

# Load datasets
common_coding_all = pd.read_csv(results_dir / "all_genes_common_coding_effect_sizes.csv")
rare_coding_all = pd.read_csv(results_dir / "all_genes_rare_coding_effect_sizes.csv")
rare_noncoding_all = pd.read_csv(results_dir / "all_genes_noncoding_effect_sizes.csv")
sfari_common_coding = pd.read_csv(results_dir / "sfari_common_coding_effect_sizes.csv")
sfari_rare_coding = pd.read_csv(results_dir / "sfari_rare_coding_effect_sizes.csv")
sfari_rare_noncoding = pd.read_csv(results_dir / "sfari_noncoding_effect_sizes.csv")

# Create combined rare dataset for all genes
rare_all_genes = pd.concat([rare_coding_all, rare_noncoding_all], ignore_index=True)

# Create combined rare dataset for SFARI genes
rare_sfari_genes = pd.concat([sfari_rare_coding, sfari_rare_noncoding], ignore_index=True)

# 1. Common Coding - All Genes
create_improved_forest_plot(
    common_coding_all,
    variant_subset='common_coding',
    gene_type='all',
    output_file=plots_dir / "forest_plot_common_coding_all.png"
)

# 2. Rare Coding - All Genes
create_improved_forest_plot(
    rare_coding_all,
    variant_subset='rare_coding',
    gene_type='all',
    output_file=plots_dir / "forest_plot_rare_coding_all.png"
)

# 3. Rare Noncoding - All Genes
create_improved_forest_plot(
    rare_noncoding_all,
    variant_subset='rare_noncoding',
    gene_type='all',
    output_file=plots_dir / "forest_plot_rare_noncoding_all.png"
)

# 4. All Rare Variants - All Genes
create_improved_forest_plot(
    rare_all_genes,
    variant_subset='rare_all',
    gene_type='all',
    output_file=plots_dir / "forest_plot_rare_all_genes.png"
)

# 5. Common Coding - SFARI Genes
create_improved_forest_plot(
    sfari_common_coding,
    variant_subset='common_coding',
    gene_type='sfari',
    output_file=plots_dir / "forest_plot_common_coding_sfari.png"
)

# 6. Rare Coding - SFARI Genes
create_improved_forest_plot(
    sfari_rare_coding,
    variant_subset='rare_coding',
    gene_type='sfari',
    output_file=plots_dir / "forest_plot_rare_coding_sfari.png"
)

# 7. Rare Noncoding - SFARI Genes
create_improved_forest_plot(
    sfari_rare_noncoding,
    variant_subset='rare_noncoding',
    gene_type='sfari',
    output_file=plots_dir / "forest_plot_rare_noncoding_sfari.png"
)

# 8. All Rare Variants - SFARI Genes
create_improved_forest_plot(
    rare_sfari_genes,
    variant_subset='rare_all',
    gene_type='sfari',
    output_file=plots_dir / "forest_plot_rare_sfari_genes.png"
)

print("All plots created successfully!")


#%%
# Option 3: draw with all clusters without CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D

def create_improved_forest_plot(effect_sizes_df, variant_subset='rare_coding', gene_type='all', 
                               display_plot=True, output_file=None, sfari_significant_only=False):
    """
    Create a forest plot similar to Huang et al. with proper hierarchical ordering.
    Shows both significant (p < 0.05) and trending (p < 0.1) results.
    
    Args:
        effect_sizes_df: DataFrame with effect sizes
        variant_subset: 'common_coding', 'rare_coding', 'rare_noncoding', or 'rare_all'
        gene_type: 'all' or 'sfari'
        display_plot: Whether to display the plot (for interactive mode)
        output_file: Path to save the plot
        sfari_significant_only: If True and gene_type='sfari', only plot significant SFARI findings
    """
    # Make a copy to avoid modifying the original
    df = effect_sizes_df.copy()
    
    # Extract components from feature string
    df[['Test', 'Gene_Tolerance', 'Category']] = df['feature'].str.split(' - ', expand=True)
    
    # Create the feature_label column (ensure it exists before any filtering)
    df['feature_label'] = df['Test'] + ' - ' + df['Gene_Tolerance']
    
    # Filter by gene type if specified
    if gene_type == 'sfari':
        # Filter ONLY for the "All SFARI" category, not individual SFARI scores
        df = df[df['Category'] == 'All SFARI']
        
        # If sfari_significant_only is True, only keep significant findings
        if sfari_significant_only:
            # Find significant features first (now including trending)
            significant_features = df[(df['significant'] == True) | (df['trending'] == True)]['feature_label'].unique()
            if len(significant_features) == 0:
                # Try to find specific known significant SFARI findings directly
                special_sfari_df = effect_sizes_df.copy()
                special_sfari_df[['Test', 'Gene_Tolerance', 'Category']] = special_sfari_df['feature'].str.split(' - ', expand=True)
                
                # Look for the known significant findings
                matches = []
                
                # Missense (CADD 0-20) - All genes - SFARI 3
                missense_sfari3 = special_sfari_df[
                    (special_sfari_df['Test'] == 'Missense (CADD 0-20)') & 
                    (special_sfari_df['Gene_Tolerance'] == 'All genes') & 
                    (special_sfari_df['Category'] == 'SFARI 3')
                ]
                if not missense_sfari3.empty:
                    matches.append(missense_sfari3)
                
                # Missense (CADD 0-20) - pLI < 0.9 - SFARI 3
                missense_pli_sfari3 = special_sfari_df[
                    (special_sfari_df['Test'] == 'Missense (CADD 0-20)') & 
                    (special_sfari_df['Gene_Tolerance'] == 'pLI < 0.9') & 
                    (special_sfari_df['Category'] == 'SFARI 3')
                ]
                if not missense_pli_sfari3.empty:
                    matches.append(missense_pli_sfari3)
                
                # Promoter - Eigen > 0 - SFARI S
                promoter_sfariS = special_sfari_df[
                    (special_sfari_df['Test'] == 'Promoter') & 
                    (special_sfari_df['Gene_Tolerance'] == 'Eigen > 0') & 
                    (special_sfari_df['Category'] == 'SFARI S')
                ]
                if not promoter_sfariS.empty:
                    matches.append(promoter_sfariS)
                
                if matches:
                    df = pd.concat(matches)
                    # Create special feature labels for these
                    df['feature_label'] = df.apply(lambda row: f"{row['Test']} - {row['Gene_Tolerance']} ({row['Category']})", axis=1)
                else:
                    print("No known significant SFARI findings found. Using all data.")
                    df = effect_sizes_df.copy()
                    df[['Test', 'Gene_Tolerance', 'Category']] = df['feature'].str.split(' - ', expand=True)
                    df['feature_label'] = df['Test'] + ' - ' + df['Gene_Tolerance']
    
    # Filter by variant subset and set title_prefix
    if not sfari_significant_only:  # Skip this for the SFARI significant-only plot
        if variant_subset == 'common_coding':
            # Filter for common coding variants
            coding_types = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)', 
                           'Missense (CADD 0-20)', 'Synonymous']
            df = df[df['Test'].isin(coding_types)]
            title_prefix = "Common Coding Variants"
        elif variant_subset == 'rare_coding':
            # Filter for rare coding variants
            coding_types = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)', 
                           'Missense (CADD 0-20)', 'Synonymous']
            df = df[df['Test'].isin(coding_types)]
            title_prefix = "Rare Coding Variants"
        elif variant_subset == 'rare_noncoding':
            # Filter for rare noncoding variants
            noncoding_types = ['Enhancer', 'Promoter', 'Enhancer,Promoter']
            df = df[df['Test'].isin(noncoding_types) & 
                   (df['Gene_Tolerance'].isin(['All genes', 'Eigen > 0', 'Eigen <= 0']))]
            title_prefix = "Rare Noncoding Variants"
        elif variant_subset == 'rare_all':
            # Include all rare variants (coding and noncoding)
            # Exclude "Other" category
            all_types = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)', 
                       'Missense (CADD 0-20)', 'Synonymous', 'Enhancer', 'Promoter', 'Enhancer,Promoter']
            df = df[df['Test'].isin(all_types)]
            title_prefix = "All Rare Variants"
        else:
            raise ValueError(f"Invalid variant_subset: {variant_subset}")
    else:
        title_prefix = "Significant & Trending SFARI Variants"
    
    # Define ordering for tests (in reverse order to match Huang et al.)
    coding_order = [
        'PTV',
        'Missense (CADD>30)',
        'Missense (CADD 20-30)',
        'Missense (CADD 0-20)',
        'Synonymous'
    ]
    
    noncoding_order = [
        'Enhancer',
        'Promoter',
        'Enhancer,Promoter'
    ]
    
    # For significant-only mode, just use the feature_labels directly
    if sfari_significant_only:
        unique_features = sorted(df['feature_label'].unique())
    else:
        # Choose the right test order based on variant subset
        if variant_subset in ['common_coding', 'rare_coding']:
            test_order = coding_order
        elif variant_subset == 'rare_noncoding':
            test_order = noncoding_order
        else:  # rare_all
            test_order = coding_order + noncoding_order
        
        # Define ordering for gene tolerance
        if variant_subset in ['common_coding', 'rare_coding']:
            tolerance_order = ['All genes', 'pLI >= 0.9', 'pLI < 0.9']
        elif variant_subset == 'rare_noncoding':
            tolerance_order = ['All genes', 'Eigen > 0', 'Eigen <= 0']
        else:  # rare_all
            tolerance_order = ['All genes', 'pLI >= 0.9', 'pLI < 0.9', 'Eigen > 0', 'Eigen <= 0']
        
        # Get unique feature labels in the correct hierarchical order
        unique_features = []
        for test in test_order:
            for tolerance in tolerance_order:
                feature_label = f"{test} - {tolerance}"
                if feature_label in df['feature_label'].values:
                    unique_features.append(feature_label)
    
    # Check if we have any data left after filtering
    if not unique_features:
        print(f"No data found for variant_subset={variant_subset}, gene_type={gene_type}")
        return None, None
    
    # Create a mapping for y positions (reverse order to match Huang et al. with PTV at top)
    y_positions = {feature: len(unique_features) - 1 - i for i, feature in enumerate(unique_features)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(unique_features) * 0.5 + 3))
    
    # Set up colors and markers for clusters
    cluster_colors = {
        1: 'royalblue',
        2: 'orange',
        3: 'teal',
        4: 'crimson'
    }
    
    cluster_markers = {
        1: 'o',
        2: 's',
        3: 'D',
        4: '^'
    }
    
    # Create dictionaries to track significant and trending features
    significant_features = {}
    trending_features = {}
    
    # Plot the data points for each cluster (NO confidence intervals)
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        
        for _, row in cluster_df.iterrows():
            feature_label = row['feature_label']
            if feature_label in y_positions:
                y_pos = y_positions[feature_label]
                
                # Add small offset for each cluster to avoid overlap
                offset = (cluster - 2.5) * 0.15
                
                # Get significance and trending status
                is_significant = row['significant']
                is_trending = row.get('trending', False)  # Handle cases where trending column might not exist
                
                # Marker size based on significance
                marker_size = 9 if (is_significant or is_trending) else 7
                marker_style = cluster_markers[cluster]
                
                # Use filled marker for significant and trending results, hollow for non-significant
                if is_significant:
                    ax.plot(row['log2fc'], y_pos + offset, 
                           marker=marker_style, color=cluster_colors[cluster], 
                           markersize=marker_size, alpha=0.9)
                    # Track significant findings
                    if feature_label not in significant_features:
                        significant_features[feature_label] = 0
                    significant_features[feature_label] += 1
                elif is_trending:
                    ax.plot(row['log2fc'], y_pos + offset, 
                           marker=marker_style, color=cluster_colors[cluster], 
                           markersize=marker_size, alpha=0.7)
                    # Track trending findings
                    if feature_label not in trending_features:
                        trending_features[feature_label] = 0
                    trending_features[feature_label] += 1
                else:
                    ax.plot(row['log2fc'], y_pos + offset, 
                           marker=marker_style, color=cluster_colors[cluster], 
                           markersize=marker_size, mfc='white', alpha=0.7)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Customize the plot
    reversed_unique_features = list(reversed(unique_features))
    ax.set_yticks(list(reversed(list(y_positions.values()))))
    
    # Create y-tick labels with significance indicators
    y_labels = []
    for feature in reversed_unique_features:
        label = feature
        
        # In significant-only mode, we already know everything is significant or trending
        if not sfari_significant_only:
            if feature in significant_features:
                label = f"{label} **"  # Double asterisk for significance (p < 0.05)
            elif feature in trending_features:
                label = f"{label} *"   # Single asterisk for trending (p < 0.1)
        
        y_labels.append(label)
    
    ax.set_yticklabels(y_labels)
    
    # Add horizontal lines between gene categories
    if not sfari_significant_only:
        current_test = None
        for i, feature in enumerate(reversed_unique_features):
            test = feature.split(' - ')[0]
            if current_test is None or test != current_test:
                current_test = test
                if i > 0:
                    ax.axhline(y=i-0.5, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Set x-axis label
    ax.set_xlabel('Beta for Cluster Assignment', fontsize=12)
    
    # Set reasonable x-axis limits without the extreme CI values
    # Calculate a reasonable range based on the actual log2fc values
    log2fc_values = df['log2fc'].dropna().values
    if len(log2fc_values) > 0:
        x_min = np.percentile(log2fc_values, 1)  # 1st percentile
        x_max = np.percentile(log2fc_values, 99)  # 99th percentile
        
        # Add some padding
        x_range = x_max - x_min
        x_min = max(-0.5, x_min - 0.1 * x_range)
        x_max = min(0.5, x_max + 0.1 * x_range)
    else:
        x_min, x_max = -0.5, 0.5
    
    # Make sure we include zero
    x_min = min(x_min, -0.05)
    x_max = max(x_max, 0.05)
    
    ax.set_xlim([x_min, x_max])
    
    # Add labels for "Depleted" and "Enriched"
    ax.text(ax.get_xlim()[0], -0.5, '← Depleted', ha='left', va='center', fontsize=10)
    ax.text(ax.get_xlim()[1], -0.5, 'Enriched →', ha='right', va='center', fontsize=10)
    
    # Create a legend for clusters
    legend_elements = [
        Line2D([0], [0], marker=cluster_markers[i], color=cluster_colors[i], 
              markersize=8, label=f'Cluster {i}') for i in sorted(cluster_colors.keys())
    ]
    
    # For normal plots (not significant-only), add significance legend
    if not sfari_significant_only:
        legend_elements.append(Line2D([0], [0], marker='o', color='black', 
                                     markersize=8, label='Significant (p < 0.05)'))
        legend_elements.append(Line2D([0], [0], marker='o', color='black', 
                                     markersize=8, alpha=0.7, label='Trending (p < 0.1)'))
        legend_elements.append(Line2D([0], [0], marker='o', color='black', 
                                     markersize=8, mfc='white', label='Non-significant'))
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9)
    
    # Set title based on variant and gene type
    if sfari_significant_only:
        plot_title = "Significant & Trending SFARI Variant Associations"
    else:
        gene_str = "SFARI Genes" if gene_type == 'sfari' else "All Genes"
        plot_title = f"{title_prefix} ({gene_str})"
    
    plt.title(plot_title, fontsize=14)
    
    # Add subtitle explaining significance (further to the left)
    plt.figtext(0.1, 0.01, "** p < 0.05, * p < 0.1", ha='left', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Display the plot if requested
    if display_plot:
        plt.show()
    
    return fig, ax


# Load the required data
results_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/cluster_summaries')
plots_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/plots')

# Load datasets
common_coding_all = pd.read_csv(results_dir / "all_genes_common_coding_effect_sizes.csv")
rare_coding_all = pd.read_csv(results_dir / "all_genes_rare_coding_effect_sizes.csv")
rare_noncoding_all = pd.read_csv(results_dir / "all_genes_rare_noncoding_effect_sizes.csv")
sfari_common_coding = pd.read_csv(results_dir / "sfari_common_coding_effect_sizes.csv")
sfari_rare_coding = pd.read_csv(results_dir / "sfari_rare_coding_effect_sizes.csv")
sfari_rare_noncoding = pd.read_csv(results_dir / "sfari_rare_noncoding_effect_sizes.csv")

# Create combined rare dataset for all genes
rare_all_genes = pd.concat([rare_coding_all, rare_noncoding_all], ignore_index=True)

# Create combined rare dataset for SFARI genes
rare_sfari_genes = pd.concat([sfari_rare_coding, sfari_rare_noncoding], ignore_index=True)

# Combined SFARI dataset for the significant-only plot
all_sfari_genes = pd.concat([sfari_common_coding, sfari_rare_coding, sfari_rare_noncoding], ignore_index=True)

# 1. Common Coding - All Genes
create_improved_forest_plot(
    common_coding_all,
    variant_subset='common_coding',
    gene_type='all',
    output_file=plots_dir / "forest_plot_common_coding_all.png"
)

# 2. Rare Coding - All Genes
create_improved_forest_plot(
    rare_coding_all,
    variant_subset='rare_coding',
    gene_type='all',
    output_file=plots_dir / "forest_plot_rare_coding_all.png"
)

# 3. Rare Noncoding - All Genes
create_improved_forest_plot(
    rare_noncoding_all,
    variant_subset='rare_noncoding',
    gene_type='all',
    output_file=plots_dir / "forest_plot_rare_noncoding_all.png"
)

# 4. All Rare Variants - All Genes
create_improved_forest_plot(
    rare_all_genes,
    variant_subset='rare_all',
    gene_type='all',
    output_file=plots_dir / "forest_plot_rare_all_genes.png"
)

# 5. Common Coding - SFARI Genes
create_improved_forest_plot(
    sfari_common_coding,
    variant_subset='common_coding',
    gene_type='sfari',
    output_file=plots_dir / "forest_plot_common_coding_sfari.png"
)

# 6. Rare Coding - SFARI Genes
create_improved_forest_plot(
    sfari_rare_coding,
    variant_subset='rare_coding',
    gene_type='sfari',
    output_file=plots_dir / "forest_plot_rare_coding_sfari.png"
)

# 7. Rare Noncoding - SFARI Genes
create_improved_forest_plot(
    sfari_rare_noncoding,
    variant_subset='rare_noncoding',
    gene_type='sfari',
    output_file=plots_dir / "forest_plot_rare_noncoding_sfari.png"
)

# 8. All Rare Variants - SFARI Genes
create_improved_forest_plot(
    rare_sfari_genes,
    variant_subset='rare_all',
    gene_type='sfari',
    output_file=plots_dir / "forest_plot_rare_sfari_genes.png"
)

print("All plots created successfully!")
# %%