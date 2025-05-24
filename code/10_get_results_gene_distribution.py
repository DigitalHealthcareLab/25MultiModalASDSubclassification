"""
Analyze gene variant distribution across participant clusters.

This script:
1. Loads data for each participant
2. Analyzes coding and noncoding variants at the individual level
3. Aggregates results by cluster
4. Generates summary statistics for cluster comparison

Optimized version with parallel processing and memory optimizations.
"""

from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm
import concurrent.futures
import time
import warnings
warnings.filterwarnings('ignore')

# Import our utilities
from src.utils_gene_preproc import analyze_sfari_variants_by_burden

from src.utils_gene_dataloader import (
    analyze_coding_variants, analyze_noncoding_variants,
)

from src.utils_gene_cluster_compare import (
    generate_coding_cluster_summaries,
    generate_noncoding_cluster_summaries,
    create_coding_cluster_comparison_table, 
    create_noncoding_cluster_comparison_table, 
    calculate_rare_common_ratio
)

# Define base directories
wgs_dir = Path('/home/cko4/2024_asd_multimodal/data/2025_asd_wgs')
results_dir = wgs_dir / 'genetype_results' / 'cluster_analysis'
results_dir.mkdir(parents=True, exist_ok=True)

# Create directory for individual results
individual_results_dir = results_dir / 'individual_analyses'
individual_results_dir.mkdir(parents=True, exist_ok=True)

# Create directory for cluster summaries
cluster_summaries_dir = results_dir / 'cluster_summaries'
cluster_summaries_dir.mkdir(parents=True, exist_ok=True)

# Variant types
variant_types = {
    "3_1_Common_Coding": {"variant_type1": "Common", "variant_type2": "Coding"},
    "3_2_Rare_Coding": {"variant_type1": "Rare", "variant_type2": "Coding"},
    "3_3_Rare_Noncoding": {"variant_type1": "Rare", "variant_type2": "Noncoding"}
}

def load_cluster_info():
    """
    Load cluster assignments for participants.
    
    Returns:
        tuple: (cluster_df, participant_to_cluster) where:
            - cluster_df is a DataFrame with participant info and clusters
            - participant_to_cluster is a dict mapping participant IDs to cluster IDs
    """
    cluster_file = wgs_dir / 'dataframes' / 'participant_clinical_brain_cluster_df.csv'
    cluster_df = pd.read_csv(cluster_file)
    cluster_df = cluster_df.dropna(subset=['cluster'])
    cluster_df['cluster'] = cluster_df['cluster'].astype(int)
    cluster_df['id'] = cluster_df['id'].astype(str)
    participant_to_cluster = dict(zip(cluster_df['id'], cluster_df['cluster']))
    return cluster_df, participant_to_cluster


def process_participant(participant_data):
    """
    Process a single participant's data for parallel execution.
    
    Args:
        participant_data (tuple): (participant_id, cluster_id, reanalyze)
        
    Returns:
        tuple: (participant_id, coding_results, noncoding_results)
    """
    participant_id, cluster_id, reanalyze = participant_data
    
    # Standardize ID format
    if not participant_id.startswith("asd"):
        participant_id = f"asd{participant_id}"
    id_num = int(participant_id[3:])
    participant_id = f"asd{id_num:03d}"
    
    # Check if results already exist
    coding_file = individual_results_dir / f"{participant_id}_coding_analysis.pkl"
    noncoding_file = individual_results_dir / f"{participant_id}_noncoding_analysis.pkl"

    if not reanalyze and coding_file.exists() and noncoding_file.exists():
        print(f"Skipping {participant_id} — both outputs already exist")
        with open(coding_file, 'rb') as f:
            coding_result = pickle.load(f)
        with open(noncoding_file, 'rb') as f:
            noncoding_result = pickle.load(f)
        return participant_id, coding_result, noncoding_result

    coding_result = {}
    noncoding_result = {}

    ### Process coding variants
    if reanalyze or not coding_file.exists():
        print(f"\nAnalyzing coding variants for participant {participant_id} (Cluster {cluster_id})")
        
        # Analyze common coding
        common_coding_result = analyze_coding_variants(
            participant_id, '3_1_Common_Coding', wgs_dir, variant_types
        )
        
        # Analyze rare coding
        rare_coding_result = analyze_coding_variants(
            participant_id, '3_2_Rare_Coding', wgs_dir, variant_types
        )
        
        coding_result = {}
        
        if common_coding_result:
            common_burden = analyze_sfari_variants_by_burden(
                participant_id,
                variant_type="3_1_Common_Coding",
                categorized_df=common_coding_result['categorized_df']
            )
            coding_result['common_coding'] = {
                'summary_df': common_burden['summary_df'],
                'high_priority_genes': common_burden['high_priority_genes'],
                'full_df': common_coding_result['categorized_df']  
            }
        
        if rare_coding_result:
            rare_burden = analyze_sfari_variants_by_burden(
                participant_id,
                variant_type="3_2_Rare_Coding",
                categorized_df=rare_coding_result['categorized_df']
            )
            coding_result['rare_coding'] = {
                'summary_df': rare_burden['summary_df'],
                'high_priority_genes': rare_burden['high_priority_genes'],
                'full_df': rare_coding_result['categorized_df']  
            }
        
        if coding_result:
            with open(coding_file, 'wb') as f:
                pickle.dump(coding_result, f)
    else:
        try:
            with open(coding_file, 'rb') as f:
                coding_result = pickle.load(f)
        except Exception as e:
            print(f"Error loading existing coding results for {participant_id}: {e}")

    ### Process noncoding variants
    if reanalyze or not noncoding_file.exists():
        print(f"\n Analyzing noncoding variants for participant {participant_id} (Cluster {cluster_id})")
        
        noncoding_result = analyze_noncoding_variants(
            participant_id, '3_3_Rare_Noncoding', wgs_dir, variant_types
        )
        
        if noncoding_result:
            slim_result = {
                'participant_id': noncoding_result['participant_id'],
                'total_variants': noncoding_result['total_variants'],
                'sfari_variants': noncoding_result['sfari_variants'],
                'percent_sfari': noncoding_result['percent_sfari'],
                'region_distribution': noncoding_result['region_distribution'],
                'impact_distribution': noncoding_result['impact_distribution'],
                'regulatory_variants': noncoding_result['regulatory_variants'],
                'sfari_impact_distribution': noncoding_result['sfari_impact_distribution'],
                'region_impact_distribution': noncoding_result['region_impact_distribution'],
                'filtered_df': noncoding_result['categorized_df'][[
                    'GeneName', 'Simplified_Region', 'Regulatory_Region',
                    'eigen_bin', 'SFARI_gene_score', 'regulatory_flag'
                ]]
            }
            with open(noncoding_file, 'wb') as f:
                pickle.dump(slim_result, f)
            noncoding_result = slim_result
    else:
        try:
            with open(noncoding_file, 'rb') as f:
                noncoding_result = pickle.load(f)
        except Exception as e:
            print(f"Error loading existing noncoding results for {participant_id}: {e}")
    
    return participant_id, coding_result, noncoding_result


def main(num_workers=6, reanalyze=False, specific_participants=None):
    """
    Main function to run the full analysis pipeline.
    
    Args:
        num_workers (int): Number of parallel workers
        reanalyze (bool): If True, reanalyze participants even if results exist
        specific_participants (list, optional): List of specific participant IDs to analyze.
                                               If None, analyze all participants.
    """
    start_time = time.time()
    
    # Load cluster information
    cluster_df, participant_to_cluster = load_cluster_info()
    print(f"Loaded cluster information for {len(participant_to_cluster)} participants")
    
    # Filter for specific participants if provided
    if specific_participants:
        participant_to_cluster = {pid: cluster for pid, cluster in participant_to_cluster.items() 
                                 if pid in specific_participants or f"asd{pid}" in specific_participants}
        print(f"Filtered to {len(participant_to_cluster)} specific participants")
    
    # Prepare participant data for processing
    participant_data = [(pid, cid, reanalyze) for pid, cid in participant_to_cluster.items()]
    
    # 1. Individual-level analysis
    coding_results = {}
    noncoding_results = {}
    
    # Use parallel processing for performance
    if num_workers <= 1:
        # Sequential processing (for debugging or small datasets)
        for data in tqdm(participant_data, desc="Analyzing participants"):
            participant_id, coding_result, noncoding_result = process_participant(data)
            coding_results[participant_id] = coding_result
            noncoding_results[participant_id] = noncoding_result
    else:
        # Parallel processing for performance
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_participant, data) for data in participant_data]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                               total=len(futures),
                               desc="Analyzing participants"):
                try:
                    participant_id, coding_result, noncoding_result = future.result()
                    coding_results[participant_id] = coding_result
                    noncoding_results[participant_id] = noncoding_result
                except Exception as e:
                    print(f"Error processing participant: {e}")
    
    individual_time = time.time()
    print(f"Individual analysis completed in {individual_time - start_time:.2f} seconds")
    
    # 2. Cluster-level summaries
    print("\n\nGenerating cluster-level summaries...")
    
    # Separate common and rare coding results
    common_coding_results = {pid: res.get('common_coding') for pid, res in coding_results.items() 
                            if res and 'common_coding' in res}
    rare_coding_results = {pid: res.get('rare_coding') for pid, res in coding_results.items() 
                          if res and 'rare_coding' in res}
    
    # Generate cluster summaries
    common_coding_summaries = generate_coding_cluster_summaries(common_coding_results, participant_to_cluster)
    rare_coding_summaries = generate_coding_cluster_summaries(rare_coding_results, participant_to_cluster)
    
    # Use optimized noncoding cluster summaries
    noncoding_summaries = generate_noncoding_cluster_summaries(noncoding_results, participant_to_cluster)
    
    # 3. Save cluster summaries
    with open(cluster_summaries_dir / 'common_coding_cluster_summaries.pkl', 'wb') as f:
        pickle.dump(common_coding_summaries, f)
    
    with open(cluster_summaries_dir / 'rare_coding_cluster_summaries.pkl', 'wb') as f:
        pickle.dump(rare_coding_summaries, f)
    
    with open(cluster_summaries_dir / 'noncoding_cluster_summaries.pkl', 'wb') as f:
        pickle.dump(noncoding_summaries, f)
    
    summary_time = time.time()
    print(f"Cluster summaries completed in {summary_time - individual_time:.2f} seconds")
    
    # 4. Generate comparison tables
    print("\n\nGenerating cluster comparison tables...")
    
    # Create summary table for common coding variants
    common_coding_table = create_coding_cluster_comparison_table(common_coding_summaries, 'common_coding')
    common_coding_table.to_csv(cluster_summaries_dir / 'common_coding_cluster_comparison.csv', index=False)
    
    # Create summary table for rare coding variants
    rare_coding_table = create_coding_cluster_comparison_table(rare_coding_summaries, 'rare_coding')
    rare_coding_table.to_csv(cluster_summaries_dir / 'rare_coding_cluster_comparison.csv', index=False)
    
    # Create summary table for noncoding variants
    noncoding_table = create_noncoding_cluster_comparison_table(noncoding_summaries)
    noncoding_table.to_csv(cluster_summaries_dir / 'noncoding_cluster_comparison.csv', index=False)
    
    # 5. Calculate rare:common ratio
    # (1) Overall rare:common (all variants)
    rare_common_ratio = calculate_rare_common_ratio(rare_coding_summaries, common_coding_summaries)
    rare_common_ratio.to_csv(cluster_summaries_dir / 'rare_common_ratio_by_cluster.csv', index=False)

    # (2) SFARI-only rare:common
    sfari_rare_common_ratio = calculate_rare_common_ratio(rare_coding_summaries, common_coding_summaries, sfari_only=True)
    sfari_rare_common_ratio.to_csv(cluster_summaries_dir / 'sfari_rare_common_ratio_by_cluster.csv', index=False)

    comparison_time = time.time()
    print(f"Comparison tables completed in {comparison_time - summary_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\nAnalysis complete in {total_time:.2f} seconds. Outputs saved to:")
    print(f"- Individual results: {individual_results_dir}")
    print(f"- Cluster summaries: {cluster_summaries_dir}")

# if __name__ == "__main__":
#     #Run for just 3 specific participants
#     main(num_workers=8, reanalyze=True, specific_participants=["asd001", "asd002", "asd003"])

# For production run with all participants:
if __name__ == "__main__":
    main(num_workers=8, reanalyze=False)

#%%
## Test the code with a sample participant ## 
# import pickle
# from pathlib import Path

# # Set path to individual analysis files
# individual_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/individual_analyses')

# # Check a coding file
# with open(individual_dir / 'asd001_coding_analysis.pkl', 'rb') as f:
#     coding_data = pickle.load(f)

# # Verify structure
# print("Coding data keys:", coding_data.keys())
# if 'common_coding' in coding_data:
#     print("\nCommon coding keys:", coding_data['common_coding'].keys())
#     print("Common coding full_df shape:", coding_data['common_coding']['full_df'].shape)
#     print("Common coding summary_df shape:", coding_data['common_coding']['summary_df'].shape)
#     print("Number of high priority genes:", len(coding_data['common_coding']['high_priority_genes']))

# # Check a noncoding file
# with open(individual_dir / 'asd001_noncoding_analysis.pkl', 'rb') as f:
#     noncoding_data = pickle.load(f)

# # Verify structure
# print("\nNoncoding data keys:", noncoding_data.keys())
# if 'filtered_df' in noncoding_data:
#     print("Noncoding filtered_df shape:", noncoding_data['filtered_df'].shape)

# #%%
# # Set path to cluster summaries
# cluster_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/cluster_summaries')

# # Check common coding summaries
# with open(cluster_dir / 'common_coding_cluster_summaries.pkl', 'rb') as f:
#     common_summaries = pickle.load(f)

# # Verify structure
# clusters = common_summaries.keys()
# print("\nClusters found:", list(clusters))

# for cluster in clusters:
#     print(f"\nCluster {cluster}:")
#     print(f"  Participants: {len(common_summaries[cluster]['participants'])}")
#     print(f"  Variant type stats: {list(common_summaries[cluster]['variant_type_stats'].keys())}")
#     print(f"  SFARI score stats: {list(common_summaries[cluster]['sfari_score_stats'].keys())}")
    
#     # Check a specific variant type for detailed stats
#     if 'PTV' in common_summaries[cluster]['variant_type_stats']:
#         ptv_stats = common_summaries[cluster]['variant_type_stats']['PTV']
#         print(f"  PTV median variants: {ptv_stats.get('median_variants', 'N/A')}")
#         print(f"  PTV total unique genes: {ptv_stats.get('total_unique_genes', 'N/A')}")

# #%%
# import pandas as pd

# # Read the comparison CSV files
# cluster_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/cluster_summaries')
# common_coding_table = pd.read_csv(cluster_dir / 'common_coding_cluster_comparison.csv')
# rare_coding_table = pd.read_csv(cluster_dir / 'rare_coding_cluster_comparison.csv')
# noncoding_table = pd.read_csv(cluster_dir / 'noncoding_cluster_comparison.csv')
# rare_common_ratio = pd.read_csv(cluster_dir / 'rare_common_ratio_by_cluster.csv')
# sfari_rare_common_ratio = pd.read_csv(cluster_dir / 'sfari_rare_common_ratio_by_cluster.csv')

# # Display more rows from each table
# print("\nCommon coding comparison table - variant categories:")
# variant_rows = common_coding_table[common_coding_table['Statistic'].str.contains('PTV|Missense|Synonymous', na=False)]
# print(variant_rows)

# print("\nRare coding comparison table - SFARI categories:")
# sfari_rows = rare_coding_table[rare_coding_table['Statistic'].str.contains('SFARI', na=False)]
# print(sfari_rows)

# print("\nNoncoding comparison table - regulatory elements:")
# regulatory_rows = noncoding_table[noncoding_table['Statistic'].str.contains('Enhancer|Promoter', na=False)]
# print(regulatory_rows)

# # Show the rare:common ratios
# print("\nRare:Common ratios (all genes):")
# print(rare_common_ratio)

# print("\nRare:Common ratios (SFARI genes only):")
# print(sfari_rare_common_ratio)


#%%
###### Old Codes ######

# import pandas as pd
# from pathlib import Path
# import pickle

# def print_burden_table_from_df(full_df, sample_id="sample", sfari_only=False):
#     """
#     Print CADD + Gene Tolerance burden table from full_df.

#     Args:
#         full_df (pd.DataFrame): The full coding variant DataFrame
#         sample_id (str): Sample ID to print in header
#         sfari_only (bool): If True, only include SFARI genes
#     """
#     df = full_df.copy()

#     if sfari_only:
#         df = df[df['SFARI_gene_score'] != '.']

#     total_variants = len(full_df)
#     sfari_variants = df.shape[0]
#     percent_sfari = (sfari_variants / total_variants * 100) if total_variants > 0 else 0

#     print(f"===== Variant Burden Analysis for {sample_id} ({'SFARI only' if sfari_only else 'All variants'}) =====")
#     print(f"Total variants analyzed: {total_variants}")
#     print(f"{'Subset variants':<25}: {sfari_variants} ({percent_sfari:.2f}%)\n")

#     # Build Test column (if not already there)
#     def build_test_label(row):
#         if row['Variant_Category'] == 'Missense':
#             if row['cadd_bin'] == 'CADD>30':
#                 return 'Missense (CADD>30)'
#             elif row['cadd_bin'] == 'CADD20-30':
#                 return 'Missense (CADD 20-30)'
#             elif row['cadd_bin'] == 'CADD<20':
#                 return 'Missense (CADD 0-20)'
#         return row['Variant_Category']
    
#     if 'Test' not in df.columns:
#         df['Test'] = df.apply(build_test_label, axis=1)
    
#     # Gene Tolerance
#     df['PLI'] = pd.to_numeric(df['PLI'], errors='coerce')
#     df['Gene_Tolerance'] = 'pLI < 0.9'
#     df.loc[df['PLI'] >= 0.9, 'Gene_Tolerance'] = 'pLI >= 0.9'

#     # Also include "All genes"
#     tolerance_cats = ['All genes', 'pLI < 0.9', 'pLI >= 0.9']
#     test_cats = ['PTV', 'Missense (CADD>30)', 'Missense (CADD 20-30)',
#                  'Missense (CADD 0-20)', 'Synonymous', 'Other']

#     rows = []
#     for test in test_cats:
#         for tol in tolerance_cats:
#             if tol == 'All genes':
#                 sub = df[df['Test'] == test]
#             else:
#                 sub = df[(df['Test'] == test) & (df['Gene_Tolerance'] == tol)]

#             genes = [g for g in sub['GeneName'].unique() if g not in ['.', '--'] and pd.notna(g)]
#             rows.append({
#                 'Test': test,
#                 'Gene_Tolerance': tol,
#                 'No_of_genes': len(genes),
#                 'Variant_Count': len(sub)
#             })

#     result_df = pd.DataFrame(rows)

#     # Pretty print table
#     print(f"{'Test':<26} {'Gene_Tolerance':<15} {'No_of_genes':<14} {'Variant_Count':<15}")
#     print("-" * 75)
#     for _, row in result_df.iterrows():
#         print(f"{row['Test']:<26} {row['Gene_Tolerance']:<15} {row['No_of_genes']:<14} {row['Variant_Count']:<15}")

#     return result_df

# # Load the result file
# individual_results_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/individual_analyses')
# with open(individual_results_dir / 'asd001_coding_analysis.pkl', 'rb') as f:
#     data = pickle.load(f)

# # Choose common or rare
# full_df = data['common_coding']['full_df']

# # Print ALL variant burden table
# print_burden_table_from_df(full_df, sample_id='asd001', sfari_only=False)

# # Print SFARI-only variant burden table
# print_burden_table_from_df(full_df, sample_id='asd001', sfari_only=True)

# # %%
# def generate_cluster_burden_unfiltered(coding_results, participant_to_cluster, sfari_only=False):
#     """
#     Aggregate full_df burden stats across clusters.
    
#     Args:
#         coding_results (dict): {participant_id: coding_analysis_dict}
#         participant_to_cluster (dict): {participant_id: cluster_id}
#         sfari_only (bool): Whether to filter to SFARI variants
        
#     Returns:
#         dict: Cluster -> Summary DF (like what print_burden_table_from_df returns)
#     """
#     from collections import defaultdict
#     cluster_dfs = defaultdict(list)

#     for pid, res in coding_results.items():
#         if not res or 'common_coding' not in res:
#             continue

#         full_df = res['common_coding']['full_df']
#         cluster = participant_to_cluster.get(pid)
#         if cluster is None:
#             continue

#         df_filtered = full_df[full_df['SFARI_gene_score'] != '.'] if sfari_only else full_df

#         # build Test and Gene_Tolerance
#         df_filtered = df_filtered.copy()
#         df_filtered['PLI'] = pd.to_numeric(df_filtered['PLI'], errors='coerce')
#         df_filtered['Gene_Tolerance'] = 'pLI < 0.9'
#         df_filtered.loc[df_filtered['PLI'] >= 0.9, 'Gene_Tolerance'] = 'pLI >= 0.9'
        
#         def build_test_label(row):
#             if row['Variant_Category'] == 'Missense':
#                 if row['cadd_bin'] == 'CADD>30':
#                     return 'Missense (CADD>30)'
#                 elif row['cadd_bin'] == 'CADD20-30':
#                     return 'Missense (CADD 20-30)'
#                 elif row['cadd_bin'] == 'CADD<20':
#                     return 'Missense (CADD 0-20)'
#             return row['Variant_Category']

#         df_filtered['Test'] = df_filtered.apply(build_test_label, axis=1)

#         cluster_dfs[cluster].append(df_filtered)

#     # Combine all dfs per cluster and compute burden table
#     cluster_burden_tables = {}
#     for cluster, dfs in cluster_dfs.items():
#         merged = pd.concat(dfs, ignore_index=True)
#         burden_df = print_burden_table_from_df(merged, sample_id=f"Cluster {cluster}", sfari_only=sfari_only)
#         cluster_burden_tables[cluster] = burden_df

#     return cluster_burden_tables

# #%%
# with open('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/individual_analyses/asd001_coding_analysis.pkl', 'rb') as f:
#     coding_results = {'asd001': pickle.load(f)}  # Or load all if needed

# # Load cluster info
# cluster_df = pd.read_csv('/home/data/2025_asd_wgs/dataframes/participant_clinical_brain_cluster_df.csv')
# participant_to_cluster = dict(zip(cluster_df['id'], cluster_df['cluster']))

# # Generate unfiltered burden table (all genes)
# cluster_burden_unfiltered = generate_cluster_burden_unfiltered(coding_results, participant_to_cluster, sfari_only=False)

# # Generate filtered burden table (SFARI only)
# cluster_burden_sfari = generate_cluster_burden_unfiltered(coding_results, participant_to_cluster, sfari_only=True)


# # %%
# from pathlib import Path
# import pickle
# import pandas as pd

# # Load noncoding results for one participant (e.g., asd001)
# individual_results_dir = Path('/home/data/2025_asd_wgs/genetype_results/cluster_analysis/individual_analyses')
# participant_id = 'asd001'
# noncoding_file = individual_results_dir / f'{participant_id}_noncoding_analysis.pkl'

# with open(noncoding_file, 'rb') as f:
#     noncoding_data = pickle.load(f)

# # Extract the filtered dataframe
# df = noncoding_data['filtered_df'].copy()

# # Clean Eigen bin
# df['Eigen_bin'] = df['eigen_bin'].replace({'Eigen>0': 'Eigen >= 0', 'Eigen<=0': 'Eigen < 0'})

# # Define regulatory elements from Regulatory_Region column
# def map_regulatory_region(raw):
#     raw = str(raw).lower()
#     if 'enhancer' in raw and 'promoter' in raw:
#         return 'Enhancer,Promoter'
#     elif 'enhancer' in raw:
#         return 'Enhancer'
#     elif 'promoter' in raw:
#         return 'Promoter'
#     else:
#         return 'Other'

# df['Regulatory_Clean'] = df['Regulatory_Region'].apply(map_regulatory_region)

# # Group by Regulatory Element and Eigen bin
# grouped = df.groupby(['Regulatory_Clean', 'Eigen_bin'])

# # Summarize variant count and unique gene count
# summary_rows = []
# for (reg_elem, eigen_bin), group in grouped:
#     genes = group['GeneName'].dropna().unique()
#     summary_rows.append({
#         'Regulatory_Element': reg_elem,
#         'Gene_Tolerance': eigen_bin,
#         'No_of_genes': len(genes),
#         'Variant_Count': len(group)
#     })

# # Create and print the summary DataFrame
# summary_df = pd.DataFrame(summary_rows)
# print(summary_df.sort_values(by=['Regulatory_Element', 'Gene_Tolerance']))

# # %%
# # Filter to only SFARI genes (SFARI_gene_score not equal to '.')
# sfari_df = df[df['SFARI_gene_score'] != '.'].copy()

# # Group by Regulatory Element and Eigen bin (same as before)
# grouped = sfari_df.groupby(['Regulatory_Clean', 'Eigen_bin'])

# # Summarize variant count and unique gene count
# summary_rows = []
# for (reg_elem, eigen_bin), group in grouped:
#     genes = group['GeneName'].dropna().unique()
#     summary_rows.append({
#         'Regulatory_Element': reg_elem,
#         'Gene_Tolerance': eigen_bin,
#         'No_of_genes': len(genes),
#         'Variant_Count': len(group)
#     })

# # Create and print the SFARI-only summary DataFrame
# sfari_summary_df = pd.DataFrame(summary_rows)
# print(sfari_summary_df.sort_values(by=['Regulatory_Element', 'Gene_Tolerance']))

# %%
