#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Check multiple possible paths
possible_paths = [
    "/home/data/2025_asd_wgs/fgsea_results/fgsea_all_clusters_bidirectional_results.csv",
    "/home/data/2025_asd_wgs/reference/fgsea_all_clusters_bidirectional_results.csv",
    "./fgsea_all_clusters_bidirectional_results.csv"  # Current directory
]

result_file_path = None
for path in possible_paths:
    if os.path.exists(path):
        result_file_path = path
        print(f"Found results file at: {path}")
        break

if result_file_path is None:
    # Try searching for the file
    for root, dirs, files in os.walk("/home/data/2025_asd_wgs"):
        if "fgsea_all_clusters_bidirectional_results.csv" in files:
            result_file_path = os.path.join(root, "fgsea_all_clusters_bidirectional_results.csv")
            print(f"Found results file at: {result_file_path}")
            break

if result_file_path is None:
    raise FileNotFoundError("Could not find fgsea_all_clusters_bidirectional_results.csv file")

# Path to save the results
save_figures_path = "/home/cko4/2025_asd_multimodal/figures"
save_tables_path = "/home/cko4/2025_asd_multimodal/tables"

# Load the data
df = pd.read_csv(result_file_path)

# 1. Basic information
print(f"Total pathways in analysis: {len(df)}")

# 2. Count pathways by cluster
cluster_counts = df['cluster'].value_counts().sort_index()
print("\nPathways per cluster:")
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} pathways")

# 3. Find significant pathways (FDR < 0.25)
sig_pathways = df[df['padj'] < 0.25]
print(f"\nSignificant pathways (FDR < 0.25): {len(sig_pathways)}")

# 4. Group significant pathways by cluster
if len(sig_pathways) > 0:
    print("\nSignificant pathways by cluster:")
    for cluster in sorted(sig_pathways['cluster'].unique()):
        cluster_paths = sig_pathways[sig_pathways['cluster'] == cluster]
        print(f"\nCluster {cluster}: {len(cluster_paths)} significant pathways")
        
        # Show top 5 significant pathways for each cluster
        top5 = cluster_paths.sort_values('padj').head(5)
        print("\nTop 5 pathways by FDR:")
        for i, row in top5.iterrows():
            # Clean pathway name for display
            pathway_clean = row['pathway']
            for prefix in ['HALLMARK_', 'REACTOME_', 'KEGG_', 'GO_']:
                if pathway_clean.startswith(prefix):
                    pathway_clean = pathway_clean[len(prefix):]
            pathway_clean = pathway_clean.replace('_', ' ')
            
            print(f"- {pathway_clean} (FDR = {row['padj']:.4f}, NES = {row['NES']:.2f})")
else:
    print("No significant pathways found with FDR < 0.25")

# 5. Pathway direction analysis
if len(df) > 0:
    print("\nPathway direction analysis:")
    positive_nes = df[df['NES'] > 0]
    negative_nes = df[df['NES'] < 0]
    print(f"Pathways with positive NES: {len(positive_nes)} ({len(positive_nes)/len(df)*100:.1f}%)")
    print(f"Pathways with negative NES: {len(negative_nes)} ({len(negative_nes)/len(df)*100:.1f}%)")

# Function to categorize pathways into functional groups
def categorize_pathway(pathway_name):
    # Remove prefixes like C2, C5, etc.
    clean_name = re.sub(r'^(C\d+\.)?', '', pathway_name)
    
    # Special cases for specific pathways - manual categorization
    if 'AV NODE CELL TO BUNDLE OF HIS' in clean_name.upper():
        return 'Cardiac Development'
    if 'CALCIUM ION TRANSMEMBRANE' in clean_name.upper():
        return 'Neuro Development'
    
    # Define functional categories and associated keywords, with neuro at the top
    categories = {
        'Neuro Development': ['NEUROMUSCULAR', 'NEURO', 'BRAIN', 'AXON', 
                             'SYNAP', 'NERVE', 'NEURAL', 'NEURON', 'DENDRITE', 'GLIAL'],
        'Cardiac Development': ['CARDIAC', 'HEART', 'CHAMBER', 'BUNDLE OF HIS'],
        'Muscle Function': ['MUSCLE', 'CONTRACTION', 'ACTION POTENTIAL'],
        'Cell Communication': ['CELL JUNCTION', 'CELL COMMUNICATION', 'CELL PROJECTION'],
        'Ion Transport': ['ION', 'ELECTRON TRANSPORT'],
        'Structural Components': ['COLLAGEN', 'GRANULE', 'CHAIN', 'TRIMER'],
        'Tissue Morphogenesis': ['MORPHOGENESIS', 'ORGAN', 'ANIMAL'],
        'Cell Regulation': ['REGULATION', 'KINASE', 'WOUND HEALING'],
        'Depolarization': ['DEPOLARISATION', 'PHASE 0'],
        'Other': []  # Default category
    }
    
    # Exclude certain pathways completely
    exclude_terms = ['DRUG ADME', 'HIGH CONFIDENCE', 'ASD RDNV']
    if any(term in clean_name.upper() for term in exclude_terms):
        return 'EXCLUDE'
    
    # Check which category the pathway belongs to
    for category, keywords in categories.items():
        if any(keyword in clean_name.upper() for keyword in keywords):
            return category
    
    return 'Other'  # Default if no match

# 6. Create improved visualization of pathways grouped by function
try:
    # Clean pathway names first to be able to filter using the clean names
    df['pathway_clean'] = df['pathway'].apply(lambda x: re.sub(r'^(C\d+\.)?', '', x))
    
    # Remove GO gene ontology prefixes 
    df['pathway_clean'] = df['pathway_clean'].str.replace('GOBP', '', regex=False)
    df['pathway_clean'] = df['pathway_clean'].str.replace('GOMF', '', regex=False)
    df['pathway_clean'] = df['pathway_clean'].str.replace('GOCC', '', regex=False)
    
    # Remove remaining database prefixes
    for prefix in ['HALLMARK_', 'REACTOME_', 'KEGG_', 'GO_']:
        df['pathway_clean'] = df['pathway_clean'].str.replace(prefix, '', regex=False)
    
    # Replace underscores with spaces for readability
    df['pathway_clean'] = df['pathway_clean'].str.replace('_', ' ', regex=False)
    
    # Now filter to remove "High Confidence", "DRUG ADME", and "ASD RDNV"
    df_clean = df[~df['pathway_clean'].str.contains('High Confidence', case=False, na=False)]
    df_clean = df_clean[~df_clean['pathway_clean'].str.contains('DRUG ADME', case=False, na=False)]
    df_clean = df_clean[~df_clean['pathway_clean'].str.contains('ASD RDNV', case=False, na=False)]
    
    # Assign functional categories
    df_clean['category'] = df_clean['pathway_clean'].apply(categorize_pathway)
    
    # Remove any pathways marked for exclusion
    df_clean = df_clean[df_clean['category'] != 'EXCLUDE']
    
    # Get top pathways per cluster, ensuring representation from different categories
    top_paths = []
    for cluster in sorted(df_clean['cluster'].unique()):
        cluster_df = df_clean[df_clean['cluster'] == cluster]
        
        # For each category, get the top pathway by significance (lowest padj)
        for category in df_clean['category'].unique():
            category_df = cluster_df[cluster_df['category'] == category]
            if not category_df.empty:
                top = category_df.sort_values('padj').head(1)
                top_paths.append(top)
    
    top_paths_df = pd.concat(top_paths)
    
    # Define category order with Neuro at the top
    category_order = ['Neuro Development', 'Cardiac Development', 'Muscle Function', 
                      'Ion Transport', 'Cell Communication', 'Structural Components',
                      'Tissue Morphogenesis', 'Cell Regulation', 'Depolarization', 'Other']
    
    # Create a category sort key that respects our custom order
    category_sort_dict = {cat: i for i, cat in enumerate(category_order)}
    top_paths_df['category_sort'] = top_paths_df['category'].map(lambda x: category_sort_dict.get(x, 999))
    
    # Sort by our custom category order, then by pathway name
    top_paths_df = top_paths_df.sort_values(['category_sort', 'pathway_clean'])
    
    # Create visualization
    plt.figure(figsize=(14, 12))
    
    # Prepare data for plotting
    plot_data = top_paths_df.pivot(index='pathway_clean', columns='cluster', values='NES')
    padj_data = top_paths_df.pivot(index='pathway_clean', columns='cluster', values='padj')
    
    # Use our custom category order to organize the pathways
    ordered_categories = [cat for cat in category_order if cat in top_paths_df['category'].unique()]
    pathways_by_category = []
    
    for category in ordered_categories:
        cat_pathways = top_paths_df[top_paths_df['category'] == category]['pathway_clean'].unique()
        pathways_by_category.extend(cat_pathways)
    
    # Reindex to maintain category grouping with Neuro at the top
    plot_data = plot_data.reindex(pathways_by_category)
    padj_data = padj_data.reindex(pathways_by_category)
    
    # Add significance markers to pathway names
    pathway_labels = plot_data.index.tolist()
    new_labels = []
    
    for pathway in pathway_labels:
        # Check if any padj value for this pathway is < 0.1 across clusters
        pathway_padj = padj_data.loc[pathway]
        is_significant = (pathway_padj < 0.1).any()
        
        if is_significant:
            new_labels.append(f"{pathway} †")
        else:
            new_labels.append(pathway)
    
    # Update the index with new labels
    plot_data.index = new_labels
    
    # Plot with custom color map
    sns.heatmap(plot_data, cmap='RdBu_r', center=0, 
                linewidths=0.5, annot=True, fmt='.2f',
                cbar_kws={'label': 'Normalized Enrichment Score'})
    
    plt.title('Gene Set Enrichment Analysis: Pathway Enrichment by Functional Categories', fontsize=14)
    plt.ylabel('Pathway', fontsize=12)
    plt.xlabel('Cluster', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(fontsize=10)
    
    # Add legend for significance markers
    plt.figtext(0.01, 0.01, "† p < 0.1", ha='left', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure as PDF and PNG
    plt.savefig(os.path.join(save_figures_path, 'functional_pathways_heatmap.pdf'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_figures_path, 'functional_pathways_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    print("\nSaved functional heatmap visualization as PDF and PNG")
    
    # Create a separate table with the pathway categories and p-values
    pathway_info = top_paths_df[['pathway_clean', 'category', 'padj', 'NES', 'cluster']].copy()
    pathway_info.to_csv(os.path.join(save_tables_path, 'pathway_categories_with_stats.csv'), index=False)
    print("Saved pathway information to 'pathway_categories_with_stats.csv'")
    
except Exception as e:
    print(f"Couldn't create visualization: {e}")


#%%
# Check SFARI gene enrichment # 
# Second part: SFARI gene enrichment analysis
print("\n\n=== SFARI Gene Enrichment Analysis ===\n")

# Find SFARI enrichment file
sfari_paths = [
    "/home/data/2025_asd_wgs/fgsea_results/sfari_enrichment_by_cluster.csv"
]

sfari_file = None
for path in sfari_paths:
    if os.path.exists(path):
        sfari_file = path
        print(f"Found SFARI enrichment file at: {path}")
        break

if sfari_file is None:
    # Try searching for the file
    for root, dirs, files in os.walk("/home/data/2025_asd_wgs"):
        if "sfari_enrichment_by_cluster.csv" in files:
            sfari_file = os.path.join(root, "sfari_enrichment_by_cluster.csv")
            print(f"Found SFARI file at: {sfari_file}")
            break

if sfari_file is None:
    print("Could not find SFARI enrichment file")
else:
    # Load SFARI enrichment data
    sfari_df = pd.read_csv(sfari_file)
    
    # Display basic information
    print("\nSFARI gene enrichment by cluster:")
    print(sfari_df)
    
    # Add significance markers
    sfari_df['Significance'] = 'ns'
    sfari_df.loc[sfari_df['P_Value'] < 0.05, 'Significance'] = '*'
    sfari_df.loc[sfari_df['P_Value'] < 0.01, 'Significance'] = '**'
    sfari_df.loc[sfari_df['P_Value'] < 0.001, 'Significance'] = '***'
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Cluster', y='SFARI_Percent', data=sfari_df, palette='viridis')
    
    # Add percentage labels
    for i, row in sfari_df.iterrows():
        ax.text(i, row['SFARI_Percent'] + 0.5, f"{row['SFARI_Percent']:.1f}%", 
                ha='center', va='bottom', fontweight='bold')
        
        # Add significance stars
        if row['Significance'] != 'ns':
            ax.text(i, row['SFARI_Percent'] + 2, row['Significance'], 
                   ha='center', va='bottom', fontsize=16)
    
    plt.title('SFARI Gene Enrichment by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('SFARI Genes (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_figures_path, 'sfari_enrichment_barplot.png'))
    print("Saved SFARI enrichment visualization to 'sfari_enrichment_barplot.png'")

    # Third part: ASD gene set overlap analysis
    asd_overlap_file = None
    for root, dirs, files in os.walk("/home/data/2025_asd_wgs"):
        if "asd_gene_set_overlaps.csv" in files:
            asd_overlap_file = os.path.join(root, "asd_gene_set_overlaps.csv")
            print(f"\nFound ASD gene set overlap file at: {asd_overlap_file}")
            break
            
    if asd_overlap_file is None:
        print("\nCould not find ASD gene set overlap file")
    else:
        # Load ASD gene set overlap data
        asd_df = pd.read_csv(asd_overlap_file)
        
        # Get the top significant overlaps
        sig_overlaps = asd_df[asd_df['P_Value'] < 0.05].sort_values('P_Value')
        
        if len(sig_overlaps) > 0:
            print("\nSignificant ASD gene set overlaps (p < 0.05):")
            
            # Clean up gene set names for display
            sig_overlaps['ASD_Set_Clean'] = sig_overlaps['ASD_Gene_Set'].str.replace('_', ' ')
            
            # Display results
            for i, row in sig_overlaps.head(10).iterrows():
                print(f"Cluster {row['Cluster']} - {row['ASD_Set_Clean']}: {row['Overlap_Percent']:.1f}% overlap (p={row['P_Value']:.4f})")
            
            # Create heatmap of overlaps
            plt.figure(figsize=(12, 8))
            pivot_df = asd_df.pivot(index='ASD_Gene_Set', columns='Cluster', values='Overlap_Percent')
            sns.heatmap(pivot_df, cmap='YlOrRd', annot=True, fmt='.1f')
            plt.title('ASD Gene Set Overlap Percentages by Cluster')
            plt.tight_layout()
            plt.savefig('asd_gene_overlap_heatmap.png')
            print("Saved ASD gene set overlap heatmap to 'asd_gene_overlap_heatmap.pdf'")
        else:
            print("\nNo significant ASD gene set overlaps found")


#%%
# Check if the annotated gene tables show any SFARI genes
sfari_annotated_file = "/home/data/2025_asd_wgs/fgsea_results/SKAT_all_genes_SFARI_annotated.csv"

if os.path.exists(sfari_annotated_file):
    sfari_genes_df = pd.read_csv(sfari_annotated_file)
    # Check if any genes are marked as SFARI
    sfari_count = sfari_genes_df['Is_SFARI'].sum() if 'Is_SFARI' in sfari_genes_df.columns else 0
    print(f"\nFound {sfari_count} SFARI genes in the annotated gene table")
    
    if sfari_count > 0:
        print("Sample of SFARI genes found:")
        print(sfari_genes_df[sfari_genes_df['Is_SFARI'] == True].head())
    else:
        print("No SFARI genes found in the annotated table.")
else:
    print("Could not find SFARI annotated gene table.")

#%%
# Try to locate the SFARI database file
sfari_db_file = "/home/data/2025_asd_wgs/reference/SFARI-Gene_genes_04-03-2025release_04-21-2025export.csv"

if os.path.exists(sfari_db_file):
    sfari_db = pd.read_csv(sfari_db_file)
    print(f"\nSFARI database file contains {len(sfari_db)} genes")
    print("Column names:", sfari_db.columns.tolist())
    
    # Try different possible column names for gene symbols
    for col in ['gene', 'gene.symbol', 'gene-symbol', 'gene_symbol', 'symbol']:
        if col in sfari_db.columns:
            print(f"Sample of gene symbols from column '{col}':")
            print(sfari_db[col].head())
            break
else:
    print("Could not find SFARI database file.")

#%%
# Check cluster gene lists
cluster_file = "/home/data/2025_asd_wgs/skat_results/SKAT_all_cluster_specific_genes.csv"
if os.path.exists(cluster_file):
    cluster_df = pd.read_csv(cluster_file)
    print(f"\nCluster specific genes file contains {len(cluster_df)} genes")
    print("Sample genes from each cluster:")
    for cluster in sorted(cluster_df['Cluster'].unique()):
        sample_genes = cluster_df[cluster_df['Cluster'] == cluster]['Gene'].head(5).tolist()
        print(f"Cluster {cluster}: {', '.join(sample_genes)}")
else:
    print("Could not find cluster genes file.")

# %%
