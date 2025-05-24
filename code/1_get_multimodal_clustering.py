'''
Date created: 04-08-2025
Date modified: 04-10-2025
Author: Chanyoung Ko

Objective: Multimodal using selected (canonical) features extracted from fMRI & video (gaze) data
Number of gaze features per subject: 45 gaze features for 45 subjects 
Number of fMRI features per subject: 22 fMRI features for 42 subjects

Note: for the 42 subjects that have both gaze & fMRI data, perform clustering analysis

===== Loading Canonical Variates =====
Loaded gaze canonical variates: (45, 3)
Loaded fMRI canonical variates: (42, 3)

===== Merging Canonical Variates =====
Gaze subject IDs: ['asd001', 'asd002', 'asd003', 'asd005', 'asd006', 'asd007', 'asd008', 'asd009', 'asd011', 'asd012', 'asd013', 'asd014', 'asd015', 'asd016', 'asd017', 'asd018', 'asd019', 'asd021', 'asd022', 'asd023', 'asd024', 'asd025', 'asd026', 'asd027', 'asd028', 'asd029', 'asd030', 'asd031', 'asd032', 'asd033', 'asd034', 'asd035', 'asd036', 'asd037', 'asd038', 'asd039', 'asd040', 'asd041', 'asd042', 'asd043', 'asd044', 'asd045', 'asd046', 'asd047', 'asd048']
fMRI subject IDs: ['asd001', 'asd002', 'asd003', 'asd005', 'asd006', 'asd007', 'asd008', 'asd009', 'asd011', 'asd012', 'asd013', 'asd014', 'asd015', 'asd016', 'asd017', 'asd018', 'asd019', 'asd021', 'asd022', 'asd023', 'asd024', 'asd025', 'asd026', 'asd027', 'asd028', 'asd029', 'asd030', 'asd031', 'asd032', 'asd034', 'asd035', 'asd036', 'asd037', 'asd038', 'asd039', 'asd040', 'asd041', 'asd044', 'asd045', 'asd046', 'asd047', 'asd048']
Found 42 subjects common to both modalities
Gaze-only subjects: 3
fMRI-only subjects: 0
Combined canonical variates shape: (42, 6)
'''

#%%
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Import utility functions
from src.utils_io import load_canonical_variates, merge_canonical_variates, save_cluster_results

from src.utils_clustering import (
    hierarchical_clustering, 
    robust_clustering,
    analyze_cluster_profiles
)
from src.utils_clustering_visual import (
    plot_dendrogram,
    plot_cluster_evaluation_metrics,
    plot_cluster_distribution,
    create_dendrogram_heatmap,
    create_bimodal_dendrogram_heatmap,
    plot_stability_metrics,
    visualize_clusters,
    save_visualization
)

# =============== Configuration ===============
# Set fixed seeds for reproducibility
SEED = 14
rng = np.random.default_rng(SEED)

# Base paths
ROOT_PATH_FMRI = Path('/home/data/2025_asd_fmri')
ROOT_PATH_GAZE = Path('/home/data/2025_asd_video')
ROOT_PATH_MULTI = Path('/home/data/2025_asd_multimodal')
PROJECT_PATH = Path(os.getcwd())

# Analysis parameters
dataset_name = "snu"  # snu, abide
atlas_name = "power"  # power, aal
timeseries_type = "timeseries_nogsr"  # nogsr, gsr
connectome_type = "full_correlation"  # full_correlation, partial_correlation

# Paths for gaze data
GAZE_CANONICAL_VARIATES_PATH = Path(
    ROOT_PATH_GAZE,
    dataset_name,
    "results",
    "gaze_canonical_variates"
)

# Paths for fMRI data
FMRI_CANONICAL_VARIATES_PATH = Path(
    ROOT_PATH_FMRI,
    dataset_name,
    "results",
    f"{atlas_name}_{timeseries_type}_{connectome_type}",
    "fmri_canonical_variates"
)

# Path for clustering results
CLUSTERING_RESULTS_PATH = Path(ROOT_PATH_MULTI, "clustering_results")
if not os.path.exists(CLUSTERING_RESULTS_PATH):
    os.makedirs(CLUSTERING_RESULTS_PATH)

# Print the paths for verification
print(f"GAZE_CANONICAL_VARIATES_PATH: {GAZE_CANONICAL_VARIATES_PATH}")
print(f"FMRI_CANONICAL_VARIATES_PATH: {FMRI_CANONICAL_VARIATES_PATH}")
print(f"CLUSTERING_RESULTS_PATH: {CLUSTERING_RESULTS_PATH}")

def main():
    # Load canonical variates
    gaze_cvs, fmri_cvs = load_canonical_variates(
        gaze_path=GAZE_CANONICAL_VARIATES_PATH,
        fmri_path=FMRI_CANONICAL_VARIATES_PATH
    )
    
    # Merge canonical variates
    combined_cvs = merge_canonical_variates(
        gaze_cvs=gaze_cvs, 
        fmri_cvs=fmri_cvs,
        output_path=CLUSTERING_RESULTS_PATH
    )
    
    # Perform hierarchical clustering and determine optimal number of clusters
    cluster_eval = hierarchical_clustering(
        combined_cvs=combined_cvs,
        n_clusters_range=(2, 10),
    )
    
    # Visualize and save dendrograms
    dendro_fig = plot_dendrogram(
        linkage_matrix=cluster_eval['linkage'],
        truncate_mode='level',
        p=5,  # This was missing before
        title='Hierarchical Clustering Dendrogram'
    )
    save_visualization(dendro_fig, CLUSTERING_RESULTS_PATH, 'dendrogram.png')

    full_dendro_fig = plot_dendrogram(
        linkage_matrix=cluster_eval['linkage'],
        title='Full Hierarchical Clustering Dendrogram'
    )
    save_visualization(full_dendro_fig, CLUSTERING_RESULTS_PATH, 'dendrogram_full.png')
        
    # Save linkage matrix
    np.save(Path(CLUSTERING_RESULTS_PATH, 'linkage_matrix.npy'), cluster_eval['linkage'])
    
    # Save cluster evaluation metrics
    eval_metrics_df = pd.DataFrame({
        'n_clusters': cluster_eval['n_clusters_range'],
        'silhouette': cluster_eval['silhouette_scores'],
        'calinski_harabasz': cluster_eval['ch_scores'],
        'davies_bouldin': cluster_eval['db_scores']
    })
    eval_metrics_df.to_csv(Path(CLUSTERING_RESULTS_PATH, 'cluster_evaluation_metrics.csv'), index=False)
    
    # Plot and save evaluation metrics
    metrics_fig = plot_cluster_evaluation_metrics(
        n_clusters_range=cluster_eval['n_clusters_range'],
        silhouette_scores=cluster_eval['silhouette_scores'],
        ch_scores=cluster_eval['ch_scores'],
        db_scores=cluster_eval['db_scores']
    )
    save_visualization(metrics_fig, CLUSTERING_RESULTS_PATH, 'cluster_evaluation.png')
    
    # Get consensus on optimal cluster number
    # Take the most frequent optimal number across metrics
    cluster_votes = [
        cluster_eval['best_silhouette'], 
        cluster_eval['best_ch'], 
        cluster_eval['best_db']
    ]
    optimal_clusters = max(set(cluster_votes), key=cluster_votes.count)
    print(f"\nConsensus optimal number of clusters: {optimal_clusters}")
    print(f"Using consensus optimal number of clusters: {optimal_clusters}")
    
    # Perform robust clustering with resampling
    cluster_labels, combined_cvs_with_clusters, bootstrap_df, stability_metrics = robust_clustering(
        combined_cvs=combined_cvs, 
        n_clusters=optimal_clusters,
        n_resamples=1000,
        random_state=rng
    )
    
    # Save the robust clustering results
    combined_results = save_cluster_results(
        cluster_labels=cluster_labels,  # Already 1-based from new implementation
        combined_cvs=combined_cvs,
        output_path=CLUSTERING_RESULTS_PATH,
        prefix=f'combined_cvs_with_clusters_{optimal_clusters}'
    )
    
    # Save bootstrap stability information
    bootstrap_df.to_csv(Path(CLUSTERING_RESULTS_PATH, f'bootstrap_stability_{optimal_clusters}.csv'))
    
    # Save stability metrics
    pd.DataFrame({
        'metric': ['agreement', 'adjusted_rand_index'],
        'value': [stability_metrics['agreement'], stability_metrics['ari']]
    }).to_csv(Path(CLUSTERING_RESULTS_PATH, 'stability_metrics.csv'), index=False)
    
    # Save the final robust linkage matrix
    np.save(Path(CLUSTERING_RESULTS_PATH, f'robust_linkage_matrix_{optimal_clusters}.npy'), 
            stability_metrics['linkage_matrix'])
    
    # Plot and save cluster distribution
    print("\n===== Clustering Complete =====")
    dist_fig, cluster_summary = plot_cluster_distribution(cluster_labels)
    save_visualization(dist_fig, CLUSTERING_RESULTS_PATH, 'cluster_distribution.png')
    
    # Save cluster distribution summary
    cluster_summary.to_csv(Path(CLUSTERING_RESULTS_PATH, 'cluster_distribution.csv'), index=False)
    
    # Print summary information
    for cluster, row in cluster_summary.iterrows():
        print(f"Cluster {row['cluster']}: {row['count']} subjects ({row['percentage']:.1f}%)")
    
    # Visualize clusters using PCA
    cluster_viz_fig, embedding = visualize_clusters(
        combined_cvs=combined_cvs,
        cluster_labels=cluster_labels,
        method='pca'
    )
    save_visualization(cluster_viz_fig, CLUSTERING_RESULTS_PATH, 'cluster_visualization_pca.png', dpi=300)
    
    # Save the embedding for future use
    pd.DataFrame(
        embedding, 
        index=combined_cvs.index,
        columns=['component_1', 'component_2']
    ).to_csv(Path(CLUSTERING_RESULTS_PATH, 'cluster_pca_embedding.csv'))
    
    # Analyze cluster profiles
    profile_fig, cluster_profiles = analyze_cluster_profiles(
        combined_cvs=combined_cvs,
        cluster_labels=cluster_labels
    )
    save_visualization(profile_fig, CLUSTERING_RESULTS_PATH, 'cluster_profiles_heatmap.png', dpi=300)
    
    # Save cluster profiles
    cluster_profiles.to_csv(Path(CLUSTERING_RESULTS_PATH, 'cluster_profiles.csv'))
    
    # Plot stability metrics from bootstrap
    stability_fig = plot_stability_metrics(bootstrap_df)
    save_visualization(stability_fig, CLUSTERING_RESULTS_PATH, 'bootstrap_stability_plot.png', dpi=300)
    
    # Identify gaze and fMRI columns
    gaze_columns = [col for col in combined_cvs.columns if 'gaze' in str(col).lower()]
    fmri_columns = [col for col in combined_cvs.columns if 'fmri' in str(col).lower()]
    
    if not gaze_columns or not fmri_columns:
        # If column names don't contain 'gaze' or 'fmri', use positional approach
        
        # First, identify and exclude non-feature columns
        exclude_cols = []
        # Check for any ID columns
        id_cols = [col for col in combined_cvs.columns 
                   if any(term in str(col).lower() for term in ['id', 'subject'])]
        exclude_cols.extend(id_cols)
        
        # Check for cluster columns
        cluster_cols = [col for col in combined_cvs.columns 
                       if 'cluster' in str(col).lower()]
        exclude_cols.extend(cluster_cols)
        
        # Get remaining feature columns
        feature_cols = [col for col in combined_cvs.columns if col not in exclude_cols]
        
        # Split into gaze and fMRI features
        half_cols = len(feature_cols) // 2
        gaze_columns = feature_cols[:half_cols]
        fmri_columns = feature_cols[half_cols:]
        
        print(f"Using positional approach for features:")
        print(f"  Gaze columns: {gaze_columns}")
        print(f"  fMRI columns: {fmri_columns}")
    
    # Save feature column information
    feature_columns = {
        'modality': ['gaze'] * len(gaze_columns) + ['fmri'] * len(fmri_columns),
        'column': gaze_columns + fmri_columns
    }
    pd.DataFrame(feature_columns).to_csv(Path(CLUSTERING_RESULTS_PATH, 'feature_columns.csv'), index=False)
    
    # Create dendrogram heatmap visualization
    try:
        heatmap_fig = create_dendrogram_heatmap(
            linkage_matrix=cluster_eval['linkage'],
            data_matrix=combined_cvs,
            optimal_clusters=optimal_clusters,
            output_path=CLUSTERING_RESULTS_PATH,
            figsize=(16, 12),
            cmap='coolwarm',
            label_fontsize=10,
            title_fontsize=14,
            dpi=300
        )
    except Exception as e:
        print(f"Error creating standard dendrogram heatmap: {e}")
    
    # Create bimodal dendrogram heatmap visualization
    try:
        bimodal_fig = create_bimodal_dendrogram_heatmap(
            linkage_matrix=cluster_eval['linkage'],
            combined_cvs=combined_cvs,
            gaze_columns=gaze_columns,
            fmri_columns=fmri_columns,
            optimal_clusters=optimal_clusters,
            output_path=CLUSTERING_RESULTS_PATH,
            figsize=(18, 10),
            dpi=300
        )
    except Exception as e:
        print(f"Error in bimodal visualization: {e}")
        print("Trying alternative approach...")
        
        # Extract only the feature columns for visualization
        features_only = combined_cvs[gaze_columns + fmri_columns]
        
        try:
            alt_fig = create_dendrogram_heatmap(
                linkage_matrix=cluster_eval['linkage'],
                data_matrix=features_only,
                optimal_clusters=optimal_clusters,
                output_path=CLUSTERING_RESULTS_PATH,
                figsize=(16, 12),
                cmap='coolwarm',
                label_fontsize=10,
                title_fontsize=14,
                dpi=300
            )
            print("Successfully created alternative visualization")
        except Exception as e2:
            print(f"Error in alternative visualization: {e2}")
            print("Please check the data structure and column names")
    
    print(f"\nResults saved to: {CLUSTERING_RESULTS_PATH}")
    
    # Return the key results for further analysis outside this function
    return {
        'combined_cvs': combined_cvs,                # Original data
        'cluster_labels': cluster_labels,            # 1-based cluster labels
        'cluster_eval': cluster_eval,                # Evaluation metrics and linkage matrix
        'combined_results': combined_results,        # Data with cluster labels
        'cluster_profiles': cluster_profiles,        # Mean values by cluster
        'bootstrap_stability': bootstrap_df,         # Bootstrap stability information
        'feature_columns': {                         # Feature column grouping
            'gaze': gaze_columns,
            'fmri': fmri_columns
        }
    }


if __name__ == "__main__":
    results = main()

#%%