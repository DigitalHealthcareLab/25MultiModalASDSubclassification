'''
Clustering utility functions for multimodal analysis
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster


def hierarchical_clustering(combined_cvs, n_clusters_range=(2, 10)):
    """
    Perform hierarchical clustering and determine optimal number of clusters
    """
    print("\n===== Performing Hierarchical Clustering =====")

    # Calculate cosine similarity matrix
    cosine_dist = pdist(combined_cvs, metric='cosine')
    similarity_matrix = 1 - squareform(cosine_dist)
    distance_matrix = 1 - similarity_matrix

    # Perform hierarchical clustering using average linkage
    Z = linkage(squareform(distance_matrix), method='average')

    # Evaluate different numbers of clusters
    ch_scores = []
    silhouette_scores = []
    db_scores = []

    for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
        labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1

        if n_clusters > 1:
            silhouette = silhouette_score(combined_cvs, labels)
            silhouette_scores.append(silhouette)
        else:
            silhouette_scores.append(0)

        ch = calinski_harabasz_score(combined_cvs, labels)
        ch_scores.append(ch)

        db = davies_bouldin_score(combined_cvs, labels)
        db_scores.append(db)

        print(f"Clusters: {n_clusters}, Silhouette: {silhouette_scores[-1]:.3f}, "
              f"CH: {ch_scores[-1]:.3f}, DB: {db_scores[-1]:.3f}")

    best_silhouette = np.argmax(silhouette_scores) + n_clusters_range[0]
    best_ch = np.argmax(ch_scores) + n_clusters_range[0]
    best_db = np.argmin(db_scores) + n_clusters_range[0]

    print(f"\nOptimal number of clusters:")
    print(f"Based on Silhouette Score: {best_silhouette}")
    print(f"Based on Calinski-Harabasz Index: {best_ch}")
    print(f"Based on Davies-Bouldin Index: {best_db}")

    return {
        'n_clusters_range': list(range(n_clusters_range[0], n_clusters_range[1] + 1)),
        'silhouette_scores': silhouette_scores,
        'ch_scores': ch_scores,
        'db_scores': db_scores,
        'best_silhouette': best_silhouette,
        'best_ch': best_ch,
        'best_db': best_db,
        'linkage': Z,
        'distance_matrix': distance_matrix
    }


def robust_clustering(combined_cvs, n_clusters, n_resamples=1000, random_state=None):
    """
    Implement robust clustering with resampling
    """
    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = random_state  # USE the passed-in RNG

    print(f"\n===== Performing Robust Clustering with {n_resamples} Resamples =====")

    n_subjects = len(combined_cvs)
    print(f"Number of subjects: {n_subjects}")

    cosine_dist = pdist(combined_cvs, metric='cosine')
    similarity_matrix = 1 - squareform(cosine_dist)
    distance_matrix = 1 - similarity_matrix

    Z_full = linkage(squareform(distance_matrix), method='average')
    full_labels = fcluster(Z_full, t=n_clusters, criterion='maxclust') - 1

    all_labels = np.zeros((n_subjects, n_resamples))

    for i in range(n_resamples):
        sample_size = int(0.8 * n_subjects)
        train_idx = rng.choice(n_subjects, size=sample_size, replace=True) # Reproducible sampling

        train_data = combined_cvs.iloc[train_idx]

        train_cosine_dist = pdist(train_data, metric='cosine')
        train_similarity = 1 - squareform(train_cosine_dist)
        train_distance = 1 - train_similarity

        Z_train = linkage(squareform(train_distance), method='average')
        train_labels = fcluster(Z_train, t=n_clusters, criterion='maxclust') - 1

        resample_labels = np.full(n_subjects, -1)
        for j, idx in enumerate(train_idx):
            resample_labels[idx] = train_labels[j]

        all_labels[:, i] = resample_labels

        if (i + 1) % 100 == 0 or i == n_resamples-1:
            print(f"Completed {i + 1}/{n_resamples} resamples")

    mode_labels = np.zeros(len(combined_cvs), dtype=int)

    for i in range(len(combined_cvs)):
        valid_labels = all_labels[i, all_labels[i, :] != -1]
        if len(valid_labels) > 0:
            values, counts = np.unique(valid_labels, return_counts=True)
            mode_labels[i] = values[np.argmax(counts)]
        else:
            mode_labels[i] = full_labels[i]

    mode_labels = mode_labels.astype(int)

    agreement = np.mean(mode_labels == full_labels)
    print(f"Agreement between mode labels and full dataset labels: {agreement:.3f}")

    ari = adjusted_rand_score(full_labels, mode_labels)
    print(f"Adjusted Rand Index between mode and full labels: {ari:.3f}")

    combined_cvs_with_clusters = combined_cvs.copy()
    one_based_labels = mode_labels + 1
    combined_cvs_with_clusters['cluster'] = one_based_labels

    bootstrap_stats = np.zeros((n_subjects, n_clusters))
    for i in range(n_subjects):
        valid_labels = all_labels[i, all_labels[i, :] != -1]
        if len(valid_labels) > 0:
            for k in range(n_clusters):
                bootstrap_stats[i, k] = np.mean(valid_labels == k)

    bootstrap_df = pd.DataFrame(
        bootstrap_stats,
        index=combined_cvs.index,
        columns=[f'cluster_{k+1}_prob' for k in range(n_clusters)]
    )
    bootstrap_df['assigned_cluster'] = one_based_labels

    stability_metrics = {
        'agreement': agreement,
        'ari': ari,
        'full_labels': full_labels + 1,
        'linkage_matrix': Z_full
    }

    return one_based_labels, combined_cvs_with_clusters, bootstrap_df, stability_metrics


def analyze_cluster_profiles(combined_cvs, cluster_labels):
    """
    Analyze cluster profiles in terms of canonical variates
    
    Parameters:
    -----------
    combined_cvs : DataFrame
        DataFrame containing combined canonical variates
    cluster_labels : array-like
        Cluster labels for each subject
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure with the heatmap
    cluster_profiles : DataFrame
        Mean canonical variate values for each cluster
    """
    print("\n===== Analyzing Cluster Profiles =====")
    
    # Create a DataFrame with cluster labels
    data_with_clusters = combined_cvs.copy()
    data_with_clusters['cluster'] = cluster_labels
    
    # Compute cluster profiles (mean values for each feature per cluster)
    cluster_profiles = data_with_clusters.groupby('cluster').mean()
    
    # Display cluster profiles
    print("\nCluster Profiles (mean values):")
    print(cluster_profiles)
    
    # Plot cluster profiles
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap of cluster profiles
    sns.heatmap(cluster_profiles, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title('Cluster Profiles (Canonical Variate Mean Values)')
    plt.tight_layout()
    
    fig = plt.gcf()
    
    return fig, cluster_profiles