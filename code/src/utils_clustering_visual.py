'''
Visualization utilities for clustering analysis
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, fcluster


def plot_dendrogram(linkage_matrix, truncate_mode=None, p=None, orientation='top', title=None, figsize=(12, 8)):
    """
    ...
    """
    plt.figure(figsize=figsize)
    
    # Create arguments dictionary for dendrogram with conditional parameters
    dendro_args = {
        'Z': linkage_matrix,
        'orientation': orientation
    }
    
    # Only add truncate_mode and p if both are specified
    if truncate_mode is not None:
        dendro_args['truncate_mode'] = truncate_mode
        # p must be a number when truncate_mode is used
        if p is not None:
            dendro_args['p'] = p
        else:
            # Default p to a reasonable value if not provided
            dendro_args['p'] = 30
    
    # Call dendrogram with the constructed arguments
    dendrogram(**dendro_args)
    
    if title:
        plt.title(title)
        
    plt.xlabel('Subject Index' if orientation in ['top', 'bottom'] else 'Distance')
    plt.ylabel('Distance' if orientation in ['top', 'bottom'] else 'Subject Index')
    
    return plt.gcf()


def plot_cluster_evaluation_metrics(n_clusters_range, silhouette_scores, ch_scores, db_scores, figsize=(15, 5)):
    """
    Plot evaluation metrics for different numbers of clusters
    
    Parameters:
    -----------
    n_clusters_range : list
        Range of number of clusters evaluated
    silhouette_scores : list
        Silhouette scores for each number of clusters
    ch_scores : list
        Calinski-Harabasz scores for each number of clusters
    db_scores : list
        Davies-Bouldin scores for each number of clusters
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 3, 1)
    plt.plot(n_clusters_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    
    plt.subplot(1, 3, 2)
    plt.plot(n_clusters_range, ch_scores, marker='o')
    plt.title('Calinski-Harabasz Index')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    
    plt.subplot(1, 3, 3)
    plt.plot(n_clusters_range, db_scores, marker='o')
    plt.title('Davies-Bouldin Index')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    
    plt.tight_layout()
    
    return plt.gcf()


def plot_cluster_distribution(cluster_labels, figsize=(10, 6)):
    """
    Plot distribution of subjects across clusters
    
    Parameters:
    -----------
    cluster_labels : array-like
        Cluster labels for each subject
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure
    summary_df : DataFrame
        DataFrame with cluster distribution statistics
    """
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    
    plt.figure(figsize=figsize)
    bars = plt.bar(cluster_counts.index, cluster_counts.values)
    
    # Add percentage labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = height / len(cluster_labels) * 100
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1,
            f'{percentage:.1f}%',
            ha='center',
            va='bottom'
        )
    
    plt.title('Distribution of Subjects Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Subjects')
    plt.xticks(cluster_counts.index)
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'cluster': cluster_counts.index,
        'count': cluster_counts.values,
        'percentage': [count/len(cluster_labels)*100 for count in cluster_counts.values]
    })
    
    return plt.gcf(), summary_df


def create_dendrogram_heatmap(linkage_matrix, data_matrix, feature_names=None, subject_ids=None, 
                           optimal_clusters=None, output_path=None, figsize=(14, 10), 
                           cmap='coolwarm', label_fontsize=10, title_fontsize=14, dpi=300):
    """
    Create a publication-quality visualization with a dendrogram and heatmap side by side,
    similar to the Buch et al. example.
    
    Parameters:
    -----------
    linkage_matrix : array
        Linkage matrix from hierarchical clustering
    data_matrix : array or DataFrame
        Data matrix for heatmap visualization (samples x features)
    feature_names : list, optional
        Names of features for column labels
    subject_ids : list, optional
        Subject IDs for row labels
    optimal_clusters : int, optional
        Number of clusters to color in the dendrogram
    output_path : Path, optional
        Path to save visualization
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap for heatmap
    label_fontsize : int, optional
        Font size for labels
    title_fontsize : int, optional
        Font size for titles
    dpi : int, optional
        Resolution for saved image
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure object
    """
    print("\n===== Creating Dendrogram-Heatmap Visualization =====")
    
    # Convert data_matrix to DataFrame if it's not already
    if not isinstance(data_matrix, pd.DataFrame):
        if feature_names is not None:
            columns = feature_names
        else:
            columns = [f'Feature {i+1}' for i in range(data_matrix.shape[1])]
            
        if subject_ids is not None:
            index = subject_ids
        else:
            index = [f'Subject {i+1}' for i in range(data_matrix.shape[0])]
            
        data_matrix = pd.DataFrame(data_matrix, index=index, columns=columns)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                  gridspec_kw={'width_ratios': [1, 2]})
    
    # Plot dendrogram on the left
    if optimal_clusters is not None:
        # Calculate the cut threshold for the dendrogram coloring
        # This is the height at which to cut the tree to get the desired number of clusters
        color_threshold = linkage_matrix[-optimal_clusters+1, 2]
    else:
        color_threshold = 0  # Auto coloring
    
    # Create a palette of colors for the clusters
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot the dendrogram
    dend = dendrogram(
        linkage_matrix,
        ax=ax1,
        orientation='left',  # Orient dendrogram to the left like in the example
        leaf_font_size=label_fontsize,
        above_threshold_color='black',
        color_threshold=color_threshold
    )
    
    # Get the order of samples from the dendrogram
    leaf_order = dend['leaves']
    
    # Reorder the data matrix to match the dendrogram leaf order
    reordered_data = data_matrix.iloc[leaf_order]
    
    # Plot heatmap on the right
    sns.heatmap(
        reordered_data, 
        ax=ax2, 
        cmap=cmap, 
        cbar_kws={'label': 'Value'},
        xticklabels=True,
        yticklabels=False  # Hide y-tick labels as they're shown in dendrogram
    )
    
    # If optimal_clusters is provided, add cluster color bars
    if optimal_clusters is not None:
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, t=optimal_clusters, criterion='maxclust')
        
        # Create a mapping of leaf index to cluster
        leaf_to_cluster = dict(zip(range(len(cluster_labels)), cluster_labels))
        
        # Add a narrow subplot for cluster labels
        divider = plt.subplot(233)
        plt.close()  # Close the automatically created figure
        ax3 = fig.add_axes([0.725, 0.1, 0.05, 0.8])  # Adjust position as needed
        
        # Create a colormap for clusters
        cluster_cmap = plt.cm.get_cmap('tab10', optimal_clusters)
        
        # Create a matrix for the cluster colorbar
        cluster_matrix = np.zeros((len(leaf_order), 1))
        for i, leaf in enumerate(leaf_order):
            cluster_matrix[i, 0] = leaf_to_cluster.get(leaf, 0) - 1  # 0-based for colormap
        
        # Plot the cluster colorbar
        sns.heatmap(
            cluster_matrix, 
            ax=ax3, 
            cmap=cluster_cmap, 
            cbar=False,
            xticklabels=False,
            yticklabels=False
        )
        
        # Add cluster numbers as text
        for i in range(optimal_clusters):
            # Position text in the middle of each cluster's color bar section
            positions = np.where(cluster_matrix == i)[0]
            if len(positions) > 0:
                mid_pos = (positions.min() + positions.max()) / 2
                ax3.text(0.5, mid_pos, str(i+1), 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=12)
        
        ax3.set_title('Cluster', fontsize=label_fontsize)
    
    # Set titles and styling
    ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=title_fontsize)
    ax2.set_title('Feature Heatmap', fontsize=title_fontsize)
    
    # Fix dendrogram aesthetics
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_xlabel('Distance', fontsize=label_fontsize)
    ax1.set_ylabel('')
    
    # Fix heatmap aesthetics
    ax2.set_xlabel('Features', fontsize=label_fontsize)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=label_fontsize-2)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path is not None:
        output_file = Path(output_path, 'dendrogram_heatmap.png')
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved visualization to: {output_file}")
    
    return fig


def create_bimodal_dendrogram_heatmap(linkage_matrix, combined_cvs, gaze_columns=None,
                                     fmri_columns=None, output_path=None, 
                                     optimal_clusters=None, figsize=(18, 10), dpi=300,
                                     clinical_data=None):
    """
    Create a publication-quality visualization specifically for bi-modal data (fMRI & video features)
    
    Parameters:
    -----------
    linkage_matrix : array
        Linkage matrix from hierarchical clustering
    combined_cvs : DataFrame
        DataFrame containing combined canonical variates (with both fMRI and video features)
    gaze_columns : list, optional
        Column names for gaze (video) features
    fmri_columns : list, optional
        Column names for fMRI features
    output_path : Path, optional
        Path to save visualization
    optimal_clusters : int, optional
        Number of clusters to color in the dendrogram
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        Resolution for saved image
    clinical_data : DataFrame, optional
        Clinical/demographic data to include in visualization
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure object
    """
    import matplotlib.gridspec as gridspec
    
    print("\n===== Creating Bi-modal Dendrogram-Heatmap Visualization =====")
    
    # If gaze and fMRI column names are not provided, try to infer them
    if gaze_columns is None and fmri_columns is None:
        # Assume first half of columns are gaze, second half are fMRI
        n_cols = combined_cvs.shape[1]
        if 'cluster' in combined_cvs.columns:
            n_cols = n_cols - 1  # Don't count cluster column
            combined_feature_cols = [col for col in combined_cvs.columns if col != 'cluster']
        else:
            combined_feature_cols = combined_cvs.columns
            
        half_cols = len(combined_feature_cols) // 2
        gaze_columns = combined_feature_cols[:half_cols]
        fmri_columns = combined_feature_cols[half_cols:]
        
        print(f"Automatically identified {len(gaze_columns)} gaze columns and {len(fmri_columns)} fMRI columns")
    
    # Create figure with GridSpec for flexible subplot layout
    fig = plt.figure(figsize=figsize)
    
    if clinical_data is not None:
        # More complex layout with clinical data
        gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 2, 2, 1])
        ax_dendro = fig.add_subplot(gs[0])
        ax_gaze = fig.add_subplot(gs[1])
        ax_fmri = fig.add_subplot(gs[2])
        ax_clinical = fig.add_subplot(gs[3])
    else:
        # Simpler layout without clinical data
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 2, 2])
        ax_dendro = fig.add_subplot(gs[0])
        ax_gaze = fig.add_subplot(gs[1])
        ax_fmri = fig.add_subplot(gs[2])
    
    # Plot dendrogram
    if optimal_clusters is not None:
        color_threshold = linkage_matrix[-optimal_clusters+1, 2]
    else:
        color_threshold = 0  # Auto coloring
    
    dend = dendrogram(
        linkage_matrix,
        ax=ax_dendro,
        orientation='left',
        leaf_font_size=10,
        above_threshold_color='black',
        color_threshold=color_threshold
    )
    
    # Get the order of samples from the dendrogram
    leaf_order = dend['leaves']
    
    # Create a copy of the DataFrame for safety
    data_df = combined_cvs.copy()
    
    # Print columns for debugging
    print(f"Columns in data_df: {data_df.columns.tolist()}")
    
    # Identify and handle subject IDs (for reference, not required for visualization)
    subject_ids = None
    
    # Extract gaze and fMRI features
    # First remove any non-feature columns
    feature_df = data_df.copy()
    
    # Remove ID columns if they exist
    id_cols = [col for col in feature_df.columns 
               if isinstance(col, str) and any(term in col.lower() for term in ['id', 'subject'])]
    if id_cols:
        print(f"Removing ID columns: {id_cols}")
        feature_df = feature_df.drop(columns=id_cols, errors='ignore')
    
    # Remove cluster columns if they exist
    cluster_cols = [col for col in feature_df.columns 
                    if isinstance(col, str) and 'cluster' in col.lower()]
    if cluster_cols:
        print(f"Removing cluster columns: {cluster_cols}")
        feature_df = feature_df.drop(columns=cluster_cols, errors='ignore')
    
    # Now reorder the feature data according to the dendrogram
    try:
        # Try the simplest approach first - direct reordering by position
        reordered_data = feature_df.iloc[leaf_order]
        print("Successfully reordered data using iloc")
    except Exception as e1:
        print(f"Error reordering with iloc: {e1}")
        print("Warning: Could not reorder data properly, visualization may not align with dendrogram")
        # Just use the original data as a fallback
        reordered_data = feature_df
    
    # Now extract gaze and fMRI portions
    try:
        # Check if the provided column names exist
        missing_gaze = [col for col in gaze_columns if col not in reordered_data.columns]
        missing_fmri = [col for col in fmri_columns if col not in reordered_data.columns]
        
        if missing_gaze or missing_fmri:
            raise KeyError(f"Missing columns: {missing_gaze + missing_fmri}")
        
        gaze_data = reordered_data[gaze_columns]
        fmri_data = reordered_data[fmri_columns]
        print("Successfully extracted gaze and fMRI data")
    except Exception as e:
        print(f"Error extracting gaze/fMRI data: {e}")
        print("Using alternative approach - splitting features in half")
        
        all_feature_cols = reordered_data.columns.tolist()
        half_point = len(all_feature_cols) // 2
        
        gaze_data = reordered_data.iloc[:, :half_point]
        fmri_data = reordered_data.iloc[:, half_point:]
        print(f"Using {gaze_data.shape[1]} columns for gaze and {fmri_data.shape[1]} columns for fMRI")
    
    # Create heatmaps
    sns.heatmap(
        gaze_data, 
        ax=ax_gaze, 
        cmap='Reds', 
        cbar_kws={'label': 'Gaze Feature Value'},
        xticklabels=True,
        yticklabels=False
    )
    
    sns.heatmap(
        fmri_data, 
        ax=ax_fmri, 
        cmap='Blues', 
        cbar_kws={'label': 'fMRI Feature Value'},
        xticklabels=True,
        yticklabels=False
    )
    
    # If clinical data is provided
    if clinical_data is not None:
        try:
            # Try to determine the ID column in clinical data
            id_column = None
            for col in clinical_data.columns:
                if isinstance(col, str) and ('id' in col.lower() or 'subject' in col.lower()):
                    id_column = col
                    break
            
            if id_column is None:
                raise ValueError("No ID column found in clinical data")
            
            # Just use the clinical data as is, ordered by the dendrogram leaf order
            # This is a simplification since we can't match subjects without proper IDs
            clinical_subset = clinical_data.drop(columns=[id_column], errors='ignore')
            
            # Just use the first N rows where N is the number of subjects in our dendrogram
            if len(clinical_subset) >= len(leaf_order):
                ordered_clinical = clinical_subset.iloc[leaf_order]
                
                # Remove any remaining ID-like columns
                id_cols = [col for col in ordered_clinical.columns 
                          if isinstance(col, str) and ('id' in col.lower() or 'subject' in col.lower())]
                ordered_clinical = ordered_clinical.drop(columns=id_cols, errors='ignore')
                
                # Create clinical heatmap
                sns.heatmap(
                    ordered_clinical, 
                    ax=ax_clinical, 
                    cmap='viridis', 
                    cbar_kws={'label': 'Clinical Measure Value'},
                    xticklabels=True,
                    yticklabels=False
                )
                
                ax_clinical.set_title('Clinical Features', fontsize=14)
                ax_clinical.set_xlabel('Clinical Measures', fontsize=10)
                ax_clinical.set_xticklabels(ax_clinical.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            else:
                print("Warning: Not enough clinical data rows to match dendrogram")
        except Exception as e:
            print(f"Error processing clinical data: {e}")
    
    # Add cluster information if optimal_clusters is provided
    if optimal_clusters is not None:
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, t=optimal_clusters, criterion='maxclust')
        
        # Add cluster labels to figure
        for i, cluster_num in enumerate(cluster_labels[leaf_order]):
            ax_dendro.text(-0.1, i, str(cluster_num), 
                      horizontalalignment='right',
                      verticalalignment='center',
                      fontsize=10,
                      fontweight='bold',
                      color=plt.cm.tab10(cluster_num % 10))
    
    # Set titles and styling
    ax_dendro.set_title('Hierarchical Clustering\nDendrogram', fontsize=14)
    ax_gaze.set_title('Gaze (Video) Features', fontsize=14)
    ax_fmri.set_title('fMRI Features', fontsize=14)
    
    # Fix dendrogram aesthetics
    ax_dendro.spines['top'].set_visible(False)
    ax_dendro.spines['right'].set_visible(False)
    ax_dendro.spines['bottom'].set_visible(False)
    ax_dendro.spines['left'].set_visible(False)
    ax_dendro.set_xlabel('Distance', fontsize=10)
    ax_dendro.set_ylabel('')
    
    # Fix heatmap x-axis labels
    for ax, title in [(ax_gaze, 'Gaze Features'), (ax_fmri, 'fMRI Features')]:
        ax.set_xlabel(title, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path is not None:
        output_file = Path(output_path, 'bimodal_dendrogram_heatmap.png')
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved bi-modal visualization to: {output_file}")
    
    return fig


def plot_stability_metrics(bootstrap_df, figsize=(12, 10)):
    """
    Plot stability metrics from bootstrap resampling
    
    Parameters:
    -----------
    bootstrap_df : DataFrame
        DataFrame with bootstrap stability information
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    
    # Extract cluster probability columns
    prob_cols = [col for col in bootstrap_df.columns if 'cluster_' in col and '_prob' in col]
    
    # Get the assigned cluster for each subject
    assigned_cluster = bootstrap_df['assigned_cluster']
    
    # Create subplot layout based on number of clusters
    n_clusters = len(prob_cols)
    n_rows = int(np.ceil(n_clusters / 2))
    
    for i, col in enumerate(prob_cols):
        plt.subplot(n_rows, 2, i+1)
        
        # Create histogram of assignment probabilities
        plt.hist(bootstrap_df[col], bins=20, alpha=0.7)
        
        # Highlight subjects assigned to this cluster
        cluster_num = int(col.split('_')[1])
        mask = assigned_cluster == cluster_num
        if mask.any():
            plt.hist(bootstrap_df.loc[mask, col], bins=20, alpha=0.7, color='red')
        
        plt.title(f'Probability of Assignment to Cluster {cluster_num}')
        plt.xlabel('Probability')
        plt.ylabel('Number of Subjects')
    
    plt.tight_layout()
    
    return plt.gcf()


def save_visualization(fig, output_path, filename, dpi=300):
    """
    Save a visualization figure to a file
    
    Parameters:
    -----------
    fig : Figure
        Matplotlib figure to save
    output_path : str or Path
        Directory to save the figure
    filename : str
        Filename for the figure
    dpi : int, optional
        Resolution for the saved image
    """
    output_file = Path(output_path, filename)
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to: {output_file}")


'''
Buch et al. 2023 inspired a specialized visualization for bimodal data (fMRI & video features).
'''
def create_bimodal_dendrogram_heatmap(
    linkage_matrix, 
    combined_cvs, 
    gaze_columns=None,
    fmri_columns=None, 
    output_path=None, 
    optimal_clusters=None,
    figsize=(18, 10),
    dpi=300,
    clinical_data=None
):
    """
    Create a publication-quality visualization specifically for bi-modal data (fMRI & video features)
    
    Parameters:
    -----------
    linkage_matrix : array
        Linkage matrix from hierarchical clustering
    combined_cvs : DataFrame
        DataFrame containing combined canonical variates (with both fMRI and video features)
    gaze_columns : list, optional
        Column names for gaze (video) features
    fmri_columns : list, optional
        Column names for fMRI features
    output_path : Path, optional
        Path to save visualization
    optimal_clusters : int, optional
        Number of clusters to color in the dendrogram
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        Resolution for saved image
    clinical_data : DataFrame, optional
        Clinical/demographic data to include in visualization
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import dendrogram, fcluster
    from pathlib import Path
    
    print("\n===== Creating Bi-modal Dendrogram-Heatmap Visualization =====")
    
    # If gaze and fMRI column names are not provided, try to infer them
    if gaze_columns is None and fmri_columns is None:
        # Assume first half of columns are gaze, second half are fMRI
        n_cols = combined_cvs.shape[1]
        if 'cluster' in combined_cvs.columns:
            n_cols = n_cols - 1  # Don't count cluster column
            combined_feature_cols = [col for col in combined_cvs.columns if col != 'cluster']
        else:
            combined_feature_cols = combined_cvs.columns
            
        half_cols = len(combined_feature_cols) // 2
        gaze_columns = combined_feature_cols[:half_cols]
        fmri_columns = combined_feature_cols[half_cols:]
        
        print(f"Automatically identified {len(gaze_columns)} gaze columns and {len(fmri_columns)} fMRI columns")
    
    # Create figure with GridSpec for flexible subplot layout
    fig = plt.figure(figsize=figsize)
    
    if clinical_data is not None:
        # More complex layout with clinical data
        gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 2, 2, 1])
        ax_dendro = fig.add_subplot(gs[0])
        ax_gaze = fig.add_subplot(gs[1])
        ax_fmri = fig.add_subplot(gs[2])
        ax_clinical = fig.add_subplot(gs[3])
    else:
        # Simpler layout without clinical data
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 2, 2])
        ax_dendro = fig.add_subplot(gs[0])
        ax_gaze = fig.add_subplot(gs[1])
        ax_fmri = fig.add_subplot(gs[2])
    
    # Plot dendrogram
    if optimal_clusters is not None:
        color_threshold = linkage_matrix[-optimal_clusters+1, 2]
    else:
        color_threshold = 0  # Auto coloring
    
    dend = dendrogram(
        linkage_matrix,
        ax=ax_dendro,
        orientation='left',
        leaf_font_size=10,
        above_threshold_color='black',
        color_threshold=color_threshold
    )
    
    # Get the order of samples from the dendrogram
    leaf_order = dend['leaves']
    
    # Create a copy of the DataFrame for safety
    data_df = combined_cvs.copy()
    
    # Print columns for debugging
    print(f"Columns in data_df: {data_df.columns.tolist()}")
    
    # Identify and handle subject IDs (for reference, not required for visualization)
    subject_ids = None
    
    # Extract gaze and fMRI features
    # First remove any non-feature columns
    feature_df = data_df.copy()
    
    # Remove ID columns if they exist
    id_cols = [col for col in feature_df.columns 
               if isinstance(col, str) and any(term in col.lower() for term in ['id', 'subject'])]
    if id_cols:
        print(f"Removing ID columns: {id_cols}")
        feature_df = feature_df.drop(columns=id_cols, errors='ignore')
    
    # Remove cluster columns if they exist
    cluster_cols = [col for col in feature_df.columns 
                    if isinstance(col, str) and 'cluster' in col.lower()]
    if cluster_cols:
        print(f"Removing cluster columns: {cluster_cols}")
        feature_df = feature_df.drop(columns=cluster_cols, errors='ignore')
    
    # Now reorder the feature data according to the dendrogram
    try:
        # Try the simplest approach first - direct reordering by position
        reordered_data = feature_df.iloc[leaf_order]
        print("Successfully reordered data using iloc")
    except Exception as e1:
        print(f"Error reordering with iloc: {e1}")
        print("Warning: Could not reorder data properly, visualization may not align with dendrogram")
        # Just use the original data as a fallback
        reordered_data = feature_df
    
    # Now extract gaze and fMRI portions
    try:
        # Check if the provided column names exist
        missing_gaze = [col for col in gaze_columns if col not in reordered_data.columns]
        missing_fmri = [col for col in fmri_columns if col not in reordered_data.columns]
        
        if missing_gaze or missing_fmri:
            raise KeyError(f"Missing columns: {missing_gaze + missing_fmri}")
        
        gaze_data = reordered_data[gaze_columns]
        fmri_data = reordered_data[fmri_columns]
        print("Successfully extracted gaze and fMRI data")
    except Exception as e:
        print(f"Error extracting gaze/fMRI data: {e}")
        print("Using alternative approach - splitting features in half")
        
        all_feature_cols = reordered_data.columns.tolist()
        half_point = len(all_feature_cols) // 2
        
        gaze_data = reordered_data.iloc[:, :half_point]
        fmri_data = reordered_data.iloc[:, half_point:]
        print(f"Using {gaze_data.shape[1]} columns for gaze and {fmri_data.shape[1]} columns for fMRI")
    
    # Create heatmaps
    sns.heatmap(
        gaze_data, 
        ax=ax_gaze, 
        cmap='Reds', 
        cbar_kws={'label': 'Gaze Feature Value'},
        xticklabels=True,
        yticklabels=False
    )
    
    sns.heatmap(
        fmri_data, 
        ax=ax_fmri, 
        cmap='Blues', 
        cbar_kws={'label': 'fMRI Feature Value'},
        xticklabels=True,
        yticklabels=False
    )
    
    # If clinical data is provided
    if clinical_data is not None:
        try:
            # Try to determine the ID column in clinical data
            id_column = None
            for col in clinical_data.columns:
                if isinstance(col, str) and ('id' in col.lower() or 'subject' in col.lower()):
                    id_column = col
                    break
            
            if id_column is None:
                raise ValueError("No ID column found in clinical data")
            
            # Just use the clinical data as is, ordered by the dendrogram leaf order
            # This is a simplification since we can't match subjects without proper IDs
            clinical_subset = clinical_data.drop(columns=[id_column], errors='ignore')
            
            # Just use the first N rows where N is the number of subjects in our dendrogram
            if len(clinical_subset) >= len(leaf_order):
                ordered_clinical = clinical_subset.iloc[leaf_order]
                
                # Remove any remaining ID-like columns
                id_cols = [col for col in ordered_clinical.columns 
                          if isinstance(col, str) and ('id' in col.lower() or 'subject' in col.lower())]
                ordered_clinical = ordered_clinical.drop(columns=id_cols, errors='ignore')
                
                # Create clinical heatmap
                sns.heatmap(
                    ordered_clinical, 
                    ax=ax_clinical, 
                    cmap='viridis', 
                    cbar_kws={'label': 'Clinical Measure Value'},
                    xticklabels=True,
                    yticklabels=False
                )
                
                ax_clinical.set_title('Clinical Features', fontsize=14)
                ax_clinical.set_xlabel('Clinical Measures', fontsize=10)
                ax_clinical.set_xticklabels(ax_clinical.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            else:
                print("Warning: Not enough clinical data rows to match dendrogram")
        except Exception as e:
            print(f"Error processing clinical data: {e}")
    
    # Add cluster information if optimal_clusters is provided
    if optimal_clusters is not None:
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, t=optimal_clusters, criterion='maxclust')
        
        # Add cluster labels to figure
        for i, cluster_num in enumerate(cluster_labels[leaf_order]):
            ax_dendro.text(-0.1, i, str(cluster_num), 
                      horizontalalignment='right',
                      verticalalignment='center',
                      fontsize=10,
                      fontweight='bold',
                      color=plt.cm.tab10(cluster_num % 10))
    
    # Set titles and styling
    ax_dendro.set_title('Hierarchical Clustering\nDendrogram', fontsize=14)
    ax_gaze.set_title('Gaze (Video) Features', fontsize=14)
    ax_fmri.set_title('fMRI Features', fontsize=14)
    
    # Fix dendrogram aesthetics
    ax_dendro.spines['top'].set_visible(False)
    ax_dendro.spines['right'].set_visible(False)
    ax_dendro.spines['bottom'].set_visible(False)
    ax_dendro.spines['left'].set_visible(False)
    ax_dendro.set_xlabel('Distance', fontsize=10)
    ax_dendro.set_ylabel('')
    
    # Fix heatmap x-axis labels
    for ax, title in [(ax_gaze, 'Gaze Features'), (ax_fmri, 'fMRI Features')]:
        ax.set_xlabel(title, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path is not None:
        output_file = Path(output_path, 'bimodal_dendrogram_heatmap.png')
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved bi-modal visualization to: {output_file}")
    
    return fig

'''
Utility function to create a publication-quality visualization with a dendrogram and heatmap side by side.
'''

def create_dendrogram_heatmap(
    linkage_matrix, 
    data_matrix, 
    feature_names=None, 
    subject_ids=None, 
    optimal_clusters=None, 
    output_path=None, 
    figsize=(14, 10), 
    cmap='coolwarm',
    label_fontsize=10,
    title_fontsize=14,
    dpi=300
):
    """
    Create a publication-quality visualization with a dendrogram and heatmap side by side,
    similar to the Buch et al. example.
    
    Parameters:
    -----------
    linkage_matrix : array
        Linkage matrix from hierarchical clustering
    data_matrix : array or DataFrame
        Data matrix for heatmap visualization (samples x features)
    feature_names : list, optional
        Names of features for column labels
    subject_ids : list, optional
        Subject IDs for row labels
    optimal_clusters : int, optional
        Number of clusters to color in the dendrogram
    output_path : Path, optional
        Path to save visualization
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap for heatmap
    label_fontsize : int, optional
        Font size for labels
    title_fontsize : int, optional
        Font size for titles
    dpi : int, optional
        Resolution for saved image
    
    Returns:
    --------
    fig : Figure
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, fcluster
    import pandas as pd
    from pathlib import Path
    
    print("\n===== Creating Dendrogram-Heatmap Visualization =====")
    
    # Convert data_matrix to DataFrame if it's not already
    if not isinstance(data_matrix, pd.DataFrame):
        if feature_names is not None:
            columns = feature_names
        else:
            columns = [f'Feature {i+1}' for i in range(data_matrix.shape[1])]
            
        if subject_ids is not None:
            index = subject_ids
        else:
            index = [f'Subject {i+1}' for i in range(data_matrix.shape[0])]
            
        data_matrix = pd.DataFrame(data_matrix, index=index, columns=columns)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                  gridspec_kw={'width_ratios': [1, 2]})
    
    # Plot dendrogram on the left
    if optimal_clusters is not None:
        # Calculate the cut threshold for the dendrogram coloring
        # This is the height at which to cut the tree to get the desired number of clusters
        color_threshold = linkage_matrix[-optimal_clusters+1, 2]
    else:
        color_threshold = 0  # Auto coloring
    
    # Create a palette of colors for the clusters
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot the dendrogram
    dend = dendrogram(
        linkage_matrix,
        ax=ax1,
        orientation='left',  # Orient dendrogram to the left like in the example
        leaf_font_size=label_fontsize,
        above_threshold_color='black',
        color_threshold=color_threshold
    )
    
    # Get the order of samples from the dendrogram
    leaf_order = dend['leaves']
    
    # Reorder the data matrix to match the dendrogram leaf order
    reordered_data = data_matrix.iloc[leaf_order]
    
    # Plot heatmap on the right
    sns.heatmap(
        reordered_data, 
        ax=ax2, 
        cmap=cmap, 
        cbar_kws={'label': 'Value'},
        xticklabels=True,
        yticklabels=False  # Hide y-tick labels as they're shown in dendrogram
    )
    
    # If optimal_clusters is provided, add cluster color bars
    if optimal_clusters is not None:
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, t=optimal_clusters, criterion='maxclust')
        
        # Create a mapping of leaf index to cluster
        leaf_to_cluster = dict(zip(range(len(cluster_labels)), cluster_labels))
        
        # Add a narrow subplot for cluster labels
        divider = plt.subplot(233)
        plt.close()  # Close the automatically created figure
        ax3 = fig.add_axes([0.725, 0.1, 0.05, 0.8])  # Adjust position as needed
        
        # Create a colormap for clusters
        cluster_cmap = plt.cm.get_cmap('tab10', optimal_clusters)
        
        # Create a matrix for the cluster colorbar
        cluster_matrix = np.zeros((len(leaf_order), 1))
        for i, leaf in enumerate(leaf_order):
            cluster_matrix[i, 0] = leaf_to_cluster.get(leaf, 0) - 1  # 0-based for colormap
        
        # Plot the cluster colorbar
        sns.heatmap(
            cluster_matrix, 
            ax=ax3, 
            cmap=cluster_cmap, 
            cbar=False,
            xticklabels=False,
            yticklabels=False
        )
        
        # Add cluster numbers as text
        for i in range(optimal_clusters):
            # Position text in the middle of each cluster's color bar section
            positions = np.where(cluster_matrix == i)[0]
            if len(positions) > 0:
                mid_pos = (positions.min() + positions.max()) / 2
                ax3.text(0.5, mid_pos, str(i+1), 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=12)
        
        ax3.set_title('Cluster', fontsize=label_fontsize)
    
    # Set titles and styling
    ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=title_fontsize)
    ax2.set_title('Feature Heatmap', fontsize=title_fontsize)
    
    # Fix dendrogram aesthetics
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_xlabel('Distance', fontsize=label_fontsize)
    ax1.set_ylabel('')
    
    # Fix heatmap aesthetics
    ax2.set_xlabel('Features', fontsize=label_fontsize)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=label_fontsize-2)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path is not None:
        output_file = Path(output_path, 'dendrogram_heatmap.png')
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved visualization to: {output_file}")
    
    return fig

def visualize_clusters(combined_cvs, cluster_labels, method='pca'):
    """
    Visualize clusters using dimensionality reduction
    
    Parameters:
    -----------
    combined_cvs : DataFrame
        DataFrame containing combined canonical variates
    cluster_labels : array-like
        Cluster labels for each subject
    method : str, optional
        Dimensionality reduction method ('pca', 'tsne', or 'umap')
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure with the visualization
    embedding : array
        2D embedding of the data
    """
    print(f"\n===== Visualizing Clusters using {method.upper()} =====")
    
    # Ensure needed packages are imported within the function
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        except ImportError:
            print("UMAP not installed. Falling back to PCA.")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            method = 'pca'
    else:
        print(f"Unknown method: {method}. Falling back to PCA.")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        method = 'pca'
    
    # Apply dimensionality reduction
    embedding = reducer.fit_transform(combined_cvs)
    
    # Plot the embedding
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Cluster Visualization using {method.upper()}')
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    
    # Add subject IDs as annotations
    for i, subj_id in enumerate(combined_cvs.index):
        plt.annotate(subj_id, (embedding[i, 0], embedding[i, 1]), fontsize=8, alpha=0.7)
    
    fig = plt.gcf()
    
    return fig, embedding