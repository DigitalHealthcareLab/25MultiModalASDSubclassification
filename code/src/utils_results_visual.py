"""
Utility functions for visualizing cluster results.
These functions support the main visualize_cluster_results.py script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


def load_data(output_tables_dir):
    """Load the necessary data files for visualization."""
    print("Loading data...")
    
    # Load participant data with clinical and cluster information
    participant_df = pd.read_csv(os.path.join(output_tables_dir, 'participant_clinical_cluster_df.csv'))
    
    print(f"Loaded participant data with shape: {participant_df.shape}")
    return participant_df


def create_clinical_profile_heatmap(df, output_path=None):
    """
    Create a heatmap showing z-scores of key clinical variables across clusters.
    
    Parameters:
    -----------
    df : DataFrame
        Participant data with cluster assignments
    output_path : str
        Where to save the figure
    """
    print("Creating clinical profile heatmap...")
    
    # Define key variables to include in the heatmap with improved labels
    domains_and_vars = [
        ('Joint Attention', [
            ('ija_success', 'IJA Success'),
            ('rja_low_success', 'Low Level RJA Success'),
            ('rja_high_success', 'High Level RJA Success')
        ]),
        ('Cognitive', [
            ('fsiq', 'FSIQ'),
            ('vci', 'VCI'),
            ('vsi', 'VSI')
        ]),
        ('Gold Standard ASD Diagnostic (ADOS)', [
            ('total_css', 'ADOS CSS Total'),
            ('sa_css', 'Social Affect CSS'),
            ('rrb_css', 'RRB CSS')
        ]),
        ('SRS Subscales', [
            ('srs_awareness', 'Awareness'),
            ('srs_cognition', 'Cognition'),
            ('srs_comm', 'Communication')
        ]),
        ('Adaptive', [
            ('vabs_daily', 'VABS Daily'),
            ('vabs_socialization', 'VABS Socialization')
        ]),
        ('Behavioral', [
            ('cbcl_total', 'CBCL Total'),
            ('withdrawal', 'Withdrawal')
        ]),
        ('Motor', [
            ('dcdq_control', 'Control During Movement'),
            ('dcdq_coord', 'General Coordination'),
            ('dcdq_finemotor', 'Finemotor/handwriting')
        ])
    ]
    
    # Filter to available variables and prepare data for heatmap
    all_vars = []
    all_labels = []
    
    for domain, vars_list in domains_and_vars:
        for var_col, var_label in vars_list:
            if var_col in df.columns:
                all_vars.append(var_col)
                all_labels.append(f"{domain}: {var_label}")
    
    # Calculate z-scores for each variable by cluster
    clusters = sorted(df['cluster'].unique())
    n_clusters = len(clusters)
    n_vars = len(all_vars)
    
    # Create matrix to hold z-scores
    z_scores = np.zeros((n_vars, n_clusters))
    
    # Calculate z-scores
    for i, var in enumerate(all_vars):
        var_mean = df[var].mean()
        var_std = df[var].std()
        if var_std == 0:  # Handle zero standard deviation
            var_std = 1
        
        for j, cluster in enumerate(clusters):
            cluster_mean = df[df['cluster'] == cluster][var].mean()
            z_scores[i, j] = (cluster_mean - var_mean) / var_std
    
    # Create figure
    plt.figure(figsize=(12, max(8, n_vars * 0.3)))
    
    # Create custom colormap (blue to white to red)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', 
                                            [(0, 0, 0.8), (1, 1, 1), (0.8, 0, 0)], 
                                            N=256)
    
    # Plot heatmap
    im = plt.imshow(z_scores, aspect='auto', cmap=cmap, vmin=-2, vmax=2)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Z-score')
    
    # Add labels
    plt.yticks(np.arange(n_vars), all_labels)
    
    # Calculate cluster sizes and add to x-labels
    cluster_sizes = []
    for cluster in clusters:
        cluster_size = sum(df['cluster'] == cluster)
        cluster_sizes.append(cluster_size)
    
    plt.xticks(np.arange(n_clusters), [f'Subgroup {c}\n(n={s})' for c, s in zip(clusters, cluster_sizes)])
    
    # Add gridlines
    plt.grid(False)
    
    # Add title with more vertical space
    plt.title('Clinical Profile Heatmap by Cluster', fontweight='bold', pad=20, fontsize=16)
    
    # Add colorbar annotations
    plt.text(n_clusters + 0.5, 2 * n_vars / 3, 'Below Average', 
             rotation=90, va='center', ha='center', color='blue', fontsize=12)
    plt.text(n_clusters + 0.5, n_vars / 3, 'Above Average', 
             rotation=90, va='center', ha='center', color='red', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved clinical profile heatmap to {output_path}")
    
    plt.close()


def create_key_boxplots(df, output_dir=None):
    """
    Create separate boxplots for key clinical variables across clusters.
    
    Parameters:
    -----------
    df : DataFrame
        Participant data with cluster assignments
    output_dir : str
        Directory where to save the figures
    """
    from matplotlib.patches import Patch
    print("Creating key boxplots...")
    
    # Define the key domains and variables to visualize with improved labels
    domain_vars = {
        'Joint Attention Performance': [
            ('ija_success', 'IJA Success'),
            ('rja_low_success', 'Low Level RJA Success'),
            ('rja_high_success', 'High Level RJA Success')
        ],
        'Cognitive Functions': [
            ('fsiq', 'FSIQ'),
            ('vci', 'VCI'),
            ('vsi', 'VSI')
        ],
        'Gold Standard ASD Diagnostic Tool (ADOS)': [
            ('total_css', 'ADOS CSS Total'),
            ('sa_css', 'Social Affect CSS'),
            ('rrb_css', 'RRB CSS')
        ],
        'Behavioral & Adaptive Problems': [
            ('cbcl_total', 'CBCL Total'),
            ('vabs_socialization', 'VABS Socialization'),
            ('vabs_daily', 'VABS Daily')
        ],
        'SRS Subscales': [
            ('srs_awareness', 'Awareness'),
            ('srs_cognition', 'Cognition'),
            ('srs_comm', 'Communication')
        ],
        'Motor Development': [
            ('dcdq_control', 'Control During Movement'),
            ('dcdq_coord', 'General Coordination'),
            ('dcdq_finemotor', 'Finemotor/handwriting')
        ]
    }
    
    # Colors for each cluster
    colors = ['purple', 'red', 'green', 'blue']
    
    # Create a separate figure for each domain
    for domain, vars_list in domain_vars.items():
        # Filter to available variables
        available_vars = [(col, label) for col, label in vars_list if col in df.columns]
        
        if not available_vars:
            print(f"Warning: No variables available for domain {domain}")
            continue
        
        # Create a figure
        plt.figure(figsize=(12, 6))
        
        # Prepare data for boxplots
        data_to_plot = []
        var_cols = [var[0] for var in available_vars]
        var_labels = [var[1] for var in available_vars]
        
        for var_col in var_cols:
            for cluster in sorted(df['cluster'].unique()):
                # Get data for this variable and cluster
                var_data = df[df['cluster'] == cluster][var_col].dropna()
                
                # Calculate z-scores
                var_mean = df[var_col].mean()
                var_std = df[var_col].std()
                var_std = var_std if var_std > 0 else 1
                z_scores = (var_data - var_mean) / var_std
                
                data_to_plot.append((var_col, cluster, z_scores))
        
        # Create positions for the boxplots
        n_vars = len(var_cols)
        n_clusters = len(sorted(df['cluster'].unique()))
        positions = []
        for j in range(n_vars):
            for k in range(n_clusters):
                positions.append(j * (n_clusters + 1) + k)
        
        # Create boxplots
        boxplots = []
        for j, (var_col, cluster, data) in enumerate(data_to_plot):
            bp = plt.boxplot(data, positions=[positions[j]], widths=0.6, 
                           patch_artist=True, showfliers=False)
            
            # Color boxes by cluster
            for box in bp['boxes']:
                box.set_facecolor(colors[int(cluster)-1])
                box.set_alpha(0.7)
            
            boxplots.append(bp)
        
        # Set x-tick positions and labels
        tick_positions = [j * (n_clusters + 1) + (n_clusters - 1) / 2 for j in range(n_vars)]
        plt.xticks(tick_positions, var_labels, rotation=0)
        
        # Set y-axis limits
        plt.ylim(-2.5, 2.5)
        
        # Add horizontal grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add title and y-label
        plt.title(domain, fontweight='bold', fontsize=14)
        plt.ylabel('Z-score', fontsize=12)
        
        # Add legend
        legend_elements = [
            Patch(facecolor=colors[j], alpha=0.7, label=f'Subgroup {j+1}') 
            for j in range(4)
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save the figure
        if output_dir:
            domain_filename = domain.lower().replace(' ', '_').replace('&', 'and').replace('(', '').replace(')', '')
            output_path = os.path.join(output_dir, f'{domain_filename}_boxplot.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved {domain} boxplot to {output_path}")
        
        plt.close()


def create_kernel_density_plots(df, output_path=None):
    """
    Create kernel density estimation plots for key clinical relationships.
    
    Parameters:
    -----------
    df : DataFrame
        Participant data with cluster assignments
    output_path : str
        Where to save the figure
    """
    print("Creating kernel density plots...")
    
    # Define the pairs to plot with improved labels
    plot_pairs = [
        ('sa_css', 'rrb_css', 'SA-related dimension', 'RRB-related dimension'),
        ('total_css', 'ija_success', 'ADOS total score', 'Initiating JA'),
        ('total_css', 'rja_low_success', 'ADOS total score', 'Responding to Low JA'),
        ('total_css', 'rja_high_success', 'ADOS total score', 'Responding to High JA')
    ]
    
    # Check which pairs are available in the data
    available_pairs = []
    for x_var, y_var, x_label, y_label in plot_pairs:
        if x_var in df.columns and y_var in df.columns:
            available_pairs.append((x_var, y_var, x_label, y_label))
    
    if not available_pairs:
        print("Warning: No variable pairs available for kernel density plots")
        return
    
    # Create a figure with subplots
    n_pairs = len(available_pairs)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Colors for each cluster
    cluster_colors = ['purple', 'red', 'green', 'blue']
    cluster_cmaps = ['Purples', 'Reds', 'Greens', 'Blues']
    
    # Create plots for each pair
    for i, (x_var, y_var, x_label, y_label) in enumerate(available_pairs):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Calculate z-scores for the variables
        x_mean, x_std = df[x_var].mean(), df[x_var].std()
        y_mean, y_std = df[y_var].mean(), df[y_var].std()
        
        x_std = x_std if x_std > 0 else 1
        y_std = y_std if y_std > 0 else 1
        
        # Create scatter plot with contours for each cluster
        for cluster in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster]
            
            # Skip if not enough data points
            if len(cluster_df) < 3:
                continue
            
            # Calculate z-scores
            x = (cluster_df[x_var] - x_mean) / x_std
            y = (cluster_df[y_var] - y_mean) / y_std
            
            # Plot scatter points
            scatter = ax.scatter(x, y, 
                              color=cluster_colors[int(cluster)-1], 
                              alpha=0.7, 
                              s=50, 
                              edgecolor='w',
                              label=f'Subgroup {cluster}')
            
            # Try to create kernel density contours
            try:
                # Create grid for contour
                xmin, xmax = x.min() - 0.5, x.max() + 0.5
                ymin, ymax = y.min() - 0.5, y.max() + 0.5
                
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                
                # Compute kernel density
                kernel = gaussian_kde(np.vstack([x, y]))
                z = np.reshape(kernel(positions).T, xx.shape)
                
                # Plot contours
                contour = ax.contour(xx, yy, z, cmap=cluster_cmaps[int(cluster)-1], alpha=0.3)
                
                # Add cluster label in the center of the distribution
                center_x = x.mean()
                center_y = y.mean()
                ax.text(center_x, center_y, str(cluster), 
                     fontsize=14, fontweight='bold', ha='center', va='center',
                     color='white', 
                     bbox=dict(facecolor=cluster_colors[int(cluster)-1], alpha=0.7))
                
            except Exception as e:
                print(f"Warning: Could not create contours for cluster {cluster}: {e}")
        
        # Add labels and grid
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_title(f"{x_label} vs {y_label}", fontweight='bold', fontsize=13)
        ax.grid(alpha=0.3)
        
        # Set axis limits
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        
        # Add legend to the first subplot only
        if i == 0:
            legend = ax.legend(title="Clusters", loc='upper right', framealpha=0.9)
            legend.get_title().set_fontweight('bold')
    
    # Hide empty subplots if any
    for i in range(len(available_pairs), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.suptitle('Cluster Relationships on Key Clinical Dimensions', fontsize=16, fontweight='bold')
    
    # Save the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved kernel density plots to {output_path}")
    
    plt.close()