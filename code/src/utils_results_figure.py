'''
Utility functions for creating visualizations of cluster comparisons.
'''
# Library imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils_config import OUTPUT_FIGURES_DIR


def create_cluster_distribution_plot(df, output_dir=None):
    """Create a bar plot showing the distribution of subjects across clusters."""
    if output_dir is None:
        output_dir = OUTPUT_FIGURES_DIR / "cluster_comparisons"
        os.makedirs(output_dir, exist_ok=True)
        
    plt.figure(figsize=(10, 6))
    cluster_counts = df['cluster'].value_counts().sort_index()
    ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
    
    # Add count labels on top of each bar
    for i, count in enumerate(cluster_counts.values):
        ax.text(i, count + 0.3, str(count), ha='center')
    
    plt.title('Distribution of Subjects Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Subjects')
    
    output_path = output_dir / "cluster_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cluster distribution plot saved to: {output_path}")


def create_boxplots(df, vars, output_dir=None):
    """Create box plots for key continuous variables across clusters (center = median)."""
    if output_dir is None:
        output_dir = OUTPUT_FIGURES_DIR / "cluster_comparisons"
        os.makedirs(output_dir, exist_ok=True)

    # Define cluster order for consistent plotting
    cluster_order = sorted(df['cluster'].unique())
    
    # Get sample size per cluster
    cluster_counts = df['cluster'].value_counts().sort_index()
    xtick_labels = [f"{c}\n(n={cluster_counts[c]})" for c in cluster_order]
    
    for var in vars:
        if var in df.columns:
            plt.figure(figsize=(12, 6))
            ax = sns.boxplot(
                x='cluster', y=var, data=df, palette='viridis', order=cluster_order,
                showmeans=False  # Seaborn boxplots use median by default
            )
            sns.swarmplot(
                x='cluster', y=var, data=df, color='black', alpha=0.5, size=4, order=cluster_order
            )
            
            ax.set_title(f'Distribution of {var} Across Clusters', fontsize=14)
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel(var, fontsize=12)
            ax.set_xticklabels(xtick_labels, fontsize=11)
            
            output_path = output_dir / f"boxplot_{var}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Box plot for {var} saved to: {output_path}")

def create_categorical_barplots(df, vars, output_dir=None):
    """Create stacked bar plots for categorical variables across clusters."""
    if output_dir is None:
        output_dir = OUTPUT_FIGURES_DIR / "cluster_comparisons"
        os.makedirs(output_dir, exist_ok=True)
        
    for var in vars:
        if var in df.columns:
            plt.figure(figsize=(14, 7))
            
            # Create crosstab and convert to percentage
            crosstab = pd.crosstab(df['cluster'], df[var])
            crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
            
            # Plot
            crosstab_pct.plot(kind='bar', stacked=True, colormap='viridis')
            
            plt.title(f'Distribution of {var} Across Clusters (%)')
            plt.xlabel('Cluster')
            plt.ylabel('Percentage')
            plt.legend(title=var)
            
            output_path = output_dir / f"barplot_{var}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Bar plot for {var} saved to: {output_path}")


def create_radar_chart(df, vars, output_dir=None):
    """Create a radar chart for key clinical measures across clusters."""
    if output_dir is None:
        output_dir = OUTPUT_FIGURES_DIR / "cluster_comparisons"
        os.makedirs(output_dir, exist_ok=True)
        
    # Check if we have all variables
    vars_available = [var for var in vars if var in df.columns]
    
    if len(vars_available) >= 3:  # Need at least 3 variables for a meaningful radar chart
        # Compute mean values for each cluster and variable
        radar_data = df.groupby('cluster')[vars_available].mean()
        
        # Normalize the data for radar chart (0-1 scale)
        radar_data_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
        
        # Number of variables
        N = len(vars_available)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Create angles for each variable
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection
        ax = plt.subplot(111, polar=True)
        
        # Add variable labels
        plt.xticks(angles[:-1], vars_available, size=12)
        
        # Plot each cluster
        for cluster in sorted(df['cluster'].unique()):
            values = radar_data_norm.loc[cluster].values.flatten().tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
            ax.fill(angles, values, alpha=0.1)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Clinical Profile Across Clusters', size=15)
        
        output_path = output_dir / "radar_chart_clinical_profile.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Radar chart saved to: {output_path}")


def setup_visualization_environment():
    """Set up the visualization environment with proper styling and directories."""
    # Set up the aesthetics
    sns.set(style="whitegrid")
    
    # Create a directory for plots
    plots_dir = OUTPUT_FIGURES_DIR / "cluster_comparisons"
    os.makedirs(plots_dir, exist_ok=True)
    
    return plots_dir


def create_visualizations(df):
    """Create a set of visualizations for key variables across clusters."""
    print("\nCreating visualizations...")
    
    # Set up the visualization environment
    plots_dir = setup_visualization_environment()
    
    # Key variables to visualize
    key_continuous_vars = [
        'age', 'fsiq', 'ados_total', 'total_css', 
        'srs_total', 'scq_total', 'ija_success', 'rja_low_success', 'rja_high_success'
    ]
    
    categorical_vars = ['sex', 'sev_ados', 'sev_ados_binary']
    
    radar_vars = ['fsiq', 'ados_total', 'srs_total', 'scq_total', 'ija_success', 'rja_high_success']
    
    # Create cluster distribution plot
    create_cluster_distribution_plot(df, plots_dir)
    
    # Create box plots for continuous variables
    create_boxplots(df, key_continuous_vars, plots_dir)
    
    # Create bar plots for categorical variables
    create_categorical_barplots(df, categorical_vars, plots_dir)
    
    # Create radar chart
    create_radar_chart(df, radar_vars, plots_dir)
    
    print(f"All visualizations saved to: {plots_dir}")