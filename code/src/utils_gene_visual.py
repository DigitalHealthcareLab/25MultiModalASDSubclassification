"""
Visualization utilities for genetic variant cluster analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text
import warnings
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# def create_forest_plot(results_df, pvalue_threshold=0.1, output_path=None):
#     """
#     Create a forest plot from one-vs-all or pairwise comparison results.
    
#     Args:
#         results_df (pd.DataFrame): Results from statistical analysis
#         pvalue_threshold (float): Threshold for statistical significance
#         output_path (str, optional): Path to save figure
        
#     Returns:
#         matplotlib.figure.Figure: The created figure
#     """
#     # Filter for significant results
#     if 'padj' in results_df.columns:
#         sig_df = results_df[results_df['padj'] < pvalue_threshold].copy()
#     else:
#         sig_df = results_df[results_df['pvalue'] < pvalue_threshold].copy()
    
#     # If no significant results, use top 15 results
#     if len(sig_df) == 0:
#         sig_df = results_df.sort_values('pvalue').head(15).copy()
    
#     # Limit to top 20 most significant results
#     if len(sig_df) > 20:
#         sig_df = sig_df.sort_values('padj' if 'padj' in sig_df.columns else 'pvalue').head(20)
    
#     # Sort by log2fc
#     sig_df = sig_df.sort_values('log2fc')
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(10, len(sig_df) * 0.4 + 2))
    
#     # Create category labels
#     if 'cluster1' in sig_df.columns:  # Pairwise comparison
#         labels = [f"C{row['cluster1']} vs C{row['cluster2']} - {row['variant_type']} {row['category']} ({row['impact']})"
#                  for _, row in sig_df.iterrows()]
#     else:  # One-vs-all comparison
#         labels = [f"C{row['cluster']} - {row['variant_type']} {row['category']} ({row['impact']})"
#                 for _, row in sig_df.iterrows()]
    
#     # Y positions
#     y_pos = np.arange(len(labels))
    
#     # Plot points and CIs
#     ax.scatter(sig_df['log2fc'], y_pos, color='blue', s=40, zorder=5)
    
#     # Add error bars if we have standard error info
#     if 'log2fc_se' in sig_df.columns:
#         for i, (_, row) in enumerate(sig_df.iterrows()):
#             ci_low = row['log2fc'] - 1.96 * row['log2fc_se']
#             ci_high = row['log2fc'] + 1.96 * row['log2fc_se']
#             ax.plot([ci_low, ci_high], [y_pos[i], y_pos[i]], color='blue', linewidth=1.5, zorder=4)
    
#     # Add vertical line at log2fc = 0
#     ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, zorder=1)
    
#     # Add gridlines
#     ax.grid(axis='x', linestyle=':', alpha=0.3, zorder=0)
    
#     # Set labels
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(labels)
#     ax.set_xlabel('Log2 Fold Change')
#     ax.set_title('Variant Burden Analysis (Significant Results)', fontsize=14)
    
#     # Add p-value annotations
#     p_col = 'padj' if 'padj' in sig_df.columns else 'pvalue'
#     for i, (_, row) in enumerate(sig_df.iterrows()):
#         # Add asterisks for significance
#         if row[p_col] < 0.01:
#             sig_str = '**'
#         elif row[p_col] < 0.05:
#             sig_str = '*'
#         elif row[p_col] < 0.1:
#             sig_str = '†'
#         else:
#             sig_str = ''
            
#         # Position based on log2fc direction
#         x_pos = row['log2fc'] + (0.1 if row['log2fc'] >= 0 else -0.1)
#         ha = 'left' if row['log2fc'] >= 0 else 'right'
        
#         # Add text
#         ax.text(x_pos, y_pos[i], f" {sig_str} p={row[p_col]:.2e}", va='center', ha=ha, fontsize=8)
    
#     # Add legend for significance
#     legend_text = '** p < 0.01, * p < 0.05, † p < 0.1'
#     ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=8, va='bottom')
    
#     plt.tight_layout()
    
#     # Save if requested
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"Saved forest plot to {output_path}")
    
#     return fig

def create_burden_heatmap(burden_df, value_col='variant_rate', output_path=None):
    """
    Create a heatmap of variant burden across clusters.
    
    Args:
        burden_df (pd.DataFrame): Variant burden summary
        value_col (str): Column to use for heatmap values
        output_path (str, optional): Path to save figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Reshape data for heatmap
    pivot_data = burden_df.pivot_table(
        index=['variant_type', 'category', 'impact'],
        columns='cluster',
        values=value_col
    )
    
    # Sort rows by variant type, category, and impact
    # Define custom sort orders
    var_type_order = {'rare_coding': 0, 'common_coding': 1, 'rare_noncoding': 2}
    category_order = {'PTV': 0, 'Missense': 1, 'Synonymous': 2, 'Promoter': 3, 'Enhancer': 4, 
                     'UTR': 5, 'Intronic': 6, 'Intergenic': 7, 'Other': 8}
    impact_order = {'High': 0, 'Moderate': 1, 'Low': 2}
    
    # Create sort keys
    pivot_data = pivot_data.reset_index()
    pivot_data['var_type_order'] = pivot_data['variant_type'].map(var_type_order)
    pivot_data['category_order'] = pivot_data['category'].map(category_order)
    pivot_data['impact_order'] = pivot_data['impact'].map(impact_order)
    
    # Sort and prepare for heatmap
    pivot_data = pivot_data.sort_values(['var_type_order', 'category_order', 'impact_order'])
    
    # Create row labels
    row_labels = [f"{row['variant_type']} - {row['category']} ({row['impact']})" 
                 for _, row in pivot_data.iterrows()]
    
    # Extract data for heatmap
    clusters = sorted([col for col in pivot_data.columns if isinstance(col, (int, float))])
    heatmap_data = pivot_data[clusters].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(len(clusters) + 2, len(row_labels)*0.5 + 2))
    
    # Create heatmap
    # Use row normalization for better visualization
    normalized_data = np.zeros_like(heatmap_data)
    for i in range(heatmap_data.shape[0]):
        row = heatmap_data[i, :]
        row_max = np.max(row)
        if row_max > 0:
            normalized_data[i, :] = row / row_max
    
    # Create custom diverging colormap
    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap', ['#ffffff', '#abd9e9', '#2c7bb6', '#000000'], N=256
    )
    
    # Plot heatmap
    im = ax.imshow(normalized_data, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Variant Rate')
    
    # Add text annotations with actual values
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data[i, j]
            color = 'white' if normalized_data[i, j] > 0.7 else 'black'
            ax.text(j, i, f"{value:.2f}", ha='center', va='center', color=color, fontsize=8)
    
    # Add labels
    ax.set_xticks(np.arange(len(clusters)))
    ax.set_xticklabels([f"Cluster {c}" for c in clusters])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    
    # Add title
    ax.set_title(f'Variant Burden Across Clusters ({value_col})', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved burden heatmap to {output_path}")
    
    return fig

def plot_gene_enrichment(enriched_genes_df, top_n=20, output_path=None):
    """
    Plot top enriched genes for a cluster.
    
    Args:
        enriched_genes_df (pd.DataFrame): Enriched genes dataframe
        top_n (int): Number of top genes to plot
        output_path (str, optional): Path to save figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Filter for top genes
    if len(enriched_genes_df) > top_n:
        top_genes = enriched_genes_df.head(top_n).copy()
    else:
        top_genes = enriched_genes_df.copy()
    
    # Sort by enrichment
    top_genes = top_genes.sort_values('enrichment')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, len(top_genes)*0.4 + 2))
    
    # Create bar plot
    bars = ax.barh(top_genes['gene'], top_genes['log2fc'], color='royalblue')
    
    # Add p-value markers
    for i, (_, row) in enumerate(top_genes.iterrows()):
        if row['pvalue'] < 0.01:
            marker = '**'
        elif row['pvalue'] < 0.05:
            marker = '*'
        elif row['pvalue'] < 0.1:
            marker = '†'
        else:
            marker = ''
        
        if marker:
            ax.text(row['log2fc'] + 0.1, i, marker, va='center', fontsize=12)
    
    # Add reference line
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    
    # Add labels
    ax.set_xlabel('Log2 Fold Enrichment')
    ax.set_title(f'Top Enriched Genes (Cluster {top_genes["cluster"].iloc[0]})', fontsize=14)
    
    # Add legend for significance
    legend_text = '** p < 0.01, * p < 0.05, † p < 0.1'
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=8, va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved gene enrichment plot to {output_path}")
    
    return fig

def create_burden_boxplots(long_df, category='PTV', impact='High', output_path=None):
    """
    Create boxplots showing the distribution of variant burden across participants within each cluster.
    
    Args:
        long_df (pd.DataFrame): Long-format variant data
        category (str): Variant category to plot
        impact (str): Impact level to plot
        output_path (str, optional): Path to save figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Filter data
    filtered_df = long_df[
        (long_df['category'] == category) &
        (long_df['impact_score_bin'] == impact) &
        (long_df['gene_name'] == '--')  # Get aggregate counts
    ].copy()
    
    if filtered_df.empty:
        print(f"No data found for {category} variants with {impact} impact")
        return None
    
    # Summarize variant counts by participant and cluster
    summary = filtered_df.groupby(['participant_id', 'cluster', 'variant_type'])['count'].sum().reset_index()
    
    # Create figure
    variant_types = summary['variant_type'].unique()
    
    fig, axes = plt.subplots(1, len(variant_types), figsize=(5*len(variant_types), 6), sharey=True)
    if len(variant_types) == 1:
        axes = [axes]
    
    # Plot each variant type
    for i, var_type in enumerate(variant_types):
        ax = axes[i]
        data = summary[summary['variant_type'] == var_type]
        
        # Create boxplot
        sns.boxplot(x='cluster', y='count', data=data, ax=ax, color='lightblue')
        
        # Add strip plot for individual points
        sns.stripplot(x='cluster', y='count', data=data, ax=ax, color='navy', alpha=0.5, jitter=True)
        
        # Add labels
        ax.set_title(f'{var_type}')
        ax.set_xlabel('Cluster')
        if i == 0:
            ax.set_ylabel(f'Count per participant')
        else:
            ax.set_ylabel('')
    
    # Add overall title
    plt.suptitle(f'{category} Variants ({impact} Impact) Across Clusters', fontsize=14)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved burden boxplots to {output_path}")
    
    return fig

def create_summary_table(results_df, output_path=None):
    """
    Create a formatted summary table of significant results.
    
    Args:
        results_df (pd.DataFrame): Results from statistical analysis
        output_path (str, optional): Path to save table
        
    Returns:
        pd.DataFrame: Formatted table for publication
    """
    # Filter for significant or top results
    if 'padj' in results_df.columns:
        sig_df = results_df[results_df['padj'] < 0.1].copy()
    else:
        sig_df = results_df[results_df['pvalue'] < 0.1].copy()
    
    # If no significant results, use top 15 results
    if len(sig_df) == 0:
        sig_df = results_df.sort_values('pvalue').head(15).copy()
    
    # Format for publication
    if 'cluster1' in sig_df.columns:  # Pairwise comparison
        formatted = sig_df[['cluster1', 'cluster2', 'variant_type', 'category', 'impact', 
                           'cluster1_rate', 'cluster2_rate', 'log2fc', 'pvalue', 'padj']].copy()
        
        # Rename columns
        formatted.columns = ['Cluster 1', 'Cluster 2', 'Variant Type', 'Category', 'Impact',
                            'Rate 1', 'Rate 2', 'Log2FC', 'P-value', 'Adj. P-value']
    else:  # One-vs-all comparison
        formatted = sig_df[['cluster', 'variant_type', 'category', 'impact', 
                           'cluster_rate', 'other_rate', 'log2fc', 'pvalue', 'padj']].copy()
        
        # Rename columns
        formatted.columns = ['Cluster', 'Variant Type', 'Category', 'Impact',
                            'Cluster Rate', 'Other Rate', 'Log2FC', 'P-value', 'Adj. P-value']
    
    # Format numeric columns
    for col in formatted.columns:
        if col in ['Log2FC', 'Rate 1', 'Rate 2', 'Cluster Rate', 'Other Rate']:
            formatted[col] = formatted[col].map(lambda x: f"{x:.2f}")
        elif col in ['P-value', 'Adj. P-value']:
            formatted[col] = formatted[col].map(lambda x: f"{x:.2e}")
    
    # Save if requested
    if output_path:
        formatted.to_csv(output_path, index=False)
        print(f"Saved summary table to {output_path}")
    
    return formatted


def visualize_rare_common_ratios(ratio_df, title, output_path):
    """
    Create a visualization of rare/common ratios across clusters.
    
    Args:
        ratio_df (pd.DataFrame): DataFrame with rare/common ratios
        title (str): Plot title
        output_path (str or Path): Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Create boxplot with individual points
    ax = sns.boxplot(x='cluster', y='ratio', data=ratio_df, palette='viridis')
    sns.stripplot(x='cluster', y='ratio', data=ratio_df, color='black', alpha=0.5, jitter=True)
    
    # Set title and labels
    plt.title(title, fontsize=14)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Rare/Common Variant Ratio', fontsize=12)
    
    # Add cluster medians as text - CHANGED FROM MEAN TO MEDIAN
    for i, cluster in enumerate(sorted(ratio_df['cluster'].unique())):
        cluster_median = ratio_df[ratio_df['cluster'] == cluster]['ratio'].median()
        ax.text(i, 0.1, f'Median: {cluster_median:.2f}', ha='center', fontsize=10)
    
    # Add annotation about statistical test
    stat_tests = []
    cluster_count = len(ratio_df['cluster'].unique())
    
    if cluster_count > 2:
        stat_tests.append("Kruskal-Wallis test across all clusters")
    
    if cluster_count > 1:
        stat_tests.append("Pairwise Mann-Whitney U tests")
    
    plt.figtext(0.5, 0.01, 
               f"Statistical tests: {', '.join(stat_tests)}", 
               ha='center', fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


    # Draw with mean
    # def visualize_rare_common_ratios(ratio_df, title, output_path):
    #     """
    #     Create a visualization of rare/common ratios across clusters.
        
    #     Args:
    #         ratio_df (pd.DataFrame): DataFrame with rare/common ratios
    #         title (str): Plot title
    #         output_path (str or Path): Path to save the figure
    #     """
    #     plt.figure(figsize=(10, 6))
        
    #     # Create boxplot with individual points
    #     ax = sns.boxplot(x='cluster', y='ratio', data=ratio_df, palette='viridis')
    #     sns.stripplot(x='cluster', y='ratio', data=ratio_df, color='black', alpha=0.5, jitter=True)
        
    #     # Set title and labels
    #     plt.title(title, fontsize=14)
    #     plt.xlabel('Cluster', fontsize=12)
    #     plt.ylabel('Rare/Common Variant Ratio', fontsize=12)
        
    #     # Add cluster means as text
    #     for i, cluster in enumerate(sorted(ratio_df['cluster'].unique())):
    #         cluster_mean = ratio_df[ratio_df['cluster'] == cluster]['ratio'].mean()
    #         ax.text(i, 0.1, f'Mean: {cluster_mean:.2f}', ha='center', fontsize=10)
        
    #     # Add annotation about statistical test
    #     stat_tests = []
    #     cluster_count = len(ratio_df['cluster'].unique())
        
    #     if cluster_count > 2:
    #         stat_tests.append("Kruskal-Wallis test across all clusters")
        
    #     if cluster_count > 1:
    #         stat_tests.append("Pairwise Mann-Whitney U tests")
        
    #     plt.figtext(0.5, 0.01, 
    #             f"Statistical tests: {', '.join(stat_tests)}", 
    #             ha='center', fontsize=10)
        
    #     # Save figure
    #     plt.tight_layout()
    #     plt.savefig(output_path, dpi=300)
    #     plt.close()


"""
Visualization utilities for genetic variant analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Union, Optional, Any

# Set style for consistent visualizations
plt.style.use('seaborn-v0_8-whitegrid')

# Or a more future-proof alternative:
try:
    plt.style.use('seaborn-whitegrid')  # For older versions
except:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')  # For newer versions
    except:
        plt.style.use('default')  # Fallback
sns.set_context("paper", font_scale=1.2)

# Define color scheme
COLORS = {
    'PTV': '#1F77B4',  # Blue
    'Missense': '#FF7F0E',  # Orange
    'Synonymous': '#2CA02C',  # Green
    'Regulatory': '#D62728',  # Red
    'Other': '#9467BD',  # Purple
    
    # For cluster-specific colors
    'Cluster1': '#1F77B4',  # Blue
    'Cluster2': '#FF7F0E',  # Orange
    'Cluster3': '#2CA02C',  # Green
    'Cluster4': '#D62728',  # Red
    
    # For impact levels
    'High_impact': '#D62728',  # Red
    'Low_impact': '#9467BD',  # Purple
    
    # For pLI bins
    'High_high_pLI': '#D62728',  # Red
    'High_low_pLI': '#FF7F0E',  # Orange
    'Moderate_high_pLI': '#9467BD',  # Purple
    'Moderate_low_pLI': '#8C564B',  # Brown
    'Low_high_pLI': '#E377C2',  # Pink
    'Low_low_pLI': '#7F7F7F',  # Gray
}

def create_forest_plot(results_df: pd.DataFrame, 
                      output_path: str,
                      title: str = "Variant Burden Analysis",
                      x_label: str = "Log2 Fold Change",
                      significant_col: str = 'adjusted_pvalue',
                      fig_size: Tuple[int, int] = (12, 10)) -> None:
    """
    Create a forest plot similar to Huang et al. figure.
    
    Args:
        results_df (pd.DataFrame): Results from hierarchical analysis
        output_path (str): Path to save the figure
        title (str): Plot title
        x_label (str): X-axis label
        significant_col (str): Column to use for significance (e.g., 'adjusted_pvalue')
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Filter for pairwise tests and handle potential missing columns
    if 'test' in results_df.columns:
        plot_df = results_df[results_df['test'] == 'pairwise'].copy()
    else:
        plot_df = results_df.copy()
    
    # Check for required columns
    required_cols = ['hierarchy', 'category', 'impact_bin', 'log2fc', significant_col]
    
    for col in required_cols:
        if col not in plot_df.columns:
            print(f"Warning: Column '{col}' not found in data. Forest plot creation skipped.")
            return
    
    # Create a more informative label for the y-axis
    plot_df['plot_label'] = plot_df.apply(
        lambda row: f"{row['category']} - {row['impact_bin']}",
        axis=1
    )
    
    # Determine significance based on the specified column
    plot_df['significant'] = plot_df[significant_col] < 0.05
    
    # Add P-value annotation
    plot_df['p_annotation'] = plot_df[significant_col].apply(
        lambda p: f"P = {p:.1e}" if p < 0.05 else "n.s."
    )
    
    # Sort by hierarchy, then by impact_bin
    hierarchy_order = ['PTV', 'Missense (CADD>30)', 'Missense (CADD20-30)', 'Missense (CADD0-20)', 'Synonymous', 'Regulatory', 'Other Non-coding']
    
    # Create a numeric sort key based on hierarchy order
    plot_df['hierarchy_sort'] = plot_df['hierarchy'].apply(
        lambda h: hierarchy_order.index(h) if h in hierarchy_order else 999
    )
    
    plot_df = plot_df.sort_values(['hierarchy_sort', 'impact_bin'])
    
    # Start plotting
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Determine point colors based on significance
    colors = plot_df['significant'].apply(
        lambda s: 'darkblue' if s else 'gray'
    )
    
    # Determine marker shape based on category
    markers = plot_df['category'].apply(
        lambda c: 'o' if c in ['PTV', 'Missense'] else 's'
    )
    
    # Determine point size based on effect size
    if 'effect_size' in plot_df.columns:
        sizes = plot_df['effect_size'].abs().apply(
            lambda e: 50 + 100 * (e if not np.isnan(e) else 0)
        )
    else:
        sizes = 100  # Default size
    
    # Plot the data points
    for i, row in plot_df.iterrows():
        # Determine marker and color
        marker = 'o' if row['category'] in ['PTV', 'Missense'] else 's'
        color = 'darkblue' if row['significant'] else 'gray'
        
        # Plot point
        ax.scatter(
            row['log2fc'], 
            row['plot_label'],
            color=color,
            s=sizes[i] if isinstance(sizes, pd.Series) else sizes,
            marker=marker,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add confidence interval if available
        if 'ci_lower' in plot_df.columns and 'ci_upper' in plot_df.columns:
            ax.plot(
                [row['ci_lower'], row['ci_upper']], 
                [row['plot_label'], row['plot_label']],
                color=color,
                linewidth=2 if row['significant'] else 1,
                alpha=0.7
            )
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add P-value annotations
    for i, row in plot_df.iterrows():
        ax.text(
            x=max(plot_df['log2fc']) + 0.1,
            y=row['plot_label'],
            s=row['p_annotation'],
            ha='left',
            va='center',
            fontsize=9,
            alpha=0.8
        )
    
    # Add number of genes if available
    if 'cluster_genes' in plot_df.columns:
        for i, row in plot_df.iterrows():
            ax.text(
                x=min(plot_df['log2fc']) - 0.5,
                y=row['plot_label'],
                s=f"n={row['cluster_genes']}",
                ha='right',
                va='center',
                fontsize=9,
                alpha=0.8
            )
    
    # Customize the plot
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("")  # No y-label needed with the custom labels
    ax.set_title(title, fontsize=14)
    
    # Add a light grid
    ax.grid(axis='x', alpha=0.3)
    
    # Ensure the x-axis is centered around 0 with some padding
    max_abs_x = max(abs(plot_df['log2fc'].max()), abs(plot_df['log2fc'].min()))
    ax.set_xlim(-max_abs_x * 1.2, max_abs_x * 1.2)
    
    # Add a legend for marker shapes and significance
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
               markersize=10, label='Coding - Significant'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Coding - Non-significant'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='darkblue', 
               markersize=10, label='Non-coding - Significant'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, label='Non-coding - Non-significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add text explaining effect direction
    plt.figtext(0.15, 0.01, "← Depleted in cluster", ha='center', fontsize=10)
    plt.figtext(0.85, 0.01, "Enriched in cluster →", ha='center', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_boxplots_by_cluster(data_df: pd.DataFrame,
                              output_path: str,
                              title: str = "Variant Counts by Cluster",
                              y_label: str = "Count",
                              hue_col: str = None,
                              fig_size: Tuple[int, int] = (10, 6)) -> None:
    """
    Create boxplots showing the distribution of counts across clusters.
    
    Args:
        data_df (pd.DataFrame): DataFrame with count data
        output_path (str): Path to save the figure
        title (str): Plot title
        y_label (str): Y-axis label
        hue_col (str): Column to use for coloring (e.g., 'impact_bin')
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Check for required columns
    required_cols = ['cluster', 'count']
    for col in required_cols:
        if col not in data_df.columns:
            print(f"Warning: Column '{col}' not found in data. Boxplot creation skipped.")
            return
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create boxplot
    if hue_col and hue_col in data_df.columns:
        # Sort hue categories if possible
        # Create boxplot with hue
        sns.boxplot(
            x='cluster',
            y='count',
            hue=hue_col,
            data=data_df,
            ax=ax
        )
        
        # Adjust legend
        ax.legend(title=hue_col.replace('_', ' ').title(), fontsize=10)
    else:
        # Create simple boxplot
        sns.boxplot(
            x='cluster',
            y='count',
            data=data_df,
            ax=ax
        )
    
    # Add individual data points
    sns.stripplot(
        x='cluster',
        y='count',
        data=data_df,
        size=5,
        color='black',
        alpha=0.5,
        jitter=True,
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add a light grid
    ax.grid(axis='y', alpha=0.3)
    
    # Add cluster sizes if available
    if 'participants' in data_df.columns:
        cluster_sizes = data_df.groupby('cluster')['participants'].first()
        
        for i, cluster in enumerate(sorted(data_df['cluster'].unique())):
            ax.text(
                i, 
                ax.get_ylim()[0] * 1.1,
                f"n={cluster_sizes.get(cluster, '?')}",
                ha='center',
                fontsize=9
            )
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_heatmap_across_categories(results_df: pd.DataFrame,
                                   output_path: str,
                                   title: str = "Variant Burden Across Categories",
                                   significant_col: str = 'adjusted_pvalue',
                                   metric: str = 'log2fc',
                                   fig_size: Tuple[int, int] = (14, 10)) -> None:
    """
    Create a heatmap showing variant burden across all categories and clusters.
    
    Args:
        results_df (pd.DataFrame): Results from hierarchical analysis
        output_path (str): Path to save the figure
        title (str): Plot title
        significant_col (str): Column to use for significance (e.g., 'adjusted_pvalue')
        metric (str): Column to use for heatmap values (e.g., 'log2fc')
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Filter for pairwise tests if test column exists
    if 'test' in results_df.columns:
        plot_df = results_df[results_df['test'] == 'pairwise'].copy()
    else:
        plot_df = results_df.copy()
    
    # Check for required columns
    required_cols = ['hierarchy', 'category', 'impact_bin' if 'impact_bin' in plot_df.columns else 'impact', 
                     metric, significant_col, 'cluster1', 'cluster2']
    
    for col in required_cols:
        if col not in plot_df.columns:
            print(f"Warning: Column '{col}' not found in data. Heatmap creation skipped.")
            return
    
    # Ensure we have a consistent impact column
    impact_col = 'impact_bin' if 'impact_bin' in plot_df.columns else 'impact'
    
    # Create a more informative label for the categories
    plot_df['plot_label'] = plot_df.apply(
        lambda row: f"{row['hierarchy']}\n{row['category']} - {row[impact_col]}",
        axis=1
    )
    
    # Determine significance based on the specified column
    plot_df['significant'] = plot_df[significant_col] < 0.05
    
    # Create a pivot table for the heatmap
    if 'cluster1' in plot_df.columns and 'cluster2' in plot_df.columns:
        # For pairwise comparisons, create a multi-level index
        pivot_df = plot_df.pivot_table(
            index='plot_label',
            columns=['cluster1', 'cluster2'],
            values=metric,
            aggfunc='first'
        )
    else:
        # For one-vs-all comparisons
        pivot_df = plot_df.pivot_table(
            index='plot_label',
            columns='cluster',
            values=metric,
            aggfunc='first'
        )
    
    # Create a significance mask
    if 'cluster1' in plot_df.columns and 'cluster2' in plot_df.columns:
        sig_pivot = plot_df.pivot_table(
            index='plot_label',
            columns=['cluster1', 'cluster2'],
            values='significant',
            aggfunc='first'
        )
    else:
        sig_pivot = plot_df.pivot_table(
            index='plot_label',
            columns='cluster',
            values='significant',
            aggfunc='first'
        )
    
    # Convert to boolean mask where True means NOT significant (will be covered with hatches)
    sig_mask = ~sig_pivot
    
    # Sort by hierarchy and impact
    hierarchy_order = ['PTV', 'Missense (CADD>30)', 'Missense (CADD20-30)', 
                       'Missense (CADD0-20)', 'Synonymous', 'Regulatory', 'Other Non-coding']
    
    # Reindex the pivot table based on the hierarchy in the original data
    hierarchy_map = plot_df.set_index('plot_label')['hierarchy'].to_dict()
    
    # Create a sort key for the index
    sort_key = []
    for label in pivot_df.index:
        hierarchy = hierarchy_map.get(label, '')
        sort_key.append((
            hierarchy_order.index(hierarchy) if hierarchy in hierarchy_order else 999,
            label
        ))
    
    # Sort the pivot table
    sorted_index = [label for _, label in sorted(sort_key)]
    pivot_df = pivot_df.reindex(sorted_index)
    sig_mask = sig_mask.reindex(sorted_index)
    
    # Create a colormap for log2fc
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        mask=sig_mask.isna(),  # Don't show cells with no data
        cbar_kws={"label": metric},
        ax=ax
    )
    
    # Add significance markers using hatching
    for i, idx in enumerate(pivot_df.index):
        for j, col in enumerate(pivot_df.columns):
            if not sig_mask.iloc[i, j]:  # If significant
                # Add a star or border to significant cells
                ax.add_patch(plt.Rectangle(
                    (j, i), 
                    1, 
                    1, 
                    fill=False, 
                    edgecolor='black', 
                    lw=2
                ))
    
    # Customize the plot
    ax.set_title(title, fontsize=14)
    
    # Fix x-axis labels for pairwise comparisons
    if isinstance(pivot_df.columns, pd.MultiIndex):
        ax.set_xticklabels([f"{c1} vs {c2}" for c1, c2 in pivot_df.columns], rotation=45, ha='right')
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_summary_barplots(coding_results: pd.DataFrame,
                          noncoding_results: Optional[pd.DataFrame] = None,
                          output_path: str = None,
                          title: str = "Variant Burden Summary",
                          use_proportions: bool = False,
                          fig_size: Tuple[int, int] = (12, 8)) -> None:
    """
    Create summary barplots for variant burden across clusters.
    
    Args:
        coding_results (pd.DataFrame): Coding variant results
        noncoding_results (pd.DataFrame): Non-coding variant results (optional)
        output_path (str): Path to save the figure
        title (str): Plot title
        use_proportions (bool): Whether to plot proportions instead of counts
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Handle the case where only coding results are provided
    if use_proportions and 'proportion' not in coding_results.columns:
        print("Warning: 'proportion' column not found but use_proportions=True. Switching to counts.")
        use_proportions = False
    
    # Prepare the data for plotting
    if 'category' in coding_results.columns:
        # For the functional impact distribution analysis
        value_col = 'proportion' if use_proportions else 'count'
        group_cols = ['cluster', 'category']
        
        if value_col not in coding_results.columns:
            print(f"Warning: Column '{value_col}' not found in data. Barplot creation skipped.")
            return
        
        # Prepare the plot data
        plot_data = coding_results.copy()
        
        # Create figure with subplots for each cluster
        clusters = sorted(plot_data['cluster'].unique())
        n_clusters = len(clusters)
        
        # Determine grid layout
        n_cols = min(2, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True, sharey=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Create a barplot for each cluster
        for i, cluster in enumerate(clusters):
            cluster_data = plot_data[plot_data['cluster'] == cluster]
            
            # Sort by category
            category_order = ['PTV', 'Missense', 'Synonymous']
            cluster_data['category_order'] = cluster_data['category'].apply(
                lambda c: category_order.index(c) if c in category_order else 999
            )
            cluster_data = cluster_data.sort_values('category_order')
            
            # Create the barplot
            sns.barplot(
                x='category',
                y=value_col,
                data=cluster_data,
                palette={
                    cat: COLORS.get(cat, f"C{i}") 
                    for i, cat in enumerate(cluster_data['category'].unique())
                },
                ax=axes[i]
            )
            
            # Customize subplot
            axes[i].set_title(f"Cluster {cluster}")
            axes[i].set_xlabel("")
            
            if i % n_cols == 0:  # Only add y-label to leftmost plots
                axes[i].set_ylabel("Proportion" if use_proportions else "Count")
            else:
                axes[i].set_ylabel("")
            
            # Add value labels on bars
            for j, p in enumerate(axes[i].patches):
                value = p.get_height()
                axes[i].annotate(
                    f"{value:.2f}" if use_proportions else f"{int(value)}",
                    (p.get_x() + p.get_width() / 2., value),
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    rotation=0
                )
        
        # Hide any unused subplots
        for i in range(n_clusters, len(axes)):
            axes[i].set_visible(False)
    
    else:
        # For the hierarchical analysis results
        # Filter for one-vs-all comparison results if possible
        if 'cluster' in coding_results.columns and 'other_clusters' in coding_results.columns:
            coding_plot = coding_results.copy()
        elif 'cluster1' in coding_results.columns and 'cluster2' in coding_results.columns:
            # For pairwise results, we'll use cluster1 for grouping
            coding_plot = coding_results.rename(columns={'cluster1': 'cluster'}).copy()
        else:
            print("Warning: Required columns not found in data. Barplot creation skipped.")
            return
        
        # Include non-coding results if provided
        if noncoding_results is not None:
            if 'cluster' in noncoding_results.columns and 'other_clusters' in noncoding_results.columns:
                noncoding_plot = noncoding_results.copy()
            elif 'cluster1' in noncoding_results.columns and 'cluster2' in noncoding_results.columns:
                noncoding_plot = noncoding_results.rename(columns={'cluster1': 'cluster'}).copy()
            else:
                noncoding_plot = None
                
            if noncoding_plot is not None:
                # Combine coding and non-coding data
                coding_plot['domain'] = 'Coding'
                noncoding_plot['domain'] = 'Non-coding'
                plot_data = pd.concat([coding_plot, noncoding_plot])
            else:
                plot_data = coding_plot
                plot_data['domain'] = 'Coding'
        else:
            plot_data = coding_plot
            plot_data['domain'] = 'Coding'
        
        # Get the metric to plot - prioritize log2fc if available
        if 'log2fc' in plot_data.columns:
            metric = 'log2fc'
        elif 'effect_size' in plot_data.columns:
            metric = 'effect_size'
        else:
            print("Warning: No suitable metric found for barplot. Creation skipped.")
            return
        
        # Create figure with subplots for each hierarchy level
        hierarchies = sorted(plot_data['hierarchy'].unique())
        n_hierarchies = len(hierarchies)
        
        # Determine grid layout
        n_cols = min(2, n_hierarchies)
        n_rows = (n_hierarchies + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Create a barplot for each hierarchy level
        for i, hierarchy in enumerate(hierarchies):
            hierarchy_data = plot_data[plot_data['hierarchy'] == hierarchy]
            
            # Group by cluster and domain
            grouped = hierarchy_data.groupby(['cluster', 'domain'])[metric].mean().reset_index()
            
            # Create the barplot
            sns.barplot(
                x='cluster',
                y=metric,
                hue='domain',
                data=grouped,
                palette={
                    'Coding': COLORS.get('PTV', 'blue'),
                    'Non-coding': COLORS.get('Regulatory', 'red')
                },
                ax=axes[i]
            )
            
            # Add a line at y=0
            axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            # Customize subplot
            axes[i].set_title(hierarchy)
            axes[i].set_xlabel("Cluster")
            
            if i % n_cols == 0:  # Only add y-label to leftmost plots
                axes[i].set_ylabel(metric)
            else:
                axes[i].set_ylabel("")
            
            # Add legend for the first subplot only
            if i == 0:
                axes[i].legend(title="Domain")
            else:
                axes[i].get_legend().remove()
        
        # Hide any unused subplots
        for i in range(n_hierarchies, len(axes)):
            axes[i].set_visible(False)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make space for the suptitle
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig, axes
    
    # For the first case (category in coding_results)
    # These settings apply outside the else block
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make space for the suptitle
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, axes

def plot_cluster_enrichment_improved(results_df: pd.DataFrame,
                                output_path: str,
                                title: str = "Cluster-specific Variant Enrichment",
                                x_col: str = 'log2fc',
                                y_col: str = 'hierarchy',
                                stratify_col: str = 'impact_bin',
                                hue_col: str = 'cluster',
                                significant_col: str = 'adjusted_pvalue',
                                fig_size: Tuple[int, int] = (14, 12)) -> None:
    """
    Create an improved cluster-specific enrichment plot with explicit stratification.
    
    Args:
        results_df (pd.DataFrame): Results from analysis
        output_path (str): Path to save the figure
        title (str): Plot title
        x_col (str): Column to use for x-axis
        y_col (str): Column to use for primary y-axis categories
        stratify_col (str): Column to use for stratification within categories
        hue_col (str): Column to use for coloring points
        significant_col (str): Column to use for significance
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Check for required columns
    required_cols = [x_col, y_col, hue_col]
    if stratify_col:
        required_cols.append(stratify_col)
        
    for col in required_cols:
        if col not in results_df.columns:
            print(f"Warning: Column '{col}' not found in data. Enrichment plot creation skipped.")
            return
    
    # Create a more informative label for the y-axis by combining hierarchy and stratification
    if stratify_col:
        # Map pLI bin values to more readable labels
        pli_mapping = {
            'High_high_pLI': 'High pLI (≥0.9)',
            'High_low_pLI': 'Low pLI (<0.9)',
            'Moderate_high_pLI': 'High pLI (≥0.9)',
            'Moderate_low_pLI': 'Low pLI (<0.9)',
            'Low_high_pLI': 'High pLI (≥0.9)',
            'Low_low_pLI': 'Low pLI (<0.9)'
        }
        
        # Create a new combined label
        results_df['plot_label'] = results_df.apply(
            lambda row: f"{row[y_col]} → {pli_mapping.get(row[stratify_col], row[stratify_col])}",
            axis=1
        )
    else:
        results_df['plot_label'] = results_df[y_col]
    
    # Determine significance
    if significant_col in results_df.columns:
        results_df['significant'] = results_df[significant_col] < 0.05
    else:
        results_df['significant'] = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Get all unique clusters
    clusters = sorted(results_df[hue_col].unique())
    
    # Create color mapping for clusters
    cluster_colors = {
        cluster: COLORS.get(f'Cluster{cluster}', plt.cm.tab10(i/len(clusters))) 
        for i, cluster in enumerate(clusters)
    }
    
    # Create a plot showing each hierarchy level with stratification
    for cluster in clusters:
        cluster_data = results_df[results_df[hue_col] == cluster]
        
        # Plot non-significant points first (as background)
        non_sig = cluster_data[~cluster_data['significant']]
        sig = cluster_data[cluster_data['significant']]
        
        color = cluster_colors[cluster]
        
        # Non-significant points (more transparent)
        if not non_sig.empty:
            ax.scatter(
                non_sig[x_col],
                non_sig['plot_label'],
                s=80,
                color=color,
                alpha=0.3,
                marker='o',
                label=f'Cluster {cluster} (n.s.)' if sig.empty else None
            )
        
        # Significant points
        if not sig.empty:
            ax.scatter(
                sig[x_col],
                sig['plot_label'],
                s=100,
                color=color,
                alpha=0.8,
                edgecolor='black',
                linewidth=1,
                marker='o',
                label=f'Cluster {cluster}'
            )
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Customize the plot
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel("Variant Category", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add a light grid
    ax.grid(alpha=0.3)
    
    # Add a legend
    ax.legend(title=hue_col.replace('_', ' ').title(), fontsize=10)
    
    # Add text explaining effect direction if using log2fc
    if x_col == 'log2fc':
        plt.figtext(0.15, 0.01, "← Depleted", ha='center', fontsize=10)
        plt.figtext(0.85, 0.01, "Enriched →", ha='center', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_cluster_enrichment_faceted(results_df: pd.DataFrame,
                                  output_path: str,
                                  title: str = "Cluster-specific Variant Enrichment",
                                  x_col: str = 'log2fc',
                                  y_col: str = 'hierarchy',
                                  stratify_col: str = 'impact_bin',
                                  hue_col: str = 'cluster',
                                  significant_col: str = 'adjusted_pvalue',
                                  fig_size: Tuple[int, int] = (16, 10)) -> None:
    """
    Create a faceted cluster-specific enrichment plot with one panel per cluster.
    
    Args:
        results_df (pd.DataFrame): Results from analysis
        output_path (str): Path to save the figure
        title (str): Plot title
        x_col (str): Column to use for x-axis
        y_col (str): Column to use for primary y-axis categories
        stratify_col (str): Column to use for stratification within categories
        hue_col (str): Column to use for coloring points
        significant_col (str): Column to use for significance
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Check for required columns
    required_cols = [x_col, y_col, hue_col]
    if stratify_col:
        required_cols.append(stratify_col)
        
    for col in required_cols:
        if col not in results_df.columns:
            print(f"Warning: Column '{col}' not found in data. Enrichment plot creation skipped.")
            return
    
    # Create a more informative label for the y-axis by combining hierarchy and stratification
    if stratify_col:
        # Map pLI bin values to more readable labels
        pli_mapping = {
            'High_high_pLI': 'High pLI (≥0.9)',
            'High_low_pLI': 'Low pLI (<0.9)',
            'Moderate_high_pLI': 'High pLI (≥0.9)',
            'Moderate_low_pLI': 'Low pLI (<0.9)',
            'Low_high_pLI': 'High pLI (≥0.9)',
            'Low_low_pLI': 'Low pLI (<0.9)'
        }
        
        # Create a new combined label
        results_df['plot_label'] = results_df.apply(
            lambda row: f"{row[y_col]} → {pli_mapping.get(row[stratify_col], row[stratify_col])}",
            axis=1
        )
    else:
        results_df['plot_label'] = results_df[y_col]
    
    # Determine significance
    if significant_col in results_df.columns:
        results_df['significant'] = results_df[significant_col] < 0.05
    else:
        results_df['significant'] = False
    
    # Get all unique clusters
    clusters = sorted(results_df[hue_col].unique())
    
    # Create color mapping for stratification
    if stratify_col:
        unique_strats = results_df[stratify_col].unique()
        strat_colors = {}
        
        for i, strat in enumerate(unique_strats):
            if 'high_pLI' in str(strat):
                strat_colors[strat] = 'red'
            elif 'low_pLI' in str(strat):
                strat_colors[strat] = 'blue'
            elif 'High_impact' in str(strat):
                strat_colors[strat] = 'darkred'
            elif 'Low_impact' in str(strat):
                strat_colors[strat] = 'darkblue'
            else:
                strat_colors[strat] = plt.cm.tab10(i/len(unique_strats))
    
    # Create subplots, one for each cluster
    fig, axes = plt.subplots(2, 2, figsize=fig_size, sharey=True)
    axes = axes.flatten()
    
    # Find global x limits for consistent scales
    min_x = results_df[x_col].min()
    max_x = results_df[x_col].max()
    
    # Add some padding to x limits
    padding = 0.1 * (max_x - min_x)
    x_limits = (min_x - padding, max_x + padding)
    
    # Plot each cluster in its own subplot
    for i, cluster in enumerate(clusters):
        if i >= len(axes):
            print(f"Warning: More clusters than subplots. Only showing first {len(axes)} clusters.")
            break
            
        cluster_data = results_df[results_df[hue_col] == cluster]
        
        # Plot points colored by stratification
        if stratify_col and stratify_col in cluster_data.columns:
            for strat, strat_df in cluster_data.groupby(stratify_col):
                # Non-significant points first
                non_sig = strat_df[~strat_df['significant']]
                sig = strat_df[strat_df['significant']]
                
                color = strat_colors.get(strat, 'gray')
                
                # Non-significant points
                if not non_sig.empty:
                    axes[i].scatter(
                        non_sig[x_col],
                        non_sig['plot_label'],
                        s=80,
                        color=color,
                        alpha=0.3,
                        marker='o'
                    )
                
                # Significant points
                if not sig.empty:
                    axes[i].scatter(
                        sig[x_col],
                        sig['plot_label'],
                        s=100,
                        color=color,
                        alpha=0.8,
                        edgecolor='black',
                        linewidth=1,
                        marker='o'
                    )
                    
            # Add a legend for this subplot
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=strat_colors.get(strat, 'gray'),
                           markersize=8, label=pli_mapping.get(strat, strat))
                for strat in cluster_data[stratify_col].unique()
            ]
            axes[i].legend(handles=legend_elements, title=stratify_col.replace('_', ' ').title(), fontsize=8)
        else:
            # Simple plot without stratification
            axes[i].scatter(
                cluster_data[x_col],
                cluster_data['plot_label'],
                s=80,
                color='blue',
                alpha=0.7
            )
        
        # Add a vertical line at x=0
        axes[i].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Set consistent x limits
        axes[i].set_xlim(x_limits)
        
        # Customize subplot
        axes[i].set_title(f"Cluster {cluster}", fontsize=12)
        axes[i].set_xlabel(x_col.replace('_', ' ').title(), fontsize=10)
        
        # Only add y-label to leftmost subplots
        if i % 2 == 0:
            axes[i].set_ylabel("Variant Category", fontsize=10)
        
        # Add a light grid
        axes[i].grid(alpha=0.3)
    
    # Add main title
    fig.suptitle(title, fontsize=14)
    
    # Add text explaining effect direction if using log2fc
    if x_col == 'log2fc':
        for i in range(min(len(clusters), len(axes))):
            axes[i].text(
                x_limits[0] + 0.1 * (x_limits[1] - x_limits[0]),
                -0.5,  # Below the bottom of the plot
                "← Depleted",
                fontsize=8
            )
            axes[i].text(
                x_limits[1] - 0.1 * (x_limits[1] - x_limits[0]),
                -0.5,  # Below the bottom of the plot
                "Enriched →",
                fontsize=8
            )
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make space for title
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_combined_enrichment_and_ratio(enrichment_df: pd.DataFrame,
                                     ratio_df: pd.DataFrame,
                                     output_path: str,
                                     title: str = "Variant Profiles by Cluster",
                                     fig_size: Tuple[int, int] = (18, 10)) -> None:
    """
    Create a combined visualization with variant enrichment and rare:common ratio.
    
    Args:
        enrichment_df (pd.DataFrame): Results from enrichment analysis
        ratio_df (pd.DataFrame): Results from rare:common ratio analysis
        output_path (str): Path to save the figure
        title (str): Plot title
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Create a figure with two rows and two columns
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(2, 2, height_ratios=[2, 1])
    
    # Top row: Enrichment plot (larger)
    ax1 = plt.subplot(gs[0, :])
    
    # Bottom row: Rare:Common ratio plots
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    
    # 1. Create enrichment plot in top panel
    
    # Check for required columns
    if 'log2fc' not in enrichment_df.columns:
        print("Warning: 'log2fc' column not found in enrichment data.")
        return
    
    # Process data
    if 'impact_bin' in enrichment_df.columns:
        stratify_col = 'impact_bin'
    elif 'impact' in enrichment_df.columns:
        stratify_col = 'impact'
    else:
        stratify_col = None
    
    # Create more informative labels
    if stratify_col:
        # Map pLI bin values to more readable labels
        pli_mapping = {
            'High_high_pLI': 'High pLI (≥0.9)',
            'High_low_pLI': 'Low pLI (<0.9)',
            'Moderate_high_pLI': 'High pLI (≥0.9)',
            'Moderate_low_pLI': 'Low pLI (<0.9)',
            'Low_high_pLI': 'High pLI (≥0.9)',
            'Low_low_pLI': 'Low pLI (<0.9)'
        }
        
        # Create a new combined label
        enrichment_df['plot_label'] = enrichment_df.apply(
            lambda row: f"{row['hierarchy']} → {pli_mapping.get(row[stratify_col], row[stratify_col])}",
            axis=1
        )
    else:
        enrichment_df['plot_label'] = enrichment_df['hierarchy']
    
    # Determine significance
    if 'adjusted_pvalue' in enrichment_df.columns:
        enrichment_df['significant'] = enrichment_df['adjusted_pvalue'] < 0.05
    else:
        enrichment_df['significant'] = False
    
    # Get clusters
    clusters = sorted(enrichment_df['cluster1'].unique() if 'cluster1' in enrichment_df.columns else enrichment_df['cluster'].unique())
    
    # Create color mapping for clusters
    cluster_colors = {
        cluster: COLORS.get(f'Cluster{cluster}', plt.cm.tab10(i/len(clusters))) 
        for i, cluster in enumerate(clusters)
    }
    
    # Plot enrichment data
    cluster_col = 'cluster1' if 'cluster1' in enrichment_df.columns else 'cluster'
    
    for cluster in clusters:
        cluster_data = enrichment_df[enrichment_df[cluster_col] == cluster]
        
        # Plot non-significant points first
        non_sig = cluster_data[~cluster_data['significant']]
        sig = cluster_data[cluster_data['significant']]
        
        color = cluster_colors[cluster]
        
        # Non-significant points
        if not non_sig.empty:
            ax1.scatter(
                non_sig['log2fc'],
                non_sig['plot_label'],
                s=80,
                color=color,
                alpha=0.3,
                marker='o',
                label=f'Cluster {cluster} (n.s.)' if sig.empty else None
            )
        
        # Significant points
        if not sig.empty:
            ax1.scatter(
                sig['log2fc'],
                sig['plot_label'],
                s=100,
                color=color,
                alpha=0.8,
                edgecolor='black',
                linewidth=1,
                marker='o',
                label=f'Cluster {cluster}'
            )
    
    # Add a vertical line at x=0
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Customize the plot
    ax1.set_xlabel("Log2 Fold Change", fontsize=12)
    ax1.set_ylabel("Variant Category", fontsize=12)
    ax1.set_title("Variant Enrichment by Cluster", fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend(title="Cluster", fontsize=10)
    
    # 2. Create rare:common ratio boxplot in bottom left panel
    # Check if ratio_df has the needed columns
    required_ratio_cols = ['cluster', 'ratio']
    for col in required_ratio_cols:
        if col not in ratio_df.columns:
            print(f"Warning: Column '{col}' not found in ratio data.")
            return
    
    # Create boxplot
    sns.boxplot(
        x='cluster',
        y='ratio',
        data=ratio_df,
        palette=cluster_colors,
        ax=ax2
    )
    
    # Add individual data points
    sns.stripplot(
        x='cluster',
        y='ratio',
        data=ratio_df,
        color='black',
        size=5,
        alpha=0.5,
        ax=ax2
    )
    
    # Add median labels
    medians = ratio_df.groupby('cluster')['ratio'].median()
    for cluster in clusters:
        if cluster in medians:
            ax2.text(
                clusters.index(cluster),
                0.01,  # Just above the bottom
                f"Median: {medians[cluster]:.2f}",
                ha='center',
                fontsize=9
            )
    
    # Customize the plot
    ax2.set_xlabel("Cluster", fontsize=12)
    ax2.set_ylabel("Rare/Common Variant Ratio", fontsize=12)
    ax2.set_title("Rare:Common Variant Ratio by Cluster", fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Create a summary barplot in bottom right panel
    if 'category' in enrichment_df.columns:
        # Summarize log2fc by hierarchy and cluster
        summary = enrichment_df.groupby(['hierarchy', cluster_col])['log2fc'].mean().reset_index()
        
        # Create the barplot
        sns.barplot(
            x='hierarchy',
            y='log2fc',
            hue=cluster_col,
            data=summary,
            palette=cluster_colors,
            ax=ax3
        )
        
        # Add a line at y=0
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Customize the plot
        ax3.set_xlabel("Variant Category", fontsize=12)
        ax3.set_ylabel("Average Log2 Fold Change", fontsize=12)
        ax3.set_title("Average Enrichment by Category", fontsize=14)
        ax3.grid(axis='y', alpha=0.3)
        ax3.legend(title="Cluster", fontsize=10)
        
        # Rotate x-axis labels if needed
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make space for title
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_cluster_enrichment(results_df: pd.DataFrame,
                          output_path: str,
                          title: str = "Cluster-specific Enrichment",
                          x_col: str = 'log2fc',
                          y_col: str = 'hierarchy',
                          hue_col: str = 'cluster',
                          size_col: str = None,
                          significant_col: str = 'adjusted_pvalue',
                          fig_size: Tuple[int, int] = (12, 8)) -> None:
    """
    Create a cluster-specific enrichment plot.
    
    Args:
        results_df (pd.DataFrame): Results from analysis
        output_path (str): Path to save the figure
        title (str): Plot title
        x_col (str): Column to use for x-axis
        y_col (str): Column to use for y-axis categories
        hue_col (str): Column to use for coloring points
        size_col (str): Column to use for point sizes (optional)
        significant_col (str): Column to use for significance
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Check for required columns
    required_cols = [x_col, y_col, hue_col]
    for col in required_cols:
        if col not in results_df.columns:
            print(f"Warning: Column '{col}' not found in data. Enrichment plot creation skipped.")
            return
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Determine significance
    if significant_col in results_df.columns:
        results_df['significant'] = results_df[significant_col] < 0.05
    else:
        results_df['significant'] = False
    
    # Create point plot
    if size_col and size_col in results_df.columns:
        # Size points by the specified column
        sizes = results_df[size_col] * 20  # Scale sizes appropriately
        sizes = sizes.clip(upper=500)  # Cap maximum size
        
        # Create scatter plot with varying sizes
        for cluster in sorted(results_df[hue_col].unique()):
            cluster_data = results_df[results_df[hue_col] == cluster]
            
            # Plot non-significant points first (as background)
            non_sig = cluster_data[~cluster_data['significant']]
            sig = cluster_data[cluster_data['significant']]
            
            color = COLORS.get(f'Cluster{cluster}', sns.color_palette()[cluster-1])
            
            # Non-significant points (more transparent)
            if not non_sig.empty:
                ax.scatter(
                    non_sig[x_col],
                    non_sig[y_col],
                    s=non_sig[size_col] * 20,
                    color=color,
                    alpha=0.3,
                    label=f'Cluster {cluster} (n.s.)' if sig.empty else None
                )
            
            # Significant points
            if not sig.empty:
                ax.scatter(
                    sig[x_col],
                    sig[y_col],
                    s=sig[size_col] * 20,
                    color=color,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1,
                    label=f'Cluster {cluster}'
                )
    else:
        # Create regular scatter plot
        for cluster in sorted(results_df[hue_col].unique()):
            cluster_data = results_df[results_df[hue_col] == cluster]
            
            # Plot non-significant points first (as background)
            non_sig = cluster_data[~cluster_data['significant']]
            sig = cluster_data[cluster_data['significant']]
            
            color = COLORS.get(f'Cluster{cluster}', sns.color_palette()[cluster-1])
            
            # Non-significant points (more transparent)
            if not non_sig.empty:
                ax.scatter(
                    non_sig[x_col],
                    non_sig[y_col],
                    s=80,
                    color=color,
                    alpha=0.3,
                    label=f'Cluster {cluster} (n.s.)' if sig.empty else None
                )
            
            # Significant points
            if not sig.empty:
                ax.scatter(
                    sig[x_col],
                    sig[y_col],
                    s=100,
                    color=color,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1,
                    label=f'Cluster {cluster}'
                )
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Customize the plot
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add a light grid
    ax.grid(alpha=0.3)
    
    # Add a legend
    ax.legend(title=hue_col.replace('_', ' ').title(), fontsize=10)
    
    # Add text explaining effect direction if using log2fc
    if x_col == 'log2fc':
        plt.figtext(0.15, 0.01, "← Depleted", ha='center', fontsize=10)
        plt.figtext(0.85, 0.01, "Enriched →", ha='center', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_pathway_results(pathway_results: pd.DataFrame,
                       output_path: str,
                       title: str = "Pathway Enrichment Analysis",
                       max_pathways: int = 20,
                       fig_size: Tuple[int, int] = (12, 10)) -> None:
    """
    Create a visualization of pathway enrichment results.
    
    Args:
        pathway_results (pd.DataFrame): Results from pathway analysis
        output_path (str): Path to save the figure
        title (str): Plot title
        max_pathways (int): Maximum number of pathways to show per cluster
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Check for required columns
    required_cols = ['cluster', 'pathway_name', 'enrichment', 'adjusted_pvalue', 'pathway_genes']
    
    for col in required_cols:
        if col not in pathway_results.columns:
            if col == 'adjusted_pvalue' and 'pvalue' in pathway_results.columns:
                # Use pvalue if adjusted_pvalue is not available
                pathway_results['adjusted_pvalue'] = pathway_results['pvalue']
            else:
                print(f"Warning: Column '{col}' not found in data. Pathway plot creation skipped.")
                return
    
    # Filter for significant pathways and limit to top pathways per cluster
    sig_pathways = pathway_results[pathway_results['adjusted_pvalue'] < 0.05].copy()
    
    if sig_pathways.empty:
        print("No significant pathways found. Using all pathways instead.")
        sig_pathways = pathway_results.copy()
    
    # For each cluster, get the top pathways by significance
    top_pathways = []
    for cluster in sig_pathways['cluster'].unique():
        cluster_pathways = sig_pathways[sig_pathways['cluster'] == cluster]
        cluster_pathways = cluster_pathways.sort_values('adjusted_pvalue').head(max_pathways)
        top_pathways.append(cluster_pathways)
    
    plot_df = pd.concat(top_pathways)
    
    # Calculate -log10(p-value) for better visualization
    plot_df['-log10(p)'] = -np.log10(plot_df['adjusted_pvalue'])
    
    # Create a more informative label
    plot_df['plot_label'] = plot_df.apply(
        lambda row: f"{row['pathway_name']} (n={row['pathway_genes']})",
        axis=1
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create bubble plot
    scatter = ax.scatter(
        plot_df['enrichment'],
        plot_df['plot_label'],
        s=plot_df['-log10(p)'] * 20,  # Size by significance
        c=plot_df['cluster'].apply(lambda c: COLORS.get(f'Cluster{c}', sns.color_palette()[c-1])),
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add a vertical line at x=1 (no enrichment)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Customize the plot
    ax.set_xlabel("Enrichment (Observed/Expected)", fontsize=12)
    ax.set_ylabel("Pathway", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add a light grid
    ax.grid(alpha=0.3)
    
    # Create legend for clusters
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS.get(f'Cluster{c}', sns.color_palette()[c-1]), 
               markersize=10, label=f'Cluster {c}')
        for c in sorted(plot_df['cluster'].unique())
    ]
    
    # Add size legend
    p_values = [0.05, 0.01, 0.001, 0.0001]
    legend_elements += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=np.sqrt(-np.log10(p) * 20) / 2, label=f'p={p}')
        for p in p_values
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add text explaining enrichment direction
    plt.figtext(0.15, 0.01, "← Depleted", ha='center', fontsize=10)
    plt.figtext(0.85, 0.01, "Enriched →", ha='center', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_cluster_enrichment(results_df: pd.DataFrame,
                          output_path: str,
                          title: str = "Cluster-specific Enrichment",
                          x_col: str = 'log2fc',
                          y_col: str = 'hierarchy',
                          hue_col: str = 'cluster',
                          size_col: str = None,
                          significant_col: str = 'adjusted_pvalue',
                          fig_size: Tuple[int, int] = (12, 8)) -> None:
    """
    Create a cluster-specific enrichment plot.
    
    Args:
        results_df (pd.DataFrame): Results from analysis
        output_path (str): Path to save the figure
        title (str): Plot title
        x_col (str): Column to use for x-axis
        y_col (str): Column to use for y-axis categories
        hue_col (str): Column to use for coloring points
        size_col (str): Column to use for point sizes (optional)
        significant_col (str): Column to use for significance
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Check for required columns
    required_cols = [x_col, y_col, hue_col]
    for col in required_cols:
        if col not in results_df.columns:
            print(f"Warning: Column '{col}' not found in data. Enrichment plot creation skipped.")
            return
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Determine significance
    if significant_col in results_df.columns:
        results_df['significant'] = results_df[significant_col] < 0.05
    else:
        results_df['significant'] = False
    
    # Create point plot
    if size_col and size_col in results_df.columns:
        # Size points by the specified column
        sizes = results_df[size_col] * 20  # Scale sizes appropriately
        sizes = sizes.clip(upper=500)  # Cap maximum size
        
        # Create scatter plot with varying sizes
        for cluster in sorted(results_df[hue_col].unique()):
            cluster_data = results_df[results_df[hue_col] == cluster]
            
            # Plot non-significant points first (as background)
            non_sig = cluster_data[~cluster_data['significant']]
            sig = cluster_data[cluster_data['significant']]
            
            color = COLORS.get(f'Cluster{cluster}', sns.color_palette()[cluster-1])
            
            # Non-significant points (more transparent)
            if not non_sig.empty:
                ax.scatter(
                    non_sig[x_col],
                    non_sig[y_col],
                    s=non_sig[size_col] * 20,
                    color=color,
                    alpha=0.3,
                    label=f'Cluster {cluster} (n.s.)' if sig.empty else None
                )
            
            # Significant points
            if not sig.empty:
                ax.scatter(
                    sig[x_col],
                    sig[y_col],
                    s=sig[size_col] * 20,
                    color=color,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1,
                    label=f'Cluster {cluster}'
                )
    else:
        # Create regular scatter plot
        for cluster in sorted(results_df[hue_col].unique()):
            cluster_data = results_df[results_df[hue_col] == cluster]
            
            # Plot non-significant points first (as background)
            non_sig = cluster_data[~cluster_data['significant']]
            sig = cluster_data[cluster_data['significant']]
            
            color = COLORS.get(f'Cluster{cluster}', sns.color_palette()[cluster-1])
            
            # Non-significant points (more transparent)
            if not non_sig.empty:
                ax.scatter(
                    non_sig[x_col],
                    non_sig[y_col],
                    s=80,
                    color=color,
                    alpha=0.3,
                    label=f'Cluster {cluster} (n.s.)' if sig.empty else None
                )
            
            # Significant points
            if not sig.empty:
                ax.scatter(
                    sig[x_col],
                    sig[y_col],
                    s=100,
                    color=color,
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1,
                    label=f'Cluster {cluster}'
                )
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Customize the plot
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add a light grid
    ax.grid(alpha=0.3)
    
    # Add a legend
    ax.legend(title=hue_col.replace('_', ' ').title(), fontsize=10)
    
    # Add text explaining effect direction if using log2fc
    if x_col == 'log2fc':
        plt.figtext(0.15, 0.01, "← Depleted", ha='center', fontsize=10)
        plt.figtext(0.85, 0.01, "Enriched →", ha='center', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_pathway_results(pathway_results: pd.DataFrame,
                       output_path: str,
                       title: str = "Pathway Enrichment Analysis",
                       max_pathways: int = 20,
                       fig_size: Tuple[int, int] = (12, 10)) -> None:
    """
    Create a visualization of pathway enrichment results.
    
    Args:
        pathway_results (pd.DataFrame): Results from pathway analysis
        output_path (str): Path to save the figure
        title (str): Plot title
        max_pathways (int): Maximum number of pathways to show per cluster
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Check for required columns
    required_cols = ['cluster', 'pathway_name', 'enrichment', 'adjusted_pvalue', 'pathway_genes']
    
    for col in required_cols:
        if col not in pathway_results.columns:
            if col == 'adjusted_pvalue' and 'pvalue' in pathway_results.columns:
                # Use pvalue if adjusted_pvalue is not available
                pathway_results['adjusted_pvalue'] = pathway_results['pvalue']
            else:
                print(f"Warning: Column '{col}' not found in data. Pathway plot creation skipped.")
                return
    
    # Filter for significant pathways and limit to top pathways per cluster
    sig_pathways = pathway_results[pathway_results['adjusted_pvalue'] < 0.05].copy()
    
    if sig_pathways.empty:
        print("No significant pathways found. Using all pathways instead.")
        sig_pathways = pathway_results.copy()
    
    # For each cluster, get the top pathways by significance
    top_pathways = []
    for cluster in sig_pathways['cluster'].unique():
        cluster_pathways = sig_pathways[sig_pathways['cluster'] == cluster]
        cluster_pathways = cluster_pathways.sort_values('adjusted_pvalue').head(max_pathways)
        top_pathways.append(cluster_pathways)
    
    plot_df = pd.concat(top_pathways)
    
    # Calculate -log10(p-value) for better visualization
    plot_df['-log10(p)'] = -np.log10(plot_df['adjusted_pvalue'])
    
    # Create a more informative label
    plot_df['plot_label'] = plot_df.apply(
        lambda row: f"{row['pathway_name']} (n={row['pathway_genes']})",
        axis=1
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create bubble plot
    scatter = ax.scatter(
        plot_df['enrichment'],
        plot_df['plot_label'],
        s=plot_df['-log10(p)'] * 20,  # Size by significance
        c=plot_df['cluster'].apply(lambda c: COLORS.get(f'Cluster{c}', sns.color_palette()[c-1])),
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add a vertical line at x=1 (no enrichment)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Customize the plot
    ax.set_xlabel("Enrichment (Observed/Expected)", fontsize=12)
    ax.set_ylabel("Pathway", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add a light grid
    ax.grid(alpha=0.3)
    
    # Create legend for clusters
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS.get(f'Cluster{c}', sns.color_palette()[c-1]), 
               markersize=10, label=f'Cluster {c}')
        for c in sorted(plot_df['cluster'].unique())
    ]
    
    # Add size legend
    p_values = [0.05, 0.01, 0.001, 0.0001]
    legend_elements += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=np.sqrt(-np.log10(p) * 20) / 2, label=f'p={p}')
        for p in p_values
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add text explaining enrichment direction
    plt.figtext(0.15, 0.01, "← Depleted", ha='center', fontsize=10)
    plt.figtext(0.85, 0.01, "Enriched →", ha='center', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_volcano_plot(results_df: pd.DataFrame,
                      output_path: str,
                      title: str = "Variant Association Analysis",
                      x_col: str = 'log2fc',
                      y_col: str = 'pvalue',
                      category_col: str = 'category',
                      p_threshold: float = 0.05,
                      fc_threshold: float = 1.0,
                      fig_size: Tuple[int, int] = (10, 8)) -> None:
    """
    Create a volcano plot for variant associations.
    
    Args:
        results_df (pd.DataFrame): Results from analysis
        output_path (str): Path to save the figure
        title (str): Plot title
        x_col (str): Column to use for x-axis (fold change)
        y_col (str): Column to use for y-axis (p-value)
        category_col (str): Column to use for coloring points
        p_threshold (float): P-value threshold for significance
        fc_threshold (float): Fold change threshold for effect size
        fig_size (tuple): Figure dimensions (width, height)
    """
    # Check for required columns
    required_cols = [x_col, y_col]
    for col in required_cols:
        if col not in results_df.columns:
            print(f"Warning: Column '{col}' not found in data. Volcano plot creation skipped.")
            return
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Compute -log10(p-value) for y-axis
    plot_df = results_df.copy()
    plot_df['-log10(p)'] = -np.log10(plot_df[y_col])
    
    # Determine significance based on p-value and fold change
    plot_df['significant'] = (plot_df[y_col] < p_threshold) & (abs(plot_df[x_col]) > fc_threshold)
    
    # Create color mapping if category column is available
    if category_col in plot_df.columns:
        categories = plot_df[category_col].unique()
        category_colors = {
            cat: COLORS.get(cat, sns.color_palette()[i % 10]) 
            for i, cat in enumerate(categories)
        }
        
        # Plot points by category
        for category in categories:
            category_data = plot_df[plot_df[category_col] == category]
            
            # Non-significant points
            non_sig = category_data[~category_data['significant']]
            if not non_sig.empty:
                ax.scatter(
                    non_sig[x_col],
                    non_sig['-log10(p)'],
                    s=30,
                    color=category_colors[category],
                    alpha=0.3,
                    label=f'{category} (n.s.)' if category_data['significant'].any() else category
                )
            
            # Significant points
            sig = category_data[category_data['significant']]
            if not sig.empty:
                ax.scatter(
                    sig[x_col],
                    sig['-log10(p)'],
                    s=50,
                    color=category_colors[category],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5,
                    label=category
                )
    else:
        # Default coloring based on significance only
        # Non-significant points
        non_sig = plot_df[~plot_df['significant']]
        if not non_sig.empty:
            ax.scatter(
                non_sig[x_col],
                non_sig['-log10(p)'],
                s=30,
                color='gray',
                alpha=0.3,
                label='Not significant'
            )
        
        # Significant points
        sig = plot_df[plot_df['significant']]
        if not sig.empty:
            ax.scatter(
                sig[x_col],
                sig['-log10(p)'],
                s=50,
                color='red',
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5,
                label='Significant'
            )
    
    # Add threshold lines
    ax.axhline(y=-np.log10(p_threshold), color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=fc_threshold, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=-fc_threshold, color='red', linestyle='--', alpha=0.5)
    
    # Customize the plot
    ax.set_xlabel(f"{x_col.replace('_', ' ').title()}", fontsize=12)
    ax.set_ylabel("-log10(p-value)", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add a light grid
    ax.grid(alpha=0.3)
    
    # Add a legend (with de-duplication)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)
    
    # Add text for quadrants
    if x_col == 'log2fc':
        plt.figtext(0.15, 0.95, "Depleted", ha='center', fontsize=10)
        plt.figtext(0.85, 0.95, "Enriched", ha='center', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

from src.utils_gene_stats import compare_functional_impact_distributions

def create_variant_distribution_plots(coding_data: Dict,
                                    output_dir: str,
                                    variant_type: str = "rare") -> None:
    """
    Create a series of plots showing the distribution of variant types across clusters.
    
    Args:
        coding_data (dict): The loaded coding variant data
        output_dir (str): Directory to save the output plots
        variant_type (str): Type of variant to analyze ('rare' or 'common')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract functional impact distribution
    impact_distribution = compare_functional_impact_distributions(coding_data, variant_type)
    
    # Create stacked bar chart of variant types by cluster
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pivot the data for plotting
    pivot_df = impact_distribution.pivot(
        index='cluster',
        columns='category',
        values='proportion'
    ).fillna(0)
    
    # Ensure all categories are present
    for category in ['PTV', 'Missense', 'Synonymous']:
        if category not in pivot_df.columns:
            pivot_df[category] = 0
    
    # Sort columns in functional impact order
    pivot_df = pivot_df[['PTV', 'Missense', 'Synonymous']]
    
    # Create stacked bar chart
    pivot_df.plot(
        kind='bar',
        stacked=True,
        color=[COLORS.get(cat, f"C{i}") for i, cat in enumerate(['PTV', 'Missense', 'Synonymous'])],
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title(f"{variant_type.capitalize()} Variant Type Distribution by Cluster", fontsize=14)
    ax.legend(title="Variant Type")
    
    # Add percentage labels on bars
    for i, cluster in enumerate(pivot_df.index):
        cum_sum = 0
        for category in pivot_df.columns:
            value = pivot_df.at[cluster, category]
            if value > 0.03:  # Only add label if segment is large enough
                ax.text(
                    i,
                    cum_sum + (value / 2),
                    f"{value:.1%}",
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color='white' if value > 0.2 else 'black'
                )
            cum_sum += value
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{variant_type}_variant_distribution.png"), dpi=300)
    plt.close(fig)
    
    # Create a heatmap of variant types by cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        pivot_df.T,  # Transpose to get categories as rows
        cmap="YlGnBu",
        annot=True,
        fmt=".1%",
        linewidths=0.5,
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Variant Type", fontsize=12)
    ax.set_title(f"{variant_type.capitalize()} Variant Type Distribution by Cluster", fontsize=14)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{variant_type}_variant_heatmap.png"), dpi=300)
    plt.close(fig)

def create_pli_distribution_plots(coding_data: Dict,
                                output_dir: str,
                                variant_type: str = "rare") -> None:
    """
    Create plots showing the distribution of pLI constraint across clusters.
    
    Args:
        coding_data (dict): The loaded coding variant data
        output_dir (str): Directory to save the output plots
        variant_type (str): Type of variant to analyze ('rare' or 'common')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and organize pLI data
    rows = []
    
    for cluster, cluster_data in coding_data.items():
        cluster_id = int(cluster)
        
        # Get variant counts for this variant type
        variant_counts = cluster_data.get('variant_counts', {}).get(variant_type, {})
        
        # Process each category
        for category in ['PTV', 'Missense', 'Synonymous']:
            if category in variant_counts:
                # Sum high and low pLI bins
                high_pli_counts = sum(
                    count for bin_name, count in variant_counts[category].items()
                    if 'high_pLI' in bin_name
                )
                
                low_pli_counts = sum(
                    count for bin_name, count in variant_counts[category].items()
                    if 'low_pLI' in bin_name
                )
                
                # Calculate total
                total = high_pli_counts + low_pli_counts
                
                # Calculate proportions
                high_pli_prop = high_pli_counts / total if total > 0 else 0
                low_pli_prop = low_pli_counts / total if total > 0 else 0
                
                # Add high pLI row
                rows.append({
                    'cluster': cluster_id,
                    'category': category,
                    'pli_constraint': 'High (≥0.9)',
                    'count': high_pli_counts,
                    'proportion': high_pli_prop
                })
                
                # Add low pLI row
                rows.append({
                    'cluster': cluster_id,
                    'category': category,
                    'pli_constraint': 'Low (<0.9)',
                    'count': low_pli_counts,
                    'proportion': low_pli_prop
                })
    
    # Create DataFrame
    pli_df = pd.DataFrame(rows)
    
    # Create grouped bar chart of pLI distribution by cluster and category
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    
    # Plot each category in a separate subplot
    for i, category in enumerate(['PTV', 'Missense', 'Synonymous']):
        category_df = pli_df[pli_df['category'] == category]
        
        if category_df.empty:
            continue
        
        # Pivot for plotting
        pivot_df = category_df.pivot(
            index='cluster',
            columns='pli_constraint',
            values='proportion'
        ).fillna(0)
        
        # Create grouped bar chart
        pivot_df.plot(
            kind='bar',
            ax=axes[i],
            color=['#D62728', '#7F7F7F']  # Red for high pLI, gray for low pLI
        )
        
        # Customize subplot
        axes[i].set_title(category)
        axes[i].set_xlabel("Cluster")
        
        if i == 0:
            axes[i].set_ylabel("Proportion")
        
        # Add percentage labels on bars
        for j, p in enumerate(axes[i].patches):
            value = p.get_height()
            if value > 0.03:  # Only add label if bar is tall enough
                axes[i].annotate(
                    f"{value:.1%}",
                    (p.get_x() + p.get_width() / 2., value),
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    rotation=0
                )
        
        # Add legend if this is the first category
        if i == 0:
            axes[i].legend(title="pLI Constraint")
        else:
            axes[i].get_legend().remove()
    
    # Set overall title
    fig.suptitle(f"{variant_type.capitalize()} Variant pLI Constraint Distribution by Category", fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make space for suptitle
    plt.savefig(os.path.join(output_dir, f"{variant_type}_pli_distribution.png"), dpi=300)
    plt.close(fig)