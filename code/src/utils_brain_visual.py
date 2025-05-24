"""
Utility functions for visualizing brain network analysis results.

This module provides functions to create various visualizations for both
individual-level and group-level analysis results, including:
- Connectivity matrices (heatmaps)
- Brain network graphs (node and edge visualizations)
- Network integration visualizations (betweenness-based)
- Network segregation visualizations (clustering-based)
- Group comparison visualizations (bar plots and matrices)
- Edge connectivity comparisons between groups

These visualization tools support both exploratory analysis of individual 
subjects and statistical comparison between clinical groups.

Author: Chanyoung Ko
Date: 03-26-2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import seaborn as sns
import networkx as nx
from scipy import stats


def plot_connectivity_matrix(matrix, title=None, cmap='RdBu_r', vmin=None, vmax=None, labels=None):
    """
    Plot a connectivity matrix as a heatmap.
    
    Parameters
    ----------
    matrix : ndarray
        Connectivity matrix to visualize
    title : str, optional
        Title for the plot
    cmap : str, optional
        Colormap to use (default: 'RdBu_r')
    vmin, vmax : float, optional
        Value range for colormap
    labels : list, optional
        Labels for matrix rows/columns
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set default limits if not provided
    if vmin is None:
        vmin = -np.max(np.abs(matrix))
    if vmax is None:
        vmax = np.max(np.abs(matrix))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                   aspect='equal', interpolation='none')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Connectivity Strength')
    
    # Add title
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add labels if provided
    if labels:
        # Ensure we don't have too many labels that would make the plot unreadable
        if len(labels) <= 20:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
        else:
            # If too many labels, show fewer tick marks
            tick_step = max(1, len(labels) // 10)
            ax.set_xticks(np.arange(0, len(labels), tick_step))
            ax.set_yticks(np.arange(0, len(labels), tick_step))
            ax.set_xticklabels([labels[i] for i in range(0, len(labels), tick_step)], 
                              rotation=45, ha='right')
            ax.set_yticklabels([labels[i] for i in range(0, len(labels), tick_step)])
    
    # Add grid lines
    ax.set_xticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    plt.tight_layout()
    return fig


def plot_brain_networks(metric_data, atlas_mapping, title=None, cmap='viridis'):
    """
    Visualize brain network metrics on a brain network diagram.
    
    Parameters
    ----------
    metric_data : ndarray or dict
        Metric values for each node or edge
    atlas_mapping : dict
        Dictionary mapping ROI indices to network labels
    title : str, optional
        Title for the plot
    cmap : str, optional
        Colormap to use (default: 'viridis')
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract network information
    networks = sorted(set(atlas_mapping.values()))
    network_to_idx = {network: i for i, network in enumerate(networks)}
    
    # Create node positions (circular layout)
    n_networks = len(networks)
    network_angles = np.linspace(0, 2*np.pi, n_networks, endpoint=False)
    network_positions = {}
    for i, network in enumerate(networks):
        network_positions[network] = (np.cos(network_angles[i]), np.sin(network_angles[i]))
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for roi, network in atlas_mapping.items():
        if isinstance(roi, int):
            G.add_node(roi, network=network)
    
    # If metric data is for edges, add edges with weights
    if isinstance(metric_data, np.ndarray) and len(metric_data.shape) == 2:
        for i in range(metric_data.shape[0]):
            for j in range(metric_data.shape[1]):
                if i != j and abs(metric_data[i, j]) > 1e-6:
                    G.add_edge(i, j, weight=metric_data[i, j])
    
    # Position nodes
    pos = {}
    for node in G.nodes():
        network = G.nodes[node]['network']
        # Add some jitter within each network position
        jitter = np.random.uniform(-0.1, 0.1, 2)
        base_pos = network_positions[network]
        pos[node] = (base_pos[0] + jitter[0], base_pos[1] + jitter[1])
    
    # Draw network
    if isinstance(metric_data, np.ndarray) and len(metric_data.shape) == 2:
        # Edge metric visualization
        edges = [(i, j) for i, j in G.edges()]
        edge_colors = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5,
                              edge_cmap=plt.get_cmap(cmap), edge_vmin=min(edge_colors),
                              edge_vmax=max(edge_colors), ax=ax)
        
        # Add a colorbar
        sm = ScalarMappable(cmap=plt.get_cmap(cmap), 
                           norm=plt.Normalize(min(edge_colors), max(edge_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Edge Weight')
    
    elif isinstance(metric_data, np.ndarray) and len(metric_data.shape) == 1:
        # Node metric visualization
        node_colors = [metric_data[node] if node < len(metric_data) else 0 for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100,
                              cmap=plt.get_cmap(cmap), vmin=min(node_colors),
                              vmax=max(node_colors), ax=ax)
        
        # Add a colorbar
        sm = ScalarMappable(cmap=plt.get_cmap(cmap), 
                           norm=plt.Normalize(min(node_colors), max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Node Value')
    
    # Add node labels
    network_labels = {}
    for node, attrs in G.nodes(data=True):
        network = attrs['network']
        if network not in network_labels:
            network_labels[network] = network_positions[network]
    
    for network, position in network_labels.items():
        plt.text(position[0]*1.1, position[1]*1.1, network, 
                fontsize=10, ha='center', va='center')
    
    # Add title
    if title:
        plt.title(title, fontsize=14)
    
    # Set axis limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Turn off axis
    plt.axis('off')
    
    plt.tight_layout()
    return fig


def create_circular_layout(connectivity_matrix, network_names, title=None, 
                          threshold=0.1, cmap='RdBu_r'):
    """
    Create a circular layout visualization of brain networks.
    
    Parameters
    ----------
    connectivity_matrix : ndarray
        Connectivity matrix to visualize
    network_names : list
        List of network names
    title : str, optional
        Title for the plot
    threshold : float, optional
        Threshold for including connections (default: 0.1)
    cmap : str, optional
        Colormap to use (default: 'RdBu_r')
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Set up colormap
    cm = plt.get_cmap(cmap)
    vmax = np.max(np.abs(connectivity_matrix))
    vmin = -vmax
    
    # Number of networks
    n_networks = len(network_names)
    
    # Network color definition
    network_colors = plt.cm.tab20(np.linspace(0, 1, n_networks))
    
    # Calculate node positions on a circle
    node_angles = np.linspace(0, 2*np.pi, n_networks, endpoint=False)
    node_positions = np.column_stack([np.cos(node_angles), np.sin(node_angles)])
    
    # Draw network nodes
    node_size = 800
    for i, (name, pos) in enumerate(zip(network_names, node_positions)):
        circle = Circle(pos, radius=0.1, color=network_colors[i], 
                       alpha=0.8, linewidth=2, edgecolor='black')
        ax.add_patch(circle)
        
        # Add network name label
        label_pos = pos * 1.12  # Slightly outside the circle
        ax.text(label_pos[0], label_pos[1], name, ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    # Draw connections
    for i in range(n_networks):
        for j in range(i+1, n_networks):
            # Get connection strength
            weight = connectivity_matrix[i, j]
            
            # Skip weak connections
            if abs(weight) < threshold:
                continue
            
            # Calculate connection endpoints
            pos1 = node_positions[i]
            pos2 = node_positions[j]
            
            # Determine line color and width
            normalized_weight = (weight - vmin) / (vmax - vmin)
            line_color = cm(normalized_weight)
            line_width = 2 * abs(weight) / vmax
            
            # Draw curved connection
            start_pos = pos1 * 0.9  # Start slightly away from center of node
            end_pos = pos2 * 0.9    # End slightly away from center of node
            
            # Control points for curve
            mid_point = (start_pos + end_pos) / 2
            
            # Add curvature based on distance
            dist = np.linalg.norm(end_pos - start_pos)
            curve_factor = 0.3 * (1 - dist / 2)  # More curve for closer nodes
            
            # Calculate perpendicular vector for curve control point
            perp = np.array([-mid_point[1], mid_point[0]])
            perp = perp / np.linalg.norm(perp) * curve_factor
            
            # Control point
            control_point = mid_point + perp
            
            # Create curved connection
            arc = FancyArrowPatch(
                start_pos, end_pos,
                connectionstyle=f"arc3,rad={curve_factor}",
                color=line_color, linewidth=max(0.5, line_width*5), 
                alpha=0.7, zorder=0
            )
            ax.add_patch(arc)
    
    # Add colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Connectivity Strength')
    
    # Add title
    if title:
        plt.title(title, fontsize=14)
    
    # Set axis limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Turn off axis
    plt.axis('off')
    
    plt.tight_layout()
    return fig


def plot_group_difference_matrix(group1_data, group2_data, network_names, 
                                p_values=None, alpha=0.05, title=None, cmap='RdBu_r'):
    """
    Create a heatmap showing connectivity differences between two groups.
    
    Parameters
    ----------
    group1_data : ndarray
        Connectivity matrix for group 1 (average)
    group2_data : ndarray
        Connectivity matrix for group 2 (average)
    network_names : list
        List of network names
    p_values : ndarray, optional
        Matrix of p-values for significance testing
    alpha : float, optional
        Significance level (default: 0.05)
    title : str, optional
        Title for the plot
    cmap : str, optional
        Colormap to use (default: 'RdBu_r')
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Calculate difference matrix
    diff_matrix = group2_data - group1_data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up colormap
    vmax = np.max(np.abs(diff_matrix))
    vmin = -vmax
    
    # Create heatmap
    im = ax.imshow(diff_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Connectivity Difference')
    
    # Add significance markers if p-values provided
    if p_values is not None:
        # Add asterisks for significant differences
        for i in range(diff_matrix.shape[0]):
            for j in range(diff_matrix.shape[1]):
                if p_values[i, j] < alpha:
                    # Determine marker text
                    if p_values[i, j] < 0.001:
                        sig_text = '*'
                    elif p_values[i, j] < 0.01:
                        sig_text = '*'
                    else:
                        sig_text = '*'
                    
                    # Add marker to cell
                    ax.text(j, i, sig_text, ha='center', va='center',
                           color='white' if abs(diff_matrix[i, j]) > vmax/2 else 'black',
                           fontsize=12, fontweight='bold')
    
    # Add network names as labels
    if network_names:
        ax.set_xticks(np.arange(len(network_names)))
        ax.set_yticks(np.arange(len(network_names)))
        ax.set_xticklabels(network_names, rotation=45, ha='right')
        ax.set_yticklabels(network_names)
    
    # Add grid lines
    ax.set_xticks(np.arange(diff_matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(diff_matrix.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Add title
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title("Between-Network Connectivity Difference", fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_group_bar_comparison(data, significance_data=None, effect_size_data=None,
                             groups=None, group_colors=None, alpha=0.05, title=None):
    """
    Create a bar plot comparing different groups with significance markers.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping group names to arrays of values
    significance_data : ndarray, optional
        Array of p-values for significance testing
    effect_size_data : ndarray, optional
        Array of effect sizes
    groups : list, optional
        List of group names to include (default: all groups in data)
    group_colors : list, optional
        List of colors for each group
    alpha : float, optional
        Significance level (default: 0.05)
    title : str, optional
        Title for the plot
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine groups to plot
    if groups is None:
        groups = list(data.keys())
    
    # Set default colors if not provided
    if group_colors is None:
        group_colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    
    # Extract data for each group
    group_means = [np.mean(data[group]) for group in groups]
    group_errors = [np.std(data[group]) / np.sqrt(len(data[group])) for group in groups]
    
    # Create bar plot
    bar_positions = np.arange(len(groups))
    bars = ax.bar(bar_positions, group_means, yerr=group_errors, 
                 color=group_colors, alpha=0.7, ecolor='black', capsize=10)
    
    # Add significance markers if provided
    if significance_data is not None:
        # Height for significance bars
        max_height = max(group_means) + max(group_errors) * 3
        bar_height = max_height * 0.05
        
        # Add significance markers between groups
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                p_value = significance_data[i, j] if isinstance(significance_data, np.ndarray) else significance_data
                
                if p_value < alpha:
                    # Determine marker text
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    else:
                        sig_text = '*'
                    
                    # Draw significance bar
                    bar_y = max_height + (j - i) * bar_height
                    ax.plot([i, j], [bar_y, bar_y], 'k-', linewidth=1.5)
                    ax.text((i + j) / 2, bar_y + bar_height/2, sig_text, 
                           ha='center', va='center', fontsize=12)
    
    # Add axis labels and title
    ax.set_xlabel('Group')
    ax.set_ylabel('Value')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(groups)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add legend for effect sizes if provided
    if effect_size_data is not None:
        # Create legend for effect sizes
        effect_size_legend = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                effect_size = effect_size_data[i, j] if isinstance(effect_size_data, np.ndarray) else effect_size_data
                effect_size_legend.append(
                    f"{groups[i]} vs {groups[j]}: d = {effect_size:.2f}"
                )
        
        if effect_size_legend:
            ax.text(0.05, 0.95, '\n'.join(effect_size_legend),
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    return fig


def plot_graph_metric_comparison(metric_data, group_labels, metric_name,
                               significance_data=None, alpha=0.05, title=None):
    """
    Create a bar plot comparing graph metrics between groups.
    
    Parameters
    ----------
    metric_data : dict
        Dictionary mapping subject IDs to metric values
    group_labels : dict
        Dictionary mapping subject IDs to group labels
    metric_name : str
        Name of the metric being compared
    significance_data : dict, optional
        Dictionary of p-values for group comparisons
    alpha : float, optional
        Significance level (default: 0.05)
    title : str, optional
        Title for the plot
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique groups
    unique_groups = sorted(set(group_labels.values()))
    n_groups = len(unique_groups)
    
    # Group data
    grouped_data = {group: [] for group in unique_groups}
    for subject_id, value in metric_data.items():
        if subject_id in group_labels:
            group = group_labels[subject_id]
            grouped_data[group].append(value)
    
    # Calculate statistics
    group_means = [np.mean(grouped_data[group]) for group in unique_groups]
    group_errors = [np.std(grouped_data[group]) / np.sqrt(len(grouped_data[group])) 
                  for group in unique_groups]
    
    # Create bar plot
    bar_positions = np.arange(n_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    bars = ax.bar(bar_positions, group_means, yerr=group_errors, 
                 color=colors, alpha=0.7, ecolor='black', capsize=10)
    
    # Add significance markers if provided
    if significance_data is not None:
        # Height for significance bars
        max_height = max(group_means) + max(group_errors) * 3
        bar_height = max_height * 0.05
        
        # Add significance markers between groups
        for i, group1 in enumerate(unique_groups):
            for j, group2 in enumerate(unique_groups[i+1:], i+1):
                key = f"{group1}_{group2}"
                if key in significance_data and significance_data[key] < alpha:
                    # Determine marker text
                    if significance_data[key] < 0.001:
                        sig_text = '***'
                    elif significance_data[key] < 0.01:
                        sig_text = '**'
                    else:
                        sig_text = '*'
                    
                    # Draw significance bar
                    bar_y = max_height + (j - i) * bar_height
                    ax.plot([i, j], [bar_y, bar_y], 'k-', linewidth=1.5)
                    ax.text((i + j) / 2, bar_y + bar_height/2, sig_text, 
                           ha='center', va='center', fontsize=12)
    
    # Set labels and title
    ax.set_ylabel(metric_name)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(unique_groups)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"{metric_name} Comparison", fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_edge_connectivity_comparison(edge_data, group_labels, network_names, 
                                     significance_data=None, alpha=0.05, title=None):
    """
    Create bar plots comparing edge-wise connectivity between groups.
    
    Parameters
    ----------
    edge_data : dict
        Dictionary mapping edge names to arrays of values for each subject
    group_labels : ndarray
        Array of group labels for each subject
    network_names : list
        List of network names
    significance_data : dict, optional
        Dictionary mapping edge names to p-values
    alpha : float, optional
        Significance level (default: 0.05)
    title : str, optional
        Title for the plot
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Get unique groups
    unique_groups = np.unique(group_labels)
    n_groups = len(unique_groups)
    
    # Determine number of edges to plot
    n_edges = len(edge_data)
    
    # Calculate grid dimensions
    n_cols = min(4, n_edges)
    n_rows = int(np.ceil(n_edges / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    
    # Flatten axes for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Group colors
    group_colors = plt.cm.tab10(np.linspace(0, 1, n_groups))
    
    # Plot each edge
    for i, (edge_name, edge_values) in enumerate(edge_data.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Group data by group labels
        grouped_data = {}
        for group in unique_groups:
            grouped_data[group] = [edge_values[j] for j, label in enumerate(group_labels) 
                                  if label == group]
        
        # Calculate means and errors
        group_means = [np.mean(grouped_data[group]) for group in unique_groups]
        group_errors = [np.std(grouped_data[group]) / np.sqrt(len(grouped_data[group])) 
                      for group in unique_groups]
        
        # Create bar plot
        bar_positions = np.arange(n_groups)
        bars = ax.bar(bar_positions, group_means, yerr=group_errors, 
                     color=group_colors, alpha=0.7, ecolor='black', capsize=5)
        
        # Add significance markers if provided
        if significance_data is not None and edge_name in significance_data:
            # Get p-values for this edge
            p_values = significance_data[edge_name]
            
            # Height for significance bars
            max_height = max(group_means) + max(group_errors) * 2
            bar_height = max_height * 0.05
            
            # Add significance markers between groups
            for j in range(n_groups):
                for k in range(j+1, n_groups):
                    p_value_idx = f"{unique_groups[j]}_{unique_groups[k]}"
                    if p_value_idx in p_values and p_values[p_value_idx] < alpha:
                        # Determine marker text
                        if p_values[p_value_idx] < 0.001:
                            sig_text = '*'
                        elif p_values[p_value_idx] < 0.01:
                            sig_text = '*'
                        else:
                            sig_text = '*'
                        
                        # Draw significance bar
                        bar_y = max_height + (k - j) * bar_height
                        ax.plot([j, k], [bar_y, bar_y], 'k-', linewidth=1)
                        ax.text((j + k) / 2, bar_y + bar_height/2, sig_text, 
                               ha='center', va='center', fontsize=10)
        
        # Set axis labels
        ax.set_title(edge_name)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(unique_groups, rotation=45)
        
        # Hide x-axis label for all but bottom row
        if i < n_edges - n_cols:
            ax.set_xticklabels([])
    
    # Hide empty subplots
    for i in range(n_edges, len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Add common legend
    handles = [Patch(color=color, alpha=0.7, label=group) 
              for color, group in zip(group_colors, unique_groups)]
    fig.legend(handles=handles, loc='lower center', ncol=n_groups, 
              bbox_to_anchor=(0.5, 0), bbox_transform=fig.transFigure)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for legend
    return fig


def plot_network_integration(connectivity_matrix, atlas_mapping, betweenness_values=None,
                           threshold=0.1, title=None):
    """
    Visualize network integration with node size representing betweenness centrality.
    
    Parameters
    ----------
    connectivity_matrix : ndarray
        Connectivity matrix between networks
    atlas_mapping : dict or list
        Dictionary mapping ROI indices to network labels or list of network names
    betweenness_values : ndarray, optional
        Betweenness centrality values for each network
    threshold : float, optional
        Threshold for including connections (default: 0.1)
    title : str, optional
        Title for the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Extract network information
    if isinstance(atlas_mapping, dict):
        networks = sorted(set(atlas_mapping.values()))
    else:
        networks = atlas_mapping
    
    n_networks = len(networks)
    
    # If betweenness values not provided, compute them from connectivity
    if betweenness_values is None:
        # Create network graph
        G = nx.Graph()
        for i in range(n_networks):
            G.add_node(i, name=networks[i])
        
        # Add edges with abs(weight) > threshold
        for i in range(n_networks):
            for j in range(i+1, n_networks):
                if abs(connectivity_matrix[i, j]) > threshold:
                    G.add_edge(i, j, weight=abs(connectivity_matrix[i, j]))
        
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(G, weight='weight')
        betweenness_values = np.array([betweenness[i] for i in range(n_networks)])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up visual properties
    edge_cmap = plt.cm.RdBu_r  # Red-Blue for positive-negative
    node_cmap = plt.cm.YlOrRd   # Yellow-Orange-Red for betweenness
    
    # Calculate node positions using spring layout
    G = nx.Graph()
    for i in range(n_networks):
        G.add_node(i, name=networks[i])
    
    # Add edges
    for i in range(n_networks):
        for j in range(i+1, n_networks):
            if abs(connectivity_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=abs(connectivity_matrix[i, j]))
    
    # Position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Normalize betweenness for node size
    if np.sum(betweenness_values) > 0:  # Check if any non-zero values
        node_sizes = 100 + 1000 * (betweenness_values / max(betweenness_values))
    else:
        node_sizes = np.ones(n_networks) * 300
    
    # Draw edges with colors based on connectivity
    edges = [(i, j) for i in range(n_networks) for j in range(i+1, n_networks) 
            if abs(connectivity_matrix[i, j]) > threshold]
    
    # Define edge colors
    edge_colors = [connectivity_matrix[i, j] for i, j in edges]
    edge_abs = [abs(c) for c in edge_colors]
    max_edge = max(edge_abs) if edge_abs else 1
    
    # Define edge widths based on absolute strength
    edge_widths = [1 + 4 * (abs(c) / max_edge) for c in edge_colors]
    
    # Draw edges
    for i, ((src, tgt), color, width) in enumerate(zip(edges, edge_colors, edge_widths)):
        # Draw curved edges for easier visibility
        arc = FancyArrowPatch(
            pos[src], pos[tgt],
            connectionstyle=f"arc3,rad={0.1 + 0.05*i%2}",  # Alternate curve direction
            color=edge_cmap(0.5 + color/(2*max_edge)),  # Map to colormap
            linewidth=width, 
            alpha=0.7,
            arrowstyle='-'
        )
        ax.add_patch(arc)
    
    # Draw nodes
    for i in range(n_networks):
        # Use betweenness for node color
        node_color = node_cmap(betweenness_values[i] / max(betweenness_values) 
                             if max(betweenness_values) > 0 else 0)
        
        # Draw node
        circle = plt.Circle(pos[i], radius=0.03 + 0.02*(node_sizes[i]/max(node_sizes)), 
                          color=node_color, alpha=0.8, linewidth=2, edgecolor='black')
        ax.add_patch(circle)
        
        # Add network label
        plt.text(pos[i][0], pos[i][1], networks[i], 
               fontsize=10, ha='center', va='center')
    
    # Add title
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title("Network Graph (Size = Betweenness)", fontsize=14)
    
    # Create legend for edge colors
    if edge_colors:
        sm_edges = ScalarMappable(cmap=edge_cmap, 
                               norm=plt.Normalize(-max_edge, max_edge))
        sm_edges.set_array([])
        cbar_edges = plt.colorbar(sm_edges, ax=ax, fraction=0.046, pad=0.04)
        cbar_edges.set_label('Connection Strength')
    
    # Create legend for node sizes
    if max(betweenness_values) > 0:
        sm_nodes = ScalarMappable(cmap=node_cmap, 
                                norm=plt.Normalize(0, max(betweenness_values)))
        sm_nodes.set_array([])
        cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.046, pad=0.1)
        cbar_nodes.set_label('Betweenness Centrality')
    
    # Set axis limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Turn off axis
    plt.axis('off')
    
    plt.tight_layout()
    return fig


def plot_network_segregation(connectivity_matrix, atlas_mapping, clustering_values=None,
                            threshold=0.1, title=None):
    """
    Visualize network segregation with node size representing clustering coefficient.
    
    Parameters
    ----------
    connectivity_matrix : ndarray
        Connectivity matrix between networks
    atlas_mapping : dict or list
        Dictionary mapping ROI indices to network labels or list of network names
    clustering_values : ndarray, optional
        Clustering coefficient values for each network
    threshold : float, optional
        Threshold for including connections (default: 0.1)
    title : str, optional
        Title for the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Extract network information
    if isinstance(atlas_mapping, dict):
        networks = sorted(set(atlas_mapping.values()))
    else:
        networks = atlas_mapping
    
    n_networks = len(networks)
    
    # If clustering values not provided, compute them from connectivity
    if clustering_values is None:
        # Create network graph
        G = nx.Graph()
        for i in range(n_networks):
            G.add_node(i, name=networks[i])
        
        # Add edges with abs(weight) > threshold
        for i in range(n_networks):
            for j in range(i+1, n_networks):
                if abs(connectivity_matrix[i, j]) > threshold:
                    G.add_edge(i, j, weight=abs(connectivity_matrix[i, j]))
        
        # Calculate clustering coefficient
        clustering = nx.clustering(G, weight='weight')
        clustering_values = np.array([clustering[i] for i in range(n_networks)])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up visual properties
    edge_cmap = plt.cm.RdBu_r  # Red-Blue for positive-negative
    node_cmap = plt.cm.Greens   # Greens for clustering (segregation)
    
    # Calculate node positions using spring layout
    G = nx.Graph()
    for i in range(n_networks):
        G.add_node(i, name=networks[i])
    
    # Add edges
    for i in range(n_networks):
        for j in range(i+1, n_networks):
            if abs(connectivity_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=abs(connectivity_matrix[i, j]))
    
    # Position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Normalize clustering for node size
    if np.sum(clustering_values) > 0:  # Check if any non-zero values
        node_sizes = 100 + 1000 * (clustering_values / max(clustering_values))
    else:
        node_sizes = np.ones(n_networks) * 300
    
    # Draw edges with colors based on connectivity
    edges = [(i, j) for i in range(n_networks) for j in range(i+1, n_networks) 
            if abs(connectivity_matrix[i, j]) > threshold]
    
    # Define edge colors
    edge_colors = [connectivity_matrix[i, j] for i, j in edges]
    edge_abs = [abs(c) for c in edge_colors]
    max_edge = max(edge_abs) if edge_abs else 1
    
    # Define edge widths based on absolute strength
    edge_widths = [1 + 4 * (abs(c) / max_edge) for c in edge_colors]
    
    # Draw edges
    for i, ((src, tgt), color, width) in enumerate(zip(edges, edge_colors, edge_widths)):
        # Draw curved edges for easier visibility
        arc = FancyArrowPatch(
            pos[src], pos[tgt],
            connectionstyle=f"arc3,rad={0.1 + 0.05*i%2}",  # Alternate curve direction
            color=edge_cmap(0.5 + color/(2*max_edge)),  # Map to colormap
            linewidth=width, 
            alpha=0.7,
            arrowstyle='-'
        )
        ax.add_patch(arc)
    
    # Draw nodes
    for i in range(n_networks):
        # Use clustering for node color
        node_color = node_cmap(clustering_values[i] / max(clustering_values) 
                             if max(clustering_values) > 0 else 0)
        
        # Draw node
        circle = plt.Circle(pos[i], radius=0.03 + 0.02*(node_sizes[i]/max(node_sizes)), 
                          color=node_color, alpha=0.8, linewidth=2, edgecolor='black')
        ax.add_patch(circle)
        
        # Add network label
        plt.text(pos[i][0], pos[i][1], networks[i], 
               fontsize=10, ha='center', va='center')
    
    # Add title
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title("Network Graph (Size = Clustering Coefficient)", fontsize=14)
    
    # Create legend for edge colors
    if edge_colors:
        sm_edges = ScalarMappable(cmap=edge_cmap, 
                               norm=plt.Normalize(-max_edge, max_edge))
        sm_edges.set_array([])
        cbar_edges = plt.colorbar(sm_edges, ax=ax, fraction=0.046, pad=0.04)
        cbar_edges.set_label('Connection Strength')
    
    # Create legend for node sizes
    if max(clustering_values) > 0:
        sm_nodes = ScalarMappable(cmap=node_cmap, 
                                norm=plt.Normalize(0, max(clustering_values)))
        sm_nodes.set_array([])
        cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.046, pad=0.1)
        cbar_nodes.set_label('Clustering Coefficient')
    
    # Set axis limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Turn off axis
    plt.axis('off')
    
    plt.tight_layout()
    return fig