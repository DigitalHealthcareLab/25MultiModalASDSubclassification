"""
Enhanced functions for brain connectivity visualization based on Buch et al. and Vandewouw et al.
"""
import os
import numpy as np
import pandas as pd
from nilearn import plotting
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import seaborn as sns

# Display mode mapping between anatomical terms and nilearn codes
DISPLAY_MODE_MAPPING = {
    'lateral_left': 'l',
    'lateral_right': 'r',
    'lateral_both': 'lr',
    'sagittal': 'x',
    'coronal': 'y',
    'axial': 'z',
    'multiple_views': 'ortho'
}

def generate_paper_style_visualizations(comparison_results, ja_comparisons, networks, output_dir,
                                      region_info=None, network_to_regions=None):
    from pathlib import Path
    output_dir = Path(output_dir)
    
    # Get coordinates if region_info is provided
    region_coords = get_mni_coordinates(region_info) if region_info is not None else None
    
    for task, results in comparison_results.items():
        task_dir = output_dir / task
        os.makedirs(task_dir, exist_ok=True)
        
        task_features = ja_comparisons[task]['features']
        
        for comparison_key, data in results.items():
            print(f"Generating paper-style visualizations for {task} - {comparison_key}")
            
            # 1. Network-level chord diagram
            if 'network_diff' in data and 'significant_network_conns' in data:
                create_chord_diagram(
                    data['network_diff'],
                    networks,
                    significant_mask=data['significant_network_conns'],
                    title=f"{task}: {comparison_key} Network Connectivity",
                    save_path=task_dir / f"{comparison_key}_chord_diagram.png"
                )
            
            # 2. Network-level brain plot (replaces ROI-based surface plot)
            if region_coords is not None and 'network_diff' in data and 'significant_network_conns' in data:
                create_network_level_brain_plot(
                    network_matrix=data['network_diff'],
                    region_coords=region_coords,
                    network_to_regions=network_to_regions,
                    significant_mask=data['significant_network_conns'],
                    networks=networks,
                    title=f"{task}: {comparison_key} Network Differences",
                    save_path=task_dir / f"{comparison_key}_network_brain_plot.png"
                )
            
            # 3. Effect size brain rendering
            if 'effect_sizes' in data and region_coords is not None:
                create_effect_size_brain_rendering(
                    data['effect_sizes'],
                    region_coords,
                    title=f"{task}: {comparison_key} Effect Sizes",
                    save_path=task_dir / f"{comparison_key}_effect_size_brain.png"
                )
            
            # 4. Feature-specific comparisons (violin + bar)
            for feature in task_features:
                feature_result = data['feature_results'].get(feature)
                if feature_result:
                    create_enhanced_feature_comparison_plot(
                        feature,
                        feature_result,
                        comparison_key,
                        save_path=task_dir / f"{comparison_key}_{feature}_comparison.png"
                    )
            
            # 5. Subgroup connectivity (13×13 only)
            create_subgroup_connectivity_comparison_with_significance(
                data.get('cluster1_network_conn'), 
                data.get('cluster2_network_conn'),
                data.get('significant_network_conns'),
                networks,
                title=f"{task}: {comparison_key} Subgroup Connectivity",
                save_path=task_dir / f"{comparison_key}_subgroup_connectivity.png"
            )

    print(f"Paper-style visualizations saved to {output_dir}")

def get_mni_coordinates(region_info):
    """Extract MNI coordinates from region info dataframe."""
    if all(col in region_info.columns for col in ['MNI_X', 'MNI_Y', 'MNI_Z']):
        coords = region_info[['MNI_X', 'MNI_Y', 'MNI_Z']].values
    elif all(col in region_info.columns for col in ['x', 'y', 'z']):
        coords = region_info[['x', 'y', 'z']].values
    else:
        print("Warning: MNI coordinates not found in region_info")
        return None
    return coords


def create_chord_diagram(network_matrix, networks, significant_mask=None, 
                       title="Network Connectivity", save_path=None):
    """
    Create a chord diagram similar to those in Buch et al.
    
    Parameters:
    -----------
    network_matrix : numpy.ndarray
        Network connectivity matrix
    networks : list
        List of network names
    significant_mask : numpy.ndarray, optional
        Boolean mask to highlight significant connections
    title : str, optional
        Plot title
    save_path : str or Path, optional
        Path to save the figure
    """
    try:
        import networkx as nx
        import matplotlib.colors as mcolors
        
        # Create figure
        plt.figure(figsize=(12, 12))
        
        # Create colormap for edges based on correlation values
        vmax = np.max(np.abs(network_matrix))
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        edge_cmap = cm.coolwarm
        
        # Create a symmetric graph
        G = nx.Graph()
        
        # Determine node positions on a circle
        n_nodes = len(networks)
        node_positions = {}
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        
        # Create nodes with colors from qualitative colormap
        node_cmap = plt.cm.tab20
        color_indices = np.linspace(0, 1, n_nodes, endpoint=False)
        
        for i, network in enumerate(networks):
            # Position nodes on a circle
            x, y = np.cos(angles[i]), np.sin(angles[i])
            G.add_node(network, pos=(x, y))
            node_positions[network] = (x, y)
        
        # Add connections with colors and widths based on correlation
        edge_colors = []
        edge_widths = []
        
        for i in range(len(networks)):
            for j in range(i+1, len(networks)):
                connection_strength = network_matrix[i, j]
                
                # Skip non-significant connections if mask is provided
                if significant_mask is not None and not significant_mask[i, j]:
                    continue
                    
                # Use absolute value for width but keep sign for color
                edge_width = np.abs(connection_strength) * 5  # Scale for visibility
                
                # Only add edges with reasonable strength for visibility
                if edge_width > 0.05:
                    G.add_edge(networks[i], networks[j], weight=edge_width)
                    edge_colors.append(edge_cmap(norm(connection_strength)))
                    edge_widths.append(edge_width)
        
        # Draw the graph
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes with color coding
        nx.draw_networkx_nodes(G, pos, 
                            node_size=1000, 
                            node_color=[node_cmap(color_indices[i]) for i in range(n_nodes)],
                            alpha=0.9)
        
        # Draw edges with colors based on correlation and width based on strength
        if edge_colors:
            nx.draw_networkx_edges(G, pos, 
                                edge_color=edge_colors, 
                                width=edge_widths,
                                alpha=0.7,
                                connectionstyle='arc3,rad=0.1')  # Curved edges
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Create colorbar for edge colors
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
        cbar.set_label('Connectivity Difference')
        
        # Remove axis
        plt.axis('off')
        plt.title(title, fontsize=16)
        
        # Save the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
        
    except ImportError as e:
        print(f"Error importing necessary libraries for chord diagram: {e}")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f"Error creating chord diagram: {e}", 
               ha='center', va='center', fontsize=12)
        if save_path:
            plt.savefig(save_path)
        return plt.gcf()
    except Exception as e:
        print(f"Error creating chord diagram: {e}")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f"Error creating chord diagram: {e}", 
               ha='center', va='center', fontsize=12)
        if save_path:
            plt.savefig(save_path)
        return plt.gcf()

def create_enhanced_feature_comparison_plot(feature_name, feature_result, comparison_key, 
                                          save_path=None):
    """
    Create an enhanced feature comparison plot with both violin and box plots.
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature being visualized
    feature_result : dict
        Dictionary with feature results from statistical analysis
    comparison_key : str
        String identifying the comparison (e.g., "4_vs_3")
    save_path : str or Path, optional
        Path to save the figure
    """
    plt.figure(figsize=(14, 8))
    
    # Extract cluster numbers for labeling
    cluster1, cluster2 = comparison_key.split('_vs_')
    
    # Check what data we have available
    has_distributions = 'cluster1_values' in feature_result and 'cluster2_values' in feature_result
    
    if has_distributions and len(feature_result['cluster1_values']) > 0 and len(feature_result['cluster2_values']) > 0:
        # Create a DataFrame for easier plotting with seaborn
        data1 = feature_result['cluster1_values']
        data2 = feature_result['cluster2_values']
        
        # Calculate Cohen's d effect size
        try:
            effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt(
                ((len(data1) - 1) * np.var(data1, ddof=1) + 
                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                (len(data1) + len(data2) - 2))
        except:
            effect_size = np.nan
            
        # Create a DataFrame
        df = pd.DataFrame({
            'Value': data1 + data2,
            'Cluster': [f'Cluster {cluster1}'] * len(data1) + [f'Cluster {cluster2}'] * len(data2)
        })
        
        # Create a combined violin/box/swarm plot
        ax = plt.subplot(1, 2, 1)
        sns.violinplot(x='Cluster', y='Value', data=df, inner=None, palette=['#3498db', '#e74c3c'], alpha=0.7)
        sns.boxplot(x='Cluster', y='Value', data=df, width=0.3, palette=['#3498db', '#e74c3c'], 
                   boxprops={'alpha':0.8}, medianprops={'color':'black'})
        sns.swarmplot(x='Cluster', y='Value', data=df, color='white', edgecolor='black', size=5, alpha=0.7)
        
        plt.title("Distribution Comparison")
        plt.ylabel(feature_name.replace('_', ' ').title())
        
        # Add significance markers if applicable
        if feature_result.get('pvalue', 1.0) < 0.05:
            y_max = max(max(data1), max(data2))
            plt.text(0.5, y_max * 1.1, '*', fontsize=24, ha='center')
            plt.text(0.5, y_max * 1.05, f"p={feature_result['pvalue']:.3f}", 
                   fontsize=10, ha='center')
        
        # Create a plot of means with error bars
        plt.subplot(1, 2, 2)
        means = [np.mean(data1), np.mean(data2)]
        sems = [stats.sem(data1), stats.sem(data2)]
        
        bars = plt.bar([f'Cluster {cluster1}', f'Cluster {cluster2}'], means, 
                      yerr=sems, capsize=10, alpha=0.7,
                      color=['#3498db', '#e74c3c'])
        
        plt.ylabel(feature_name.replace('_', ' ').title())
        plt.title("Mean Comparison")
        
        # Add stats text box
        stats_text = (
            f"Cluster {cluster1} (n={len(data1)}):\n"
            f"  Mean: {np.mean(data1):.3f}\n"
            f"  SD: {np.std(data1):.3f}\n\n"
            f"Cluster {cluster2} (n={len(data2)}):\n"
            f"  Mean: {np.mean(data2):.3f}\n"
            f"  SD: {np.std(data2):.3f}\n\n"
            f"Stats:\n"
            f"  Difference: {feature_result.get('difference', np.mean(data1) - np.mean(data2)):.3f}\n"
            f"  t-statistic: {feature_result.get('tstat', np.nan):.3f}\n"
            f"  p-value: {feature_result.get('pvalue', np.nan):.3f}\n"
            f"  Effect size (Cohen's d): {effect_size:.3f}"
        )
        
        plt.figtext(0.95, 0.5, stats_text, fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), 
                  verticalalignment='center')
    
    else:
        # Simple bar chart for single values
        plt.bar([f"Cluster {cluster1}", f"Cluster {cluster2}"],
              [feature_result.get('cluster1_value', 0), feature_result.get('cluster2_value', 0)])
        
        plt.ylabel(feature_name.replace('_', ' ').title())
        
    plt.suptitle(f"{feature_name.replace('_', ' ').title()} Comparison", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return plt.gcf()

def create_subgroup_connectivity_comparison_with_significance(cluster1_matrix, cluster2_matrix, 
                                                               significant_matrix, networks, 
                                                               title="Subgroup Connectivity", 
                                                               save_path=None):
    """
    Create a 3D-style circular layout chord plot that visualizes subgroup connectivity comparison,
    highlighting only significant edges if provided.

    Parameters:
    -----------
    cluster1_matrix : np.ndarray
        Network-level connectivity matrix for Cluster 1.
    cluster2_matrix : np.ndarray
        Network-level connectivity matrix for Cluster 2.
    significant_matrix : np.ndarray
        Boolean matrix indicating significant differences between network pairs.
    networks : list
        List of network names.
    title : str
        Title for the plot.
    save_path : str
        If provided, path to save the image.
    """
    # Difference matrix
    diff_matrix = cluster1_matrix - cluster2_matrix
    
    # Create the graph
    G = nx.Graph()
    num_networks = len(networks)
    angles = np.linspace(0, 2 * np.pi, num_networks, endpoint=False)
    pos = {networks[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(num_networks)}

    # Add nodes with labels and positions
    for net in networks:
        G.add_node(net)

    # Normalize for color map
    vmax = np.max(np.abs(diff_matrix))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = cm.coolwarm

    edge_colors = []
    edge_widths = []

    for i in range(num_networks):
        for j in range(i + 1, num_networks):
            if significant_matrix[i, j]:
                diff = diff_matrix[i, j]
                G.add_edge(networks[i], networks[j])
                edge_colors.append(cmap(norm(diff)))
                edge_widths.append(np.abs(diff) * 25)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000,
                           node_color=plt.cm.tab20(np.linspace(0, 1, num_networks)))
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    if edge_colors:
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edge_color=edge_colors,
                               width=edge_widths,
                               connectionstyle='arc3,rad=0.2')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("Connectivity Difference")

    plt.axis('off')
    plt.title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

# Mock test run (if needed to display)
networks = ['DMN', 'DAN', 'VAN', 'FP', 'CO', 'Salience', 'Sensorimotor', 'Visual', 'Auditory', 'Cerebellar', 'Subcortical', 'Memory', 'Attention']
np.random.seed(0)
mock_c1 = np.random.rand(13, 13)
mock_c2 = np.random.rand(13, 13)
mock_significance = np.random.rand(13, 13) < 0.1
mock_c1 = (mock_c1 + mock_c1.T) / 2
mock_c2 = (mock_c2 + mock_c2.T) / 2
mock_significance = np.triu(mock_significance, 1)
mock_significance = mock_significance + mock_significance.T

fig = create_subgroup_connectivity_comparison_with_significance(mock_c1, mock_c2, mock_significance, networks)
plt.show()


def create_segregation_integration_visualization(cluster1_metrics, cluster2_metrics, region_coords,
                                              networks, network_to_regions, title="Segregation vs Integration",
                                              save_path=None):
    """
    Create a brain visualization showing segregation and integration differences
    between clusters, similar to Vandewouw et al.
    
    Parameters:
    -----------
    cluster1_metrics : dict
        Dictionary of graph metrics for cluster 1
    cluster2_metrics : dict
        Dictionary of graph metrics for cluster 2
    region_coords : numpy.ndarray
        Node coordinates in MNI space
    networks : list
        List of network names
    network_to_regions : dict
        Mapping from networks to region indices
    title : str, optional
        Plot title
    save_path : str or Path, optional
        Path to save the figure
    """
    try:
        # Create figure with 2x2 layout (segregation and integration for both hemispheres)
        fig = plt.figure(figsize=(18, 12))
        
        # Extract segregation and integration metrics
        # For segregation, we can use clustering coefficient
        # For integration, we can use participation coefficient
        
        # Check if we have these metrics
        if ('clustering_coef' not in cluster1_metrics or 'clustering_coef' not in cluster2_metrics or
            'participation_coef' not in cluster1_metrics or 'participation_coef' not in cluster2_metrics):
            raise ValueError("Required metrics (clustering_coef or participation_coef) not found in data")
        
        # Calculate differences between clusters
        segregation_diff = np.array(cluster1_metrics['clustering_coef']) - np.array(cluster2_metrics['clustering_coef'])
        integration_diff = np.array(cluster1_metrics['participation_coef']) - np.array(cluster2_metrics['participation_coef'])
        
        # Create colormap with diverging colors
        seg_vmax = np.max(np.abs(segregation_diff))
        int_vmax = np.max(np.abs(integration_diff))
        
        seg_norm = mcolors.Normalize(vmin=-seg_vmax, vmax=seg_vmax)
        int_norm = mcolors.Normalize(vmin=-int_vmax, vmax=int_vmax)
        
        # Use the display mode mapping
        left_lateral = DISPLAY_MODE_MAPPING['lateral_left']
        right_lateral = DISPLAY_MODE_MAPPING['lateral_right']
        
        # Plot segregation - left hemisphere
        ax1 = plt.subplot(2, 2, 1)
        segregation_colors = plt.cm.RdBu_r(seg_norm(segregation_diff))
        
        plotting.plot_connectome(
            np.zeros((len(region_coords), len(region_coords))),  # No connections
            region_coords,
            node_color=segregation_colors,
            node_size=50 + np.abs(segregation_diff) * 100,  # Scale node size by effect
            display_mode=left_lateral,
            axes=ax1,
            title='Segregation - Left Hemisphere'
        )
        
        # Plot segregation - right hemisphere
        ax2 = plt.subplot(2, 2, 2)
        plotting.plot_connectome(
            np.zeros((len(region_coords), len(region_coords))),
            region_coords,
            node_color=segregation_colors,
            node_size=50 + np.abs(segregation_diff) * 100,
            display_mode=right_lateral,
            axes=ax2,
            title='Segregation - Right Hemisphere'
        )
        
        # Add colorbar for segregation
        sm1 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=seg_norm)
        sm1.set_array([])
        cbar1 = plt.colorbar(sm1, ax=[ax1, ax2], shrink=0.6)
        cbar1.set_label('Segregation Difference (Cluster 1 - Cluster 2)')
        
        # Plot integration - left hemisphere
        ax3 = plt.subplot(2, 2, 3)
        integration_colors = plt.cm.RdBu_r(int_norm(integration_diff))
        
        plotting.plot_connectome(
            np.zeros((len(region_coords), len(region_coords))),
            region_coords,
            node_color=integration_colors,
            node_size=50 + np.abs(integration_diff) * 100,
            display_mode=left_lateral,
            axes=ax3,
            title='Integration - Left Hemisphere'
        )
        
        # Plot integration - right hemisphere
        ax4 = plt.subplot(2, 2, 4)
        plotting.plot_connectome(
            np.zeros((len(region_coords), len(region_coords))),
            region_coords,
            node_color=integration_colors,
            node_size=50 + np.abs(integration_diff) * 100,
            display_mode=right_lateral,
            axes=ax4,
            title='Integration - Right Hemisphere'
        )
        
        # Add colorbar for integration
        sm2 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=int_norm)
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=[ax3, ax4], shrink=0.6)
        cbar2.set_label('Integration Difference (Cluster 1 - Cluster 2)')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    except Exception as e:
        print(f"Error creating segregation-integration visualization: {e}")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f"Error creating visualization: {e}", 
               ha='center', va='center', fontsize=12)
        if save_path:
            plt.savefig(save_path)
        return plt.gcf()
    

def create_effect_size_brain_rendering(effect_sizes, node_coords, 
                                     title="Effect Sizes", save_path=None):
    """
    Create a brain surface rendering with effect sizes, similar to Vandewouw et al.
    
    Parameters:
    -----------
    effect_sizes : numpy.ndarray
        Effect sizes for each region
    node_coords : numpy.ndarray
        Node coordinates in MNI space
    title : str, optional
        Plot title
    save_path : str or Path, optional
        Path to save the figure
    """
    try:
        # Create a figure with multiple views
        fig = plt.figure(figsize=(15, 10))
        
        # Define views for the brain rendering using display mode mapping
        view_mapping = [
            (DISPLAY_MODE_MAPPING['lateral_left'], 'Left Lateral'),
            (DISPLAY_MODE_MAPPING['lateral_right'], 'Right Lateral'),
            (DISPLAY_MODE_MAPPING['sagittal'], 'Sagittal'),
            (DISPLAY_MODE_MAPPING['axial'], 'Axial')
        ]
        
        # Create a colormap for effect sizes
        cmap = plt.cm.RdBu_r
        vmax = np.max(np.abs(effect_sizes))
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        
        # Plot different views
        for i, (view_code, view_name) in enumerate(view_mapping):
            # Create a subplot for each view
            ax = plt.subplot(2, 2, i + 1)
            
            # Create a 3D rendering with node sizes based on effect sizes
            node_size = np.abs(effect_sizes) * 100  # Scale appropriately
            node_color = np.array([cmap(norm(e)) for e in effect_sizes])
            
            # Plot connectome with proper display mode
            display = plotting.plot_connectome(
                np.zeros((len(node_coords), len(node_coords))),  # No connections
                node_coords,
                node_color=node_color,
                node_size=node_size,
                display_mode=view_code,
                axes=ax,
                title=view_name
            )
                
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gcf().get_axes(), shrink=0.7)
        cbar.set_label('Effect Size')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    except Exception as e:
        print(f"Error creating effect size brain rendering: {e}")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f"Error creating brain rendering: {e}", 
               ha='center', va='center', fontsize=12)
        if save_path:
            plt.savefig(save_path)
        return plt.gcf()


def create_brain_surface_plot(diff_matrix, node_coords, mask=None, 
                               title="Significant Connections", save_path=None):
    """
    Create a surface plot visualization of significant connectivity differences.

    Parameters:
    -----------
    diff_matrix : numpy.ndarray
        Difference matrix between two groups
    node_coords : numpy.ndarray
        Node coordinates in MNI space
    mask : numpy.ndarray, optional
        Boolean mask for significant connections (same shape as diff_matrix)
    title : str
        Title for the figure
    save_path : str or Path, optional
        Path to save the figure
    """
    try:
        # Apply mask if provided
        if mask is not None and mask.shape == diff_matrix.shape:
            plot_matrix = np.zeros_like(diff_matrix)
            plot_matrix[mask] = diff_matrix[mask]
        else:
            print("Warning: Invalid or missing mask, plotting unthresholded matrix")
            plot_matrix = diff_matrix.copy()

        vmax = np.max(np.abs(plot_matrix))
        if vmax == 0:
            print("Warning: All connections are zero after masking.")
            vmax = 1e-5  # Prevent divide-by-zero

        # Plot surface views
        fig = plt.figure(figsize=(18, 12))
        views = [
            (DISPLAY_MODE_MAPPING['sagittal'], "Sagittal"),
            (DISPLAY_MODE_MAPPING['coronal'], "Coronal"),
            (DISPLAY_MODE_MAPPING['axial'], "Axial"),
            (DISPLAY_MODE_MAPPING['multiple_views'], "Multiple Views"),
        ]

        for i, (view, view_title) in enumerate(views):
            ax = plt.subplot(2, 2, i+1)
            plotting.plot_connectome(
                plot_matrix,
                node_coords,
                node_color='black',  # Keep nodes simple
                node_size=10,
                edge_vmin=-vmax,
                edge_vmax=vmax,
                edge_cmap='coolwarm',
                display_mode=view,
                axes=ax,
                title=view_title
            )

        # Add colorbar for edge differences
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=mcolors.Normalize(vmin=-vmax, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gcf().get_axes(), shrink=0.6)
        cbar.set_label("Connectivity Difference")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved surface plot to {save_path}")

        return fig
    except Exception as e:
        print(f"Error creating surface plot: {e}")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f"Error creating surface plot: {e}", 
                 ha='center', va='center', fontsize=12)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return plt.gcf()


def create_network_level_brain_plot(network_matrix, region_coords, network_to_regions, 
                                    significant_mask=None, networks=None,
                                    title="Network-Level Brain Plot", save_path=None):
    """
    Visualize network-level differences on a brain using average region coordinates per network.
    """
    try:
        # Step 1: Compute average coordinate per network
        net_centroids = []
        for network in networks:
            region_ids = network_to_regions.get(network, [])
            region_coords_net = [region_coords[i] for i in region_ids if i < len(region_coords)]
            if region_coords_net:
                avg_coord = np.mean(region_coords_net, axis=0)
            else:
                avg_coord = np.array([0, 0, 0])  # fallback
            net_centroids.append(avg_coord)
        net_centroids = np.array(net_centroids)

        # Step 2: Build edge matrix
        plot_matrix = network_matrix.copy()
        if significant_mask is not None:
            plot_matrix[~significant_mask] = 0

        # Step 3: Plot
        plt.figure(figsize=(8, 8))
        display = plotting.plot_connectome(
            plot_matrix,
            net_centroids,
            node_color='black',
            node_size=120,
            edge_cmap='coolwarm',
            edge_vmin=-np.max(np.abs(plot_matrix)),
            edge_vmax=np.max(np.abs(plot_matrix)),
            display_mode='ortho',
            title=title
        )
        
        # Add network labels
        for i, label in enumerate(networks):
            x, y, z = net_centroids[i]
            plt.text(x, y, label, fontsize=8, color='black', ha='center')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return plt.gcf()

    except Exception as e:
        print(f"Error creating network-level brain plot: {e}")
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', fontsize=12)
        if save_path:
            plt.savefig(save_path)
        return plt.gcf()


#####################Outdated Code#####################

# def create_network_brain_visualization(diff_matrix, significant_mask, effect_sizes, node_coords, 
#                                     networks, network_to_regions, 
#                                     title="Brain Network Differences", save_path=None):
#     """
#     Create a brain visualization showing network differences with significant connections.
#     """
#     try:
#         # Create a figure with multiple views
#         fig = plt.figure(figsize=(15, 12))
        
#         # Only show significant connections
#         connectivity = diff_matrix.copy()
#         connectivity[~significant_mask] = 0
        
#         # Create node colors based on network membership
#         node_colors = np.zeros(len(node_coords), dtype=object)
#         network_colors = plt.cm.tab20(np.linspace(0, 1, len(networks)))
        
#         for i, network in enumerate(networks):
#             if network in network_to_regions:
#                 valid_regions = [r for r in network_to_regions[network] if r < len(node_coords)]
#                 for region in valid_regions:
#                     node_colors[region] = network_colors[i]
        
#         # Node sizes based on effect sizes
#         node_size = 50 + np.abs(effect_sizes) * 100  # Base size + scaled effect size
        
#         # Use display mode mapping
#         view_mapping = [
#             (DISPLAY_MODE_MAPPING['sagittal'], 'Sagittal'),
#             (DISPLAY_MODE_MAPPING['coronal'], 'Coronal'),
#             (DISPLAY_MODE_MAPPING['axial'], 'Axial'),
#             (DISPLAY_MODE_MAPPING['multiple_views'], 'Multiple Views')
#         ]
        
#         # Debug: Print information to console, not to figure
#         # print(f"Shape of connectivity matrix: {connectivity.shape}")
#         # print(f"Shape of node coordinates: {node_coords.shape}")
        
#         # Plot different views
#         for i, (view_code, view_name) in enumerate(view_mapping):
#             ax = plt.subplot(2, 2, i+1)
            
#             # Convert all node_colors to hex if needed
#             plot_colors = [mcolors.to_hex(c) if isinstance(c, np.ndarray) else c for c in node_colors]
            
#             # Plot connectome with proper display mode
#             try:
#                 display = plotting.plot_connectome(
#                     connectivity,
#                     node_coords,
#                     node_color=plot_colors,
#                     node_size=node_size,
#                     edge_threshold=None,  # Show all non-zero connections
#                     edge_cmap='coolwarm',
#                     edge_vmin=-np.max(np.abs(connectivity)),
#                     edge_vmax=np.max(np.abs(connectivity)),
#                     display_mode=view_code,
#                     axes=ax,
#                     title=view_name
#                 )
#             except Exception as inner_e:
#                 # print(f"Error in plot_connectome for view {view_name}: {inner_e}")
#                 ax.text(0.5, 0.5, f"Error: {inner_e}", 
#                         ha='center', va='center', transform=ax.transAxes)
            
#         # Add colorbar to figure
#         sm = plt.cm.ScalarMappable(cmap='coolwarm', 
#                                  norm=mcolors.Normalize(vmin=-np.max(np.abs(connectivity)), 
#                                                       vmax=np.max(np.abs(connectivity))))
#         sm.set_array([])
#         cbar = plt.colorbar(sm, ax=plt.gcf().get_axes(), shrink=0.7)
#         cbar.set_label('Connectivity Difference')
        
#         plt.suptitle(title, fontsize=16)
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
#         return fig
        
#     except Exception as e:
#         print(f"Error creating network brain visualization: {e}")
#         plt.figure(figsize=(10, 8))
#         plt.text(0.5, 0.5, f"Error creating visualization: {e}", 
#                ha='center', va='center', fontsize=12)
#         if save_path:
#             plt.savefig(save_path)
#         return plt.gcf()
    
# def create_subgroup_connectivity_comparison(cluster1_matrix, cluster2_matrix, networks, 
#                                           title="Subgroup Connectivity", save_path=None):
#     """
#     Create a visualization comparing connectivity patterns between two clusters,
#     similar to Figure 2 of Buch et al.
    
#     Parameters:
#     -----------
#     cluster1_matrix : numpy.ndarray
#         Connectivity matrix for cluster 1
#     cluster2_matrix : numpy.ndarray
#         Connectivity matrix for cluster 2
#     networks : list
#         List of network names
#     title : str, optional
#         Plot title
#     save_path : str or Path, optional
#         Path to save the figure
#     """
#     if cluster1_matrix is None or cluster2_matrix is None:
#         print("Warning: Cluster matrices not available for comparison")
#         return None
    
#     # Ensure matrices are NumPy arrays
#     if not isinstance(cluster1_matrix, np.ndarray):
#         cluster1_matrix = np.array(cluster1_matrix)
#     if not isinstance(cluster2_matrix, np.ndarray):
#         cluster2_matrix = np.array(cluster2_matrix)
    
#     try:
#         # Create a figure with two chord diagrams
#         fig = plt.figure(figsize=(20, 10))
        
#         # Calculate thresholds for visualization (top 15% of connections)
#         threshold1 = np.percentile(np.abs(cluster1_matrix.flatten()), 85)
#         threshold2 = np.percentile(np.abs(cluster2_matrix.flatten()), 85)
        
#         # First chord diagram - Cluster 1
#         plt.subplot(1, 2, 1)
        
#         # Create a graph for this cluster
#         G1 = nx.Graph()
        
#         # Determine node positions on a circle
#         n_nodes = len(networks)
#         node_positions = {}
#         angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        
#         # Create nodes with colors
#         node_cmap = plt.cm.tab20
#         color_indices = np.linspace(0, 1, n_nodes, endpoint=False)
        
#         for i, network in enumerate(networks):
#             # Position nodes on a circle
#             x, y = np.cos(angles[i]), np.sin(angles[i])
#             G1.add_node(network, pos=(x, y))
#             node_positions[network] = (x, y)
        
#         # Create a colormap for edges
#         vmax = max(np.max(np.abs(cluster1_matrix)), np.max(np.abs(cluster2_matrix)))
#         norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
#         edge_cmap = cm.coolwarm
        
#         # Add connections with colors and widths
#         edge_colors1 = []
#         edge_widths1 = []
        
#         for i in range(len(networks)):
#             for j in range(i+1, len(networks)):
#                 connection_strength = cluster1_matrix[i, j]
                
#                 # Only include strong connections
#                 if np.abs(connection_strength) > threshold1:
#                     edge_width = np.abs(connection_strength) * 5  # Scale for visibility
#                     G1.add_edge(networks[i], networks[j], weight=edge_width)
#                     edge_colors1.append(edge_cmap(norm(connection_strength)))
#                     edge_widths1.append(edge_width)
        
#         # Draw the graph
#         pos = nx.get_node_attributes(G1, 'pos')
        
#         # Draw nodes with color coding
#         nx.draw_networkx_nodes(G1, pos, 
#                             node_size=1000, 
#                             node_color=[node_cmap(color_indices[i]) for i in range(n_nodes)],
#                             alpha=0.9)
        
#         # Draw edges
#         if edge_colors1:
#             nx.draw_networkx_edges(G1, pos, 
#                                 edge_color=edge_colors1, 
#                                 width=edge_widths1,
#                                 alpha=0.7,
#                                 connectionstyle='arc3,rad=0.1')  # Curved edges
        
#         # Draw node labels
#         nx.draw_networkx_labels(G1, pos, font_size=10, font_weight='bold')
        
#         plt.axis('off')
#         plt.title(f"Cluster 1 Connectivity", fontsize=14)
        
#         # Second chord diagram - Cluster 2
#         plt.subplot(1, 2, 2)
        
#         # Create a graph for this cluster
#         G2 = nx.Graph()
        
#         # Add the same nodes as in G1
#         for i, network in enumerate(networks):
#             G2.add_node(network, pos=node_positions[network])
        
#         # Add connections for Cluster 2
#         edge_colors2 = []
#         edge_widths2 = []
        
#         for i in range(len(networks)):
#             for j in range(i+1, len(networks)):
#                 connection_strength = cluster2_matrix[i, j]
                
#                 # Only include strong connections
#                 if np.abs(connection_strength) > threshold2:
#                     edge_width = np.abs(connection_strength) * 5  # Scale for visibility
#                     G2.add_edge(networks[i], networks[j], weight=edge_width)
#                     edge_colors2.append(edge_cmap(norm(connection_strength)))
#                     edge_widths2.append(edge_width)
        
#         # Draw the graph
#         nx.draw_networkx_nodes(G2, pos, 
#                             node_size=1000, 
#                             node_color=[node_cmap(color_indices[i]) for i in range(n_nodes)],
#                             alpha=0.9)
        
#         # Draw edges
#         if edge_colors2:
#             nx.draw_networkx_edges(G2, pos, 
#                                 edge_color=edge_colors2, 
#                                 width=edge_widths2,
#                                 alpha=0.7,
#                                 connectionstyle='arc3,rad=0.1')
        
#         # Draw node labels
#         nx.draw_networkx_labels(G2, pos, font_size=10, font_weight='bold')
        
#         plt.axis('off')
#         plt.title(f"Cluster 2 Connectivity", fontsize=14)
        
#         # Add a shared colorbar
#         sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
#         sm.set_array([])
#         cbar = plt.colorbar(sm, ax=plt.gcf().get_axes(), shrink=0.6, orientation='horizontal')
#         cbar.set_label('Connectivity Strength')
        
#         plt.suptitle(title, fontsize=16)
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
#         return fig
    
#     except Exception as e:
#         print(f"Error creating subgroup connectivity comparison: {e}")
#         plt.figure(figsize=(10, 8))
#         plt.text(0.5, 0.5, f"Error creating comparison: {e}", 
#                ha='center', va='center', fontsize=12)
#         if save_path:
#             plt.savefig(save_path)
#         return plt.gcf()
    

# def create_significance_heatmap(diff_matrix, mask=None, title="Significant Differences", 
#                               save_path=None, cmap="coolwarm", center=0):
#     """
#     Create a heatmap visualization of differences with significance masking.
    
#     Parameters:
#     -----------
#     diff_matrix : numpy.ndarray
#         Difference matrix
#     mask : numpy.ndarray, optional
#         Boolean mask to hide non-significant connections
#     title : str, optional
#         Plot title
#     save_path : str or Path, optional
#         Path to save the figure
#     cmap : str, optional
#         Colormap to use
#     center : float, optional
#         Center value for colormap
#     """
#     plt.figure(figsize=(12, 10))
    
#     # Ensure we have a mask that works with seaborn
#     if mask is None:
#         mask = np.zeros_like(diff_matrix, dtype=bool)
    
#     # Create a diverging colormap centered at zero
#     vmax = np.max(np.abs(diff_matrix))
#     vmin = -vmax
    
#     # Create the heatmap
#     print("Heatmap max value:", np.max(diff_matrix))
#     print("Significant connections:", np.sum(~mask))

#     # Fallback: if everything is masked, unmask top N connections
#     if np.all(mask):
#         print("Warning: All values masked. Unmasking top 100 connections for visualization.")
#         abs_diff = np.abs(diff_matrix)
#         threshold = np.sort(abs_diff.flatten())[-100]  # Top 100 values
#         mask = abs_diff < threshold

#     try:
#         ax = sns.heatmap(diff_matrix, mask=mask, cmap=cmap, center=center,
#                       vmin=vmin, vmax=vmax, square=True,
#                       cbar_kws={"shrink": 0.8, "label": "Difference (Effect Size)"})
        
#         # Adjust the plot
#         plt.title(title, fontsize=16)
        
#         # If we have a lot of regions, we might want to skip some tick labels
#         n_regions = diff_matrix.shape[0]
#         if n_regions > 50:
#             # Show fewer ticks
#             tick_step = max(1, n_regions // 20)
#             ax.set_xticks(np.arange(0, n_regions, tick_step))
#             ax.set_yticks(np.arange(0, n_regions, tick_step))
            
#         # Rotate x ticks to prevent overlap
#         plt.xticks(rotation=90)
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
#         return plt.gcf()
        
#     except Exception as e:
#         print(f"Error creating significance heatmap: {e}")
#         plt.figure(figsize=(10, 8))
#         plt.text(0.5, 0.5, f"Error creating heatmap: {e}", 
#                ha='center', va='center', fontsize=12)
#         if save_path:
#             plt.savefig(save_path)
#         return plt.gcf()
    
# def generate_paper_style_visualizations(comparison_results, ja_comparisons, networks, output_dir,
#                                       region_info=None, network_to_regions=None):
#     """
#     Generate visualizations similar to Buch et al. and Vandewouw et al. papers
    
#     Parameters:
#     -----------
#     comparison_results : dict
#         Results from task comparisons
#     ja_comparisons : dict
#         Configuration for joint attention comparisons
#     networks : list
#         List of network names
#     output_dir : Path or str
#         Directory to save output figures
#     region_info : pd.DataFrame, optional
#         DataFrame with region information including coordinates
#     network_to_regions : dict, optional
#         Mapping from networks to region indices
#     """
#     from pathlib import Path
#     output_dir = Path(output_dir)
    
#     # Get coordinates if region_info is provided
#     region_coords = get_mni_coordinates(region_info) if region_info is not None else None
    
#     for task, results in comparison_results.items():
#         # Create a directory for this task
#         task_dir = output_dir / task
#         os.makedirs(task_dir, exist_ok=True)
        
#         # Get the specific features for this task
#         task_features = ja_comparisons[task]['features']
        
#         for comparison_key, data in results.items():
#             print(f"Generating paper-style visualizations for {task} - {comparison_key}")
            
#             # 1. Create Chord Diagrams similar to Buch et al.
#             if 'network_diff' in data and len(networks) > 0:
#                 create_chord_diagram(
#                     data['network_diff'],
#                     networks,
#                     significant_mask=data.get('significant_network_conns'),
#                     title=f"{task}: {comparison_key} Network Connectivity",
#                     save_path=task_dir / f"{comparison_key}_chord_diagram.png"
#                 )
            
#             # 2. Create heatmap for region-to-region connectivity with significance masking
#             if 'diff_matrix' in data and 'significant_conns' in data:
#                 create_significance_heatmap(
#                     data['diff_matrix'],
#                     mask=~data['significant_conns'],  # Show only significant connections
#                     title=f"{task}: {comparison_key} Significant Differences",
#                     save_path=task_dir / f"{comparison_key}_significant_heatmap.png"
#                 )
            
#             # 3. Create effect size visualization similar to Vandewouw et al.
#             if 'effect_sizes' in data and region_coords is not None:
#                 create_effect_size_brain_rendering(
#                     data['effect_sizes'],
#                     region_coords,
#                     title=f"{task}: {comparison_key} Effect Sizes",
#                     save_path=task_dir / f"{comparison_key}_effect_size_brain.png"
#                 )
            
#             # 4. Create brain surface plot for significant connections
#             if region_coords is not None and 'significant_conns' in data and 'diff_matrix' in data:
#                 create_brain_surface_plot(
#                     data['diff_matrix'],
#                     region_coords,
#                     mask=data['significant_conns'],
#                     title=f"{task}: {comparison_key} Significant Connections",
#                     save_path=task_dir / f"{comparison_key}_surface_plot.png"
#                 )
            
#             # 5. Feature-specific visualizations with box+violin plots for distributions
#             for feature in task_features:
#                 feature_result = data['feature_results'].get(feature)
#                 if feature_result is None:
#                     print(f"Warning: No results for feature {feature}")
#                     continue
                
#                 create_enhanced_feature_comparison_plot(
#                     feature,
#                     feature_result,
#                     comparison_key,
#                     save_path=task_dir / f"{comparison_key}_{feature}_comparison.png"
#                 )
            
#             # 6. Create a subgroup connectivity visualization like Figure 2 in Buch et al.
#             create_subgroup_connectivity_comparison(
#                 data.get('cluster1_network_conn'), 
#                 data.get('cluster2_network_conn'),
#                 networks,
#                 title=f"{task}: {comparison_key} Subgroup Connectivity",
#                 save_path=task_dir / f"{comparison_key}_subgroup_connectivity.png"
#             )
            
#             # 7. Create brain network visualization with effect sizes
#             if region_coords is not None and 'significant_conns' in data and 'effect_sizes' in data:
#                 create_network_brain_visualization(
#                     data['diff_matrix'],
#                     data['significant_conns'],
#                     data['effect_sizes'],
#                     region_coords,
#                     networks,
#                     network_to_regions,
#                     title=f"{task}: {comparison_key} Brain Network Differences",
#                     save_path=task_dir / f"{comparison_key}_brain_network.png"
#                 )
    
#     print(f"Paper-style visualizations saved to {output_dir}")