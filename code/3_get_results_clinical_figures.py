"""
Date created: 04-10-2025
Author: Chanyoung Ko
Objective: Create advanced visualizations for cluster characteristics comparison
          inspired by Buch et al. (2023) Nature Neuroscience paper
"""

#%%
import os
import matplotlib.pyplot as plt

from src.utils_config import OUTPUT_TABLES_DIR, OUTPUT_FIGURES_DIR
from src.utils_results_visual import (
    load_data,
    create_clinical_profile_heatmap,
    create_key_boxplots,
    create_kernel_density_plots
)

def main():
    """Main function to create all visualizations."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(OUTPUT_FIGURES_DIR, 'cluster_visualization')
    boxplots_dir = os.path.join(output_dir, 'boxplots')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(boxplots_dir, exist_ok=True)
    
    print(f"Visualization results will be saved to: {output_dir}")
    
    # Set common plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Load data
    df = load_data(OUTPUT_TABLES_DIR)
    
    # Create individual visualizations - excluding hierarchical dendrogram
    create_clinical_profile_heatmap(
        df, 
        os.path.join(output_dir, 'clinical_profile_heatmap.png')
    )
    
    # Now using boxplots_dir for individual domain boxplots
    create_key_boxplots(
        df, 
        boxplots_dir
    )
    
    create_kernel_density_plots(
        df, 
        os.path.join(output_dir, 'kernel_density_plots.png')
    )
    
    print("All visualizations completed successfully!")

if __name__ == "__main__":
    main()

# %%
