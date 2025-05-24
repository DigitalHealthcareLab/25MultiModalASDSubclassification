"""
After running 1) perform_brain_comparison.py and 2) get_results_brain_stats.py, this script will visualize the results.
The results are derived from the comparison of different clusters based on their brain connectivity features. (perform_brain_comparison.py)

Results path: 
RESULTS_PATH = BASE_PATH / 'comparisons'
where inside 'comparisons' there are the following files:
Folders of pairwise comparisons:
    - 1_vs_2
    - 1_vs_3
    ...
    - 3_vs_4
all_comparisons.pkl
feature_summary.csv
seg_int_pairwise.csv
omnibus
pairwise_seg_int (folder)
"""
#%%
# Import necessary libraries
from pathlib import Path
import sys
import pickle
import numpy as np
import pandas as pd

# import ipywidgets as widgets
# from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
# Set the style for seaborn
sns.set_theme(style="white")

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from src.utils_brain_stats import compare_integration_segregation_by_cluster

# Define paths
BASE_PATH = Path('/home/data/2025_asd_multimodal/feature_brain')
RESULTS_PATH = BASE_PATH / 'comparisons'
VIZ_OUTPUT_PATH = BASE_PATH / 'visualizations'

VIZ_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

#%%
#!/usr/bin/env python
"""
Main Figure Visualization for Brain Connectivity Clusters

Generates:
  1. Bar plot of mean ± SD for integration (characteristic path length) and segregation (modularity) across clusters 1–4.
  2. Scatter plot of integration vs. segregation values per cluster, annotated.

"""
# Define a consistent color map for clusters
CLUSTER_COLORS = plt.get_cmap('tab10')


def plot_combined(cluster_averages, save_path=None):
    clusters = sorted(cluster_averages.keys(), key=int)
    n = len(clusters)

    # Extract means and stds
    int_means = np.array([cluster_averages[c]['integration_mean'] for c in clusters])
    int_stds  = np.array([cluster_averages[c]['integration_std']  for c in clusters])
    seg_means = np.array([cluster_averages[c]['segregation_mean'] for c in clusters])
    seg_stds  = np.array([cluster_averages[c]['segregation_std']  for c in clusters])

    # Set up figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Bar plot
    x = np.arange(n)
    width = 0.35
    ax1.bar(x - width/2, int_means, width, yerr=int_stds, label='Integration', color='C0')
    ax1.bar(x + width/2, seg_means, width, yerr=seg_stds, label='Segregation', color='C1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(clusters)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('A) Cluster Integration & Segregation')
    ax1.legend()

    # Panel B: Scatter plot
    for i, c in enumerate(clusters):
        xi = cluster_averages[c]['integration_mean']
        yi = cluster_averages[c]['segregation_mean']
        ax2.scatter(xi, yi, s=100, color=CLUSTER_COLORS(i), label=f'Cluster {c}')
        ax2.annotate(c, (xi, yi), textcoords="offset points", xytext=(5,5))
    ax2.set_xlabel('Integration (char path length)')
    ax2.set_ylabel('Segregation (modularity)')
    ax2.set_title('B) Integration vs Segregation by Cluster')
    ax2.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Compute or load cluster averages
    cluster_averages = compare_integration_segregation_by_cluster()

    # Visualize
    OUTPUT = Path('./cluster_main_figure.png')
    plot_combined(cluster_averages, save_path=OUTPUT)
    print(f"Saved figure to {OUTPUT}")


if __name__ == '__main__':
    main()


#%%
"""
Brain Connectivity Surface Visualization with Nilearn

This script extends the main figure by mapping network‑level effect sizes
for segregation (modularity) and integration (char_path_length) onto the fsaverage surface.

Steps:
 1. Load network_comparison.pkl for a chosen pairwise cluster comparison (e.g. '3_vs_4').
 2. Compute a network‑level effect size proxy (mean absolute t‑values) for each network.
 3. Load the Power atlas NIfTI parcellation and project network‑effect volumes to the surface.
 4. Plot segregation and integration effect maps side by side.

Dependencies:
  - numpy
  - nibabel
  - matplotlib
  - nilearn
  - get_results_brain_stats.get_atlas_mapping

Usage:
  python brain_surface_visual.py --comparison 3_vs_4

"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets

# 1) Imports from your utils
from src.utils_brain_mapping     import get_atlas_mapping
from src.utils_brain_figure      import create_network_level_brain_plot
from src.utils_config            import ATLAS_PATH

# 2) Fetch Power ROI coords (264 × 3) and ROI numbers (1–264)
power       = datasets.fetch_coords_power_2011(legacy_format=False)
rois        = power['rois']                          # pandas DataFrame
region_coords = rois[['x','y','z']].values           # shape: (264,3)
roi_numbers   = rois['roi'].astype(int).tolist()     # [1,2,…,264]

# 3) Load the region→network mapping (zero‑indexed)
region_to_network, region_df, networks, network_to_regions = get_atlas_mapping(ATLAS_PATH)
# region_to_network maps { 0: 'Visual', 1: 'Salience', … }
# network_to_regions maps each network → list of ROI indices (0–263)

# Optionally drop the 'Uncertain' network if you want exactly 13 networks
if 'Uncertain' in networks:
    networks.remove('Uncertain')
    network_to_regions.pop('Uncertain', None)

# 4) Prepare your list of comparisons
BASE        = Path('/home/data/2025_asd_multimodal/feature_brain/comparisons')
comparisons = ['1_vs_2','1_vs_3','1_vs_4','2_vs_3','2_vs_4','3_vs_4']

# 5) Loop and plot
for comp in comparisons:
    with open(BASE/comp/'network_comparison.pkl','rb') as f:
        data = pickle.load(f)

    tmat = data['tvalues']       # shape (n_networks, n_networks)
    pmat = data['pvalues']       # same shape

    sig_mask = (pmat < 0.05)     # highlight only p<.05

    fig = create_network_level_brain_plot(
        network_matrix     = tmat,
        region_coords      = region_coords,
        network_to_regions = network_to_regions,
        significant_mask   = sig_mask,
        networks           = networks,
        title              = f'Network Differences ({comp})',
        save_path          = None   # or 'figures/{comp}_network.png'
    )
    plt.show()


# %%
