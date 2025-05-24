'''
Configuration settings for the cluster analysis project.
Contains path definitions and other global settings.
'''
from pathlib import Path
import pandas as pd

# Multimodal clustering main directory
ROOT_PATH = Path('/home/data/2025_asd_multimodal')

# Input directories
CLUSTER_DIR = ROOT_PATH / "clustering_results"
DATAFRAMES_DIR = ROOT_PATH / "dataframes"

# Input file paths
CLUSTER_DF_PATH = CLUSTER_DIR / "combined_cvs_with_clusters_4.csv"
CLINICAL_DF_PATH = DATAFRAMES_DIR / "participant_npt_df(20250403).csv"
MULTIMODAL_DF_PATH = DATAFRAMES_DIR / "participant_multimodal_df(20250103).csv"

# Output directories
OUTPUT_TABLES_DIR = ROOT_PATH / "tables"
OUTPUT_FIGURES_DIR = ROOT_PATH / "figures"
OUTPUT_BRAIN_DIR = ROOT_PATH / "feature_brain"
OUTPUT_GAZE_DIR = ROOT_PATH / "feature_gaze"
OUTPUT_GENE_DIR = ROOT_PATH / "feature_gene"

# FMRI data directory
FMRI_PATH = Path('/home/data/2025_asd_fmri')

# Analysis parameters
dataset_name = "snu"  # snu, abide
atlas_name = "power"  # power, aal
timeseries_type = "timeseries_nogsr"  # nogsr, gsr
connectome_type = "full_correlation"  # full_correlation, partial_correlation


# Build the fMRI features path dynamically based on parameters
CONNECTOMES_PATH = Path(
    FMRI_PATH,
    dataset_name,
    'proc_data_fmap',
    atlas_name,
    timeseries_type,
    'connectome',
    connectome_type
)
# Print the connectome path for debugging
print(f"Connectome path: {CONNECTOMES_PATH}")


# Atlas path
ATLAS_PATH = Path(
    FMRI_PATH,
    'atlases',
    f'{atlas_name}_atlas.xlsx',
 )
# Print the atlas path for debugging
print(f"Atlas path: {ATLAS_PATH}")


def load_id_mapping(mapping_path):
    """Load the mapping between participant IDs and fMRI IDs."""
    try:
        mapping_df = pd.read_csv(mapping_path)
        # Ensure the required columns exist
        if 'id' not in mapping_df.columns or 'fmri_id' not in mapping_df.columns:
            print(f"Warning: Required columns ('id' and 'fmri_id') not found in mapping dataframe")
            return {}
            
        # Create a mapping dictionary from 'id' to 'fmri_id'
        mapping_df['id'] = mapping_df['id'].astype(str)
        mapping_df['fmri_id'] = mapping_df['fmri_id'].astype(str)
        id_to_fmri = dict(zip(mapping_df['id'], mapping_df['fmri_id']))
        
        print(f"Loaded ID mapping for {len(id_to_fmri)} participants")
        
        # Print a few examples to verify
        sample_ids = list(id_to_fmri.keys())[:3]
        for sample_id in sample_ids:
            print(f"  Sample mapping: {sample_id} -> {id_to_fmri[sample_id]}")
            
        return id_to_fmri
    except Exception as e:
        print(f"Error loading ID mapping: {e}")
        return {}