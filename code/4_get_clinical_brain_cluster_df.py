'''
Extract Brain Integration and Segregation Metrics for Each Participant
=====================================================================

This script extracts the characteristic path length (integration) and modularity (segregation)
metrics for each participant, adds them to the cluster information, and combines with
clinical data to create a comprehensive dataframe for downstream analysis.

Author: Chanyoung Ko (Modified version)
Date: 04-18-2025
'''
#%%
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Import necessary utilities from your project
from src.utils_config import (
    FMRI_PATH, CLUSTER_DF_PATH, CLINICAL_DF_PATH, 
    DATAFRAMES_DIR, load_id_mapping, MULTIMODAL_DF_PATH,
    dataset_name, atlas_name, timeseries_type, connectome_type
)

def setup_paths():
    """Set up all necessary paths based on configuration."""
    # Base paths for individual results (from batch processing)
    results_base = FMRI_PATH / dataset_name / 'results'
    base = results_base / f"{atlas_name}_{timeseries_type}_{connectome_type}"
    
    # Output directory for extracted metrics
    output_dir = DATAFRAMES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create path dictionary
    paths = {
        # Input paths (individual results)
        'graph_results': base / 'graph_analyses',
        
        # Output paths
        'output': output_dir,
        
        # Cluster and clinical data
        'cluster_dataframe': CLUSTER_DF_PATH,
        'clinical_dataframe': CLINICAL_DF_PATH
    }
    
    return paths

def load_cluster_data(cluster_path, id_column='id', cluster_column='cluster'):
    """Load cluster assignments from dataframe."""
    try:
        df = pd.read_csv(cluster_path)
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in cluster dataframe")
        if cluster_column not in df.columns:
            raise ValueError(f"Cluster column '{cluster_column}' not found in cluster dataframe")
        
        # Ensure subject IDs are strings
        df[id_column] = df[id_column].astype(str)
        
        print(f"Loaded cluster data with {len(df)} participants and {df[cluster_column].nunique()} clusters")
        cluster_counts = df[cluster_column].value_counts()
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count} participants")
        
        return df
    
    except Exception as e:
        print(f"Error loading cluster data: {e}")
        return None

def load_clinical_data(clinical_path, id_column='id'):
    """Load clinical data from dataframe."""
    try:
        df = pd.read_csv(clinical_path)
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in clinical dataframe")
        
        # Ensure subject IDs are strings
        df[id_column] = df[id_column].astype(str)
        
        print(f"Loaded clinical data for {len(df)} participants")
        return df
    
    except Exception as e:
        print(f"Error loading clinical data: {e}")
        return None

def load_graph_metrics(graph_results_path, subject_ids=None):
    """
    Load graph metrics for each subject.
    
    Parameters:
    -----------
    graph_results_path : Path
        Directory containing graph metrics pickle files
    subject_ids : list, optional
        List of subject IDs to load. If None, load all subjects.
        
    Returns:
    --------
    dict
        Dictionary of graph metrics for each subject
    """
    graph_results = {}
    graph_file_pattern = "graph_metrics_*.pkl"
    
    for file_path in graph_results_path.glob(graph_file_pattern):
        subject_id = file_path.stem.split('_')[-1]
        if subject_ids is None or subject_id in subject_ids:
            try:
                with open(file_path, 'rb') as f:
                    graph_results[subject_id] = pickle.load(f)
            except Exception as e:
                print(f"Error loading graph results for {subject_id}: {e}")
    
    print(f"Loaded graph metrics for {len(graph_results)} subjects")
    
    return graph_results

def extract_integration_segregation_metrics(graph_results):
    """
    Extract characteristic path length (integration) and modularity (segregation)
    metrics for each subject.
    
    Parameters:
    -----------
    graph_results : dict
        Dictionary of graph metrics for each subject
        
    Returns:
    --------
    dict
        Dictionary with integration and segregation metrics for each subject
    """
    metrics = {}
    
    for subject_id, results in graph_results.items():
        # Check if global metrics exist
        if 'global' not in results:
            print(f"Warning: No global metrics found for subject {subject_id}")
            continue
            
        global_metrics = results['global']
        
        # Extract characteristic path length (integration)
        integration = global_metrics.get('char_path_length')
        
        # Extract modularity (segregation)
        segregation = global_metrics.get('modularity')
        
        metrics[subject_id] = {
            'brain_integration': integration,
            'brain_segregation': segregation
        }
    
    # Count how many subjects have valid metrics
    valid_integration = sum(1 for m in metrics.values() if m['brain_integration'] is not None)
    valid_segregation = sum(1 for m in metrics.values() if m['brain_segregation'] is not None)
    print(f"Extracted valid integration metrics for {valid_integration}/{len(metrics)} subjects")
    print(f"Extracted valid segregation metrics for {valid_segregation}/{len(metrics)} subjects")
    
    return metrics

def create_comprehensive_dataframe(cluster_df, clinical_df, metrics, id_to_fmri, id_column='id'):
    """
    Create a comprehensive dataframe with cluster, clinical, and brain metrics data.
    
    Parameters:
    -----------
    cluster_df : DataFrame
        Dataframe with cluster assignments
    clinical_df : DataFrame
        Dataframe with clinical data
    metrics : dict
        Dictionary with integration and segregation metrics for each subject
    id_to_fmri : dict
        Mapping from participant IDs to fMRI IDs
    id_column : str, default='id'
        Column name for participant IDs
        
    Returns:
    --------
    DataFrame
        Comprehensive dataframe with all data
    """
    # First, merge cluster and clinical data
    merged_df = pd.merge(clinical_df, cluster_df, on=id_column, how='left')
    print(f"Merged dataframe has {len(merged_df)} rows")
    
    # Prepare brain metrics dataframe
    metrics_data = []
    for participant_id, fmri_id in id_to_fmri.items():
        if fmri_id in metrics:
            metrics_data.append({
                id_column: participant_id,
                'fmri_id': fmri_id,
                'brain_integration': metrics[fmri_id]['brain_integration'],
                'brain_segregation': metrics[fmri_id]['brain_segregation']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    print(f"Created metrics dataframe with {len(metrics_df)} rows")
    
    # Merge with brain metrics
    final_df = pd.merge(merged_df, metrics_df, on=id_column, how='left')
    print(f"Final dataframe has {len(final_df)} rows")
    
    # Check for missing values in key columns
    missing_cluster = final_df['cluster'].isna().sum()
    missing_integration = final_df['brain_integration'].isna().sum()
    missing_segregation = final_df['brain_segregation'].isna().sum()
    
    print(f"Missing values: cluster: {missing_cluster}, integration: {missing_integration}, segregation: {missing_segregation}")
    
    return final_df

def main():
    """Main execution function."""
    # Set up paths
    paths = setup_paths()
    print("\nUsing paths:")
    print(f"Graph results path: {paths['graph_results']}")
    print(f"Cluster dataframe path: {paths['cluster_dataframe']}")
    print(f"Clinical dataframe path: {paths['clinical_dataframe']}")
    print(f"Output directory: {paths['output']}")
    
    # Load cluster data
    cluster_df = load_cluster_data(paths['cluster_dataframe'])
    if cluster_df is None:
        print("Failed to load cluster data. Exiting.")
        return 1
    
    # Load clinical data
    clinical_df = load_clinical_data(paths['clinical_dataframe'])
    if clinical_df is None:
        print("Failed to load clinical data. Exiting.")
        return 1
    
    # Load ID mapping between participant IDs and fMRI IDs
    id_to_fmri = load_id_mapping(MULTIMODAL_DF_PATH)
    if not id_to_fmri:
        print("Failed to load ID mapping. Exiting.")
        return 1
    
    # Get list of fMRI IDs
    fmri_ids = list(id_to_fmri.values())
    print(f"Found {len(fmri_ids)} fMRI IDs for analysis")
    
    # Load graph metrics
    graph_results = load_graph_metrics(paths['graph_results'], fmri_ids)
    
    # Extract integration and segregation metrics
    metrics = extract_integration_segregation_metrics(graph_results)
    
    # Create comprehensive dataframe
    final_df = create_comprehensive_dataframe(cluster_df, clinical_df, metrics, id_to_fmri)
    
    # Save the final dataframe
    output_file = paths['output'] / 'participant_clinical_brain_cluster_df.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved comprehensive dataframe to {output_file}")
    
    # Also save a clean version with only essential columns
    essential_columns = ['id', 'cluster', 'brain_integration', 'brain_segregation', 'fmri_id']
    
    # Add other important columns from clinical_df if they exist
    for col in ['age', 'sex', 'diagnosis', 'IQ', 'ADOS']:
        if col in final_df.columns:
            essential_columns.append(col)
    
    # Create the clean dataframe with only essential columns that exist
    existing_columns = [col for col in essential_columns if col in final_df.columns]
    clean_df = final_df[existing_columns]
    
    # Save the clean dataframe
    clean_output_file = paths['output'] / 'participant_brain_metrics.csv'
    clean_df.to_csv(clean_output_file, index=False)
    print(f"Saved clean dataframe to {clean_output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics for Brain Metrics:")
    print("\nClustered Participants:")
    for cluster in clean_df['cluster'].dropna().unique():
        cluster_data = clean_df[clean_df['cluster'] == cluster]
        print(f"\nCluster {cluster} (n={len(cluster_data)}):")
        
        # Integration metrics
        integration_values = cluster_data['brain_integration'].dropna()
        if len(integration_values) > 0:
            print(f"  Integration (char_path_length):")
            print(f"    Mean: {integration_values.mean():.4f}")
            print(f"    Std: {integration_values.std():.4f}")
            print(f"    Min: {integration_values.min():.4f}")
            print(f"    Max: {integration_values.max():.4f}")
        else:
            print(f"  Integration: No valid data")
        
        # Segregation metrics
        segregation_values = cluster_data['brain_segregation'].dropna()
        if len(segregation_values) > 0:
            print(f"  Segregation (modularity):")
            print(f"    Mean: {segregation_values.mean():.4f}")
            print(f"    Std: {segregation_values.std():.4f}")
            print(f"    Min: {segregation_values.min():.4f}")
            print(f"    Max: {segregation_values.max():.4f}")
        else:
            print(f"  Segregation: No valid data")
    
    return 0

if __name__ == "__main__":
    main()

#%%
# Open and load this new df (csv file) and to check header
df = pd.read_csv(os.path.join(DATAFRAMES_DIR, 'participant_clinical_brain_cluster_df.csv'))
print(df.head(10))

# %%
df_brain = pd.read_csv(os.path.join(DATAFRAMES_DIR, 'participant_brain_metrics.csv'))
print(df_brain.head(10))
# %%
