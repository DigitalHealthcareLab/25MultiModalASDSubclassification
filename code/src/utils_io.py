'''
I/O utility functions for multimodal clustering
'''
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_canonical_variates(gaze_path, fmri_path):
    """
    Load the canonical variates for both modalities
    
    Parameters:
    -----------
    gaze_path : Path
        Path to the directory containing gaze canonical variates
    fmri_path : Path
        Path to the directory containing fMRI canonical variates
        
    Returns:
    --------
    gaze_cvs : DataFrame
        DataFrame containing gaze canonical variates
    fmri_cvs : DataFrame
        DataFrame containing fMRI canonical variates
    """
    print("\n===== Loading Canonical Variates =====")
    
    # Load gaze canonical variates
    gaze_cv_path = Path(gaze_path, "gaze_canonical_variates.csv")
    if gaze_cv_path.exists():
        gaze_cvs = pd.read_csv(gaze_cv_path, index_col=0)
        print(f"Loaded gaze canonical variates: {gaze_cvs.shape}")
    else:
        raise FileNotFoundError(f"Gaze canonical variates not found at {gaze_cv_path}")
    
    # Load fMRI canonical variates
    fmri_cv_path = Path(fmri_path, "fmri_canonical_variates.csv")
    if fmri_cv_path.exists():
        fmri_cvs = pd.read_csv(fmri_cv_path, index_col=0)
        print(f"Loaded fMRI canonical variates: {fmri_cvs.shape}")
    else:
        raise FileNotFoundError(f"fMRI canonical variates not found at {fmri_cv_path}")
    
    # Load canonical correlations for reference
    gaze_corr_path = Path(gaze_path, "gaze_canonical_correlations.csv")
    fmri_corr_path = Path(fmri_path, "fmri_canonical_correlations.csv")
    
    if gaze_corr_path.exists() and fmri_corr_path.exists():
        gaze_corrs = pd.read_csv(gaze_corr_path)
        fmri_corrs = pd.read_csv(fmri_corr_path)
        print("\nCanonical correlations:")
        print("Gaze:")
        print(gaze_corrs)
        print("\nfMRI:")
        print(fmri_corrs)
    
    # Load canonical-JA relationships for reference
    gaze_ja_path = Path(gaze_path, "gaze_canonical_ja_relationship.csv")
    fmri_ja_path = Path(fmri_path, "fmri_canonical_ja_relationship.csv")
    
    if gaze_ja_path.exists() and fmri_ja_path.exists():
        gaze_ja = pd.read_csv(gaze_ja_path, index_col=0)
        fmri_ja = pd.read_csv(fmri_ja_path, index_col=0)
        print("\nJoint Attention Relationships:")
        print("Gaze:")
        print(gaze_ja)
        print("\nfMRI:")
        print(fmri_ja)
    
    return gaze_cvs, fmri_cvs

def merge_canonical_variates(gaze_cvs, fmri_cvs, output_path):
    """
    Merge canonical variates from both modalities
    
    Parameters:
    -----------
    gaze_cvs : DataFrame
        DataFrame containing gaze canonical variates
    fmri_cvs : DataFrame
        DataFrame containing fMRI canonical variates
    output_path : Path
        Path to save the merged canonical variates
        
    Returns:
    --------
    combined_cvs_scaled : DataFrame
        DataFrame containing standardized combined canonical variates
    """
    print("\n===== Merging Canonical Variates =====")

    # Check if modalities have the same number of subjects - if there are missing subjects in either modality
    print("Gaze subject IDs:", sorted(list(gaze_cvs.index)))
    print("fMRI subject IDs:", sorted(list(fmri_cvs.index)))
    
    # Check for common subjects
    gaze_subjects = set(gaze_cvs.index)
    fmri_subjects = set(fmri_cvs.index)
    common_subjects = gaze_subjects.intersection(fmri_subjects)
    
    print(f"Found {len(common_subjects)} subjects common to both modalities")
    print(f"Gaze-only subjects: {len(gaze_subjects - common_subjects)}")
    print(f"fMRI-only subjects: {len(fmri_subjects - common_subjects)}")
    
    # Filter to common subjects
    # gaze_common = gaze_cvs.loc[list(common_subjects)]
    # fmri_common = fmri_cvs.loc[list(common_subjects)]
    sorted_subjects = sorted(common_subjects)
    gaze_common = gaze_cvs.loc[sorted_subjects]
    fmri_common = fmri_cvs.loc[sorted_subjects]
    
    # Verify subjects are in the same order
    if not all(gaze_common.index == fmri_common.index):
        # Reindex to ensure matching order
        fmri_common = fmri_common.loc[gaze_common.index]

    # Ensuring consistent column names    
    gaze_common.columns = [f"gaze_{col}" for col in gaze_common.columns]
    fmri_common.columns = [f"fmri_{col}" for col in fmri_common.columns]

    # Merge canonical variates
    combined_cvs = pd.concat([gaze_common, fmri_common], axis=1)
    print(f"Combined canonical variates shape: {combined_cvs.shape}")
    print("[DEBUG] First 5 subject IDs after merge:", combined_cvs.index[:5].tolist())
    
    # Standardize the combined features
    scaler = StandardScaler()
    combined_cvs_scaled = pd.DataFrame(
        scaler.fit_transform(combined_cvs),
        columns=combined_cvs.columns,
        index=combined_cvs.index
    )
    
    # Save the combined canonical variates
    combined_path = Path(output_path, "combined_canonical_variates.csv")
    combined_cvs_scaled.to_csv(combined_path)
    print(f"Saved combined canonical variates to: {combined_path}")
    
    return combined_cvs_scaled

def load_clinical_data(clinical_path, subject_ids):
    """
    Load clinical data for the subjects in the combined dataset
    
    Parameters:
    -----------
    clinical_path : Path
        Path to the clinical data file
    subject_ids : list or array-like
        List of subject IDs in the combined dataset
        
    Returns:
    --------
    clinical_data : DataFrame
        DataFrame containing clinical data for the subjects
    """
    print("\n===== Loading Clinical Data =====")
    
    # Load clinical data
    if clinical_path.exists():
        clinical_data = pd.read_csv(clinical_path)
        print(f"Loaded clinical data: {clinical_data.shape}")
        
        # Ensure subject ID column is present
        id_columns = [col for col in clinical_data.columns if 'id' in col.lower()]
        if not id_columns:
            raise ValueError("Could not identify subject ID column in clinical data")
        
        id_col = id_columns[0]
        print(f"Using '{id_col}' as subject ID column")
        
        # Filter to subjects in the combined dataset
        clinical_data = clinical_data[clinical_data[id_col].isin(subject_ids)]
        print(f"Filtered to {len(clinical_data)} subjects in combined dataset")
        
        return clinical_data
    else:
        print(f"Clinical data not found at {clinical_path}")
        return None

def save_cluster_results(cluster_labels, combined_cvs, clinical_data=None, output_path=None, prefix='cluster_results'):
    """
    Save cluster results with optional clinical data

    Parameters:
    -----------
    cluster_labels : array-like
        Cluster labels for each subject (1-based)
    combined_cvs : DataFrame
        DataFrame containing combined canonical variates
    clinical_data : DataFrame, optional
        DataFrame containing clinical data
    output_path : Path, optional
        Path to save the results
    prefix : str, optional
        Prefix for the output files

    Returns:
    --------
    results_df : DataFrame
        DataFrame containing cluster assignments and combined canonical variates
    """
    print("\n===== Saving Cluster Results =====")

    # Copy combined CVs and add cluster labels
    results_df = combined_cvs.copy()
    results_df['cluster'] = cluster_labels  # assumed to already be 1-based

    # Make index into a column named 'id'
    if results_df.index.name:
        results_df = results_df.reset_index()
        results_df = results_df.rename(columns={results_df.columns[0]: 'id'})
    else:
        results_df.insert(0, 'id', results_df.index)

    # Add clinical data if provided
    if clinical_data is not None:
        clinical_data_copy = clinical_data.copy()
        id_cols = [col for col in clinical_data_copy.columns if 'id' in col.lower()]
        if id_cols:
            id_col = id_cols[0]
            if id_col != 'id':
                clinical_data_copy = clinical_data_copy.rename(columns={id_col: 'id'})
            results_df = pd.merge(results_df, clinical_data_copy, on='id', how='left')
            print(f"Found {results_df['id'].notna().sum()} subjects with clinical data")
        else:
            print("Warning: No ID column found in clinical data. Skipping merge.")

    # Save to disk
    if output_path is not None:
        output_file = Path(output_path, f"{prefix}.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Saved cluster results to: {output_file}")

    return results_df


######Brain Network Analysis Results Saving Functions###### 
def save_results(results, paths, format='pkl'):
    if format not in ['pkl', 'npy']:
        raise ValueError(f"Invalid format: {format}. Choose either 'pkl' or 'npy'.")

    if 'fnc' in results:
        save_fnc_results(results['fnc'], paths['fnc_results'], format)
    if 'group_differences' in results:
        save_group_difference_results(results['group_differences'], paths['stats_results'], format)
    if 'graph_metrics' in results:
        save_graph_metric_results(results['graph_metrics'], paths['graph_results'], format)
    if 'stats' in results:
        save_statistical_results(results['stats'], paths['stats_results'], format)
    if 'nbs' in results:
        save_nbs_results(results['nbs'], paths['nbs_results'], format)

    save_summary(results, paths, format)

def save_fnc_results(fnc_results, output_path, format='pkl'):
    node_wise_dir = output_path / 'node_wise'
    edge_wise_dir = output_path / 'edge_wise'

    node_wise_dir.mkdir(parents=True, exist_ok=True)
    edge_wise_dir.mkdir(parents=True, exist_ok=True)

    if 'node_wise' in fnc_results:
        for subject_id, node_data in fnc_results['node_wise'].items():
            filename = f"node_wise_fnc_{subject_id}"
            save_data(node_data, node_wise_dir / filename, format)

    if 'edge_wise' in fnc_results:
        for subject_id, edge_data in fnc_results['edge_wise'].items():
            filename = f"edge_wise_fnc_{subject_id}"
            save_data(edge_data, edge_wise_dir / filename, format)

    for key, value in fnc_results.items():
        if key not in ['node_wise', 'edge_wise']:
            save_data(value, output_path / f"{key}", format)

def save_group_difference_results(group_diff_results, output_path, format='pkl'):
    group_diff_dir = output_path / 'group_differences'
    group_diff_dir.mkdir(parents=True, exist_ok=True)

    for metric_type, data in group_diff_results.items():
        metric_dir = group_diff_dir / metric_type
        metric_dir.mkdir(exist_ok=True)
        for measure_name, measure_data in data.items():
            filename = f"{metric_type}_{measure_name}"
            save_data(measure_data, metric_dir / filename, format)

def save_graph_metric_results(graph_results, output_path, format='pkl'):
    if 'global' in graph_results:
        for metric_name, metric_data in graph_results['global'].items():
            if isinstance(metric_data, dict):
                for subject_id, subject_data in metric_data.items():
                    filename = f"global_{metric_name}_{subject_id}"
                    save_data(subject_data, output_path / filename, format)
            else:
                save_data(metric_data, output_path / f"global_{metric_name}", format)

    if 'nodal' in graph_results:
        for metric_name, metric_data in graph_results['nodal'].items():
            if isinstance(metric_data, dict):
                for subject_id, subject_data in metric_data.items():
                    filename = f"nodal_{metric_name}_{subject_id}"
                    save_data(subject_data, output_path / filename, format)
            else:
                save_data(metric_data, output_path / f"nodal_{metric_name}", format)

def save_statistical_results(stats_results, output_path, format='pkl'):
    stats_dir = output_path / 'statistical_tests'
    stats_dir.mkdir(parents=True, exist_ok=True)

    for test_type, test_data in stats_results.items():
        type_dir = stats_dir / test_type
        type_dir.mkdir(exist_ok=True)
        if isinstance(test_data, dict):
            for measure_name, measure_results in test_data.items():
                filename = f"{test_type}_{measure_name}"
                save_data(measure_results, type_dir / filename, format)
        else:
            save_data(test_data, type_dir / f"{test_type}", format)

def save_nbs_results(nbs_results, output_path, format='pkl'):
    for component_name, component_data in nbs_results.items():
        filename = f"nbs_{component_name}"
        save_data(component_data, output_path / filename, format)

def save_summary(results, paths, format='pkl'):
    summary = {
        'analyses_performed': list(results.keys()),
        'num_subjects': len(results['fnc']['node_wise']) if 'fnc' in results and 'node_wise' in results['fnc'] else None,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    for path_name, path in paths.items():
        if 'results' in path_name:
            with open(path / 'analysis_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)

def save_data(data, filepath, format='pkl'):
    filepath = Path(f"{filepath}.{format}") if not str(filepath).endswith(f'.{format}') else Path(filepath)
    if filepath.exists():
        logger.warning(f"Overwriting existing file: {filepath}")

    try:
        if format == 'pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'npy':
            if isinstance(data, (np.ndarray, list, tuple)):
                np.save(filepath, data)
            elif isinstance(data, pd.DataFrame):
                data.to_pickle(filepath)
            else:
                fallback_path = filepath.with_suffix('.pkl')
                with open(fallback_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.warning(f"Unsupported type {type(data)} saved as pickle at {fallback_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}")
        raise

def load_results(results_path, format='pkl'):
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results path does not exist: {results_path}")

    results = {}
    fnc_path = results_path / 'fnc_analyses'
    if fnc_path.exists():
        results['fnc'] = load_directory_results(fnc_path, format)
    graph_path = results_path / 'graph_analyses'
    if graph_path.exists():
        results['graph_metrics'] = load_directory_results(graph_path, format)
    stats_path = results_path / 'stats_analyses'
    if stats_path.exists():
        results['stats'] = load_directory_results(stats_path, format)
    nbs_path = results_path / 'nbs_analyses'
    if nbs_path.exists():
        results['nbs'] = load_directory_results(nbs_path, format)

    return results

def load_directory_results(directory_path, format='pkl'):
    directory_path = Path(directory_path)
    results = {}

    for path in directory_path.glob(f'*.{format}'):
        key = path.stem
        results[key] = load_data(path, format)

    for sub in ['node_wise', 'edge_wise', 'network_wise', 'group_differences']:
        sub_path = directory_path / sub
        if sub_path.exists():
            results[sub] = {}
            for path in sub_path.glob(f'*.{format}'):
                subject_id = path.stem.split('_')[-1]
                results[sub][subject_id] = load_data(path, format)

    return results

def load_data(filepath, format='pkl'):
    filepath = Path(filepath)
    try:
        if format == 'pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'npy':
            return np.load(filepath, allow_pickle=True)
    except Exception as e:
        logger.warning(f"Fallback to pickle due to error loading {filepath}: {e}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def convert_format(input_path, output_path, input_format='pkl', output_format='npy'):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    results = load_results(input_path, format=input_format)
    save_results(results, {'all_results': output_path}, format=output_format)
    logger.info(f"Results converted from {input_format} to {output_format} and saved to {output_path}")
