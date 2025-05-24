'''
Utility functions for loading and processing brain connectivity data

load_connectomes
load_timeseries
load_graph_metrics - don't have this yet
'''

import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_connectomes(connectome_path: Path, subject_id_list: list):
    """
    Load connectome data for specified subject IDs.

    Parameters
    ----------
    connectome_path : Path
        Path to the directory containing connectome files
    subject_id_list : list of str
        List of subject IDs to load (should match suffix of filenames)

    Returns
    -------
    connectomes : list of np.ndarray
    subject_ids : list of str
    subject_to_idx : dict mapping subject ID to index in list
    """
    connectomes = []
    subject_ids = []
    subject_to_idx = {}

    for idx, subject_id in enumerate(subject_id_list):
        filename = f"connectome_{subject_id}.npy"
        filepath = connectome_path / filename
        if filepath.exists():
            try:
                conn = np.load(filepath)
                # Zero out diagonal
                np.fill_diagonal(conn, 0)
                connectomes.append(conn)
                subject_ids.append(subject_id)
                subject_to_idx[subject_id] = idx
            except Exception as e:
                logger.warning(f"Could not load {filepath}: {e}")
        else:
            logger.warning(f"Connectome file not found: {filepath}")

    logger.info(f"Number of connectomes loaded: {len(connectomes)}")
    return connectomes, subject_ids, subject_to_idx


def load_timeseries(timeseries_path: Path, subject_id_list: list):
    """
    Load time series data for specified subject IDs.

    Parameters
    ----------
    timeseries_path : Path
        Path to the directory containing time series files
    subject_id_list : list of str
        List of subject IDs to load

    Returns
    -------
    timeseries_data : dict mapping subject ID to np.ndarray
    """
    timeseries_data = {}

    for subject_id in subject_id_list:
        filename = f"timeseries_{subject_id}.npy"
        filepath = timeseries_path / filename
        if filepath.exists():
            try:
                ts = np.load(filepath)
                timeseries_data[subject_id] = ts
            except Exception as e:
                logger.warning(f"Could not load time series for {subject_id}: {e}")
        else:
            logger.warning(f"Time series file not found: {filepath}")

    logger.info(f"Number of time series loaded: {len(timeseries_data)}")
    return timeseries_data


# Modified version of load_timeseries to load only specific subjects
def load_timeseries_subset(timeseries_path: Path, subject_id_list: list):
    """
    Load time series data only for specific subjects
    
    Parameters:
    -----------
    timeseries_path : Path
        Path to the directory containing time series files
    subject_ids : list
        List of subject IDs to load
        
    Returns:
    --------
    dict : Dictionary mapping subject IDs to time series matrices
    """
    timeseries_data = {}
    
    for subject_id in subject_id_list:
        filename = f"timeseries_{subject_id}.npy"
        filepath = timeseries_path / filename
        if filepath.exists():
            try:
                ts = np.load(filepath)
                timeseries_data[subject_id] = ts
            except Exception as e:
                logger.warning(f"Could not load subset time series for {subject_id}: {e}")
        else:
            logger.warning(f"Subset time series file not found: {filepath}")

    logger.info(f"Loaded subset time series for {len(timeseries_data)} subjects")
    return timeseries_data

