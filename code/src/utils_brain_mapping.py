'''
Utility functions for brain mapping
Support for multiple atlas types with both network mapping and coordinates
'''
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from nilearn import datasets
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: nilearn not available. Some atlas functions will be limited.")

def get_atlas_mapping(atlas_path):
    """
    Creates a mapping between atlas regions and their corresponding networks
    
    Parameters:
    -----------
    atlas_path : str or Path
        Path to the atlas file containing region-to-network mappings
        
    Returns:
    --------
    dict : Dictionary with region numbers as keys and network names as values
    pd.DataFrame : DataFrame containing all region information
    list : List of unique network names
    dict : Dictionary with network names as keys and lists of region indices as values
    """
    atlas_file = str(atlas_path).lower()
    
    # Determine parser based on filename
    if 'power' in atlas_file:
        return _parse_power_atlas(atlas_path)
    elif 'cc200' in atlas_file:
        return _parse_cc200_atlas(atlas_path)
    elif '300roi' in atlas_file:
        return _parse_300roi_atlas(atlas_path)
    else:
        # Default case - try a generic parser
        return _parse_generic_atlas(atlas_path)

def get_atlas_coords(atlas_name, atlas_dir=None):
    """
    Get coordinates for atlas regions
    
    Parameters:
    -----------
    atlas_name : str
        Name of the atlas ('power', 'cc200', '300roi')
    atlas_dir : str or Path, optional
        Directory containing atlas files, if needed
        
    Returns:
    --------
    numpy.ndarray : Array of shape (n_regions, 3) with x, y, z coordinates for each region
    """
    atlas_name = atlas_name.lower()
    
    if atlas_name == "power":
        if not NILEARN_AVAILABLE:
            raise ImportError("nilearn is required to fetch Power atlas coordinates")
        data_power = datasets.fetch_coords_power_2011(legacy_format=False)
        return np.vstack((data_power.rois['x'], data_power.rois['y'], data_power.rois['z'])).T
    
    elif atlas_name == "cc200":
        if atlas_dir is None:
            raise ValueError("atlas_dir must be provided for CC200 atlas")
        csv_path = Path(atlas_dir) / 'cc200_atlas.csv'
        df = pd.read_csv(csv_path)
        return df[['x', 'y', 'z']].values
    
    elif atlas_name == "300roi":
        if atlas_dir is None:
            raise ValueError("atlas_dir must be provided for 300ROI atlas")
        csv_path = Path(atlas_dir) / '300roi_atlas.csv'
        df = pd.read_csv(csv_path, sep=r'\s+', header=0)
        return df[['x', 'y', 'z']].values
    
    else:
        raise ValueError(f"Atlas {atlas_name} not supported for coordinate retrieval.")

# Individual parsers for different atlas formats
def _parse_power_atlas(excel_path):
    """Parses Power atlas Excel file, preserving 'Uncertain' as its own network."""
    # Read the Excel file without headers and all columns as strings
    df = pd.read_excel(excel_path, header=None, dtype=str)
    data_df = df.iloc[1:]  # skip header row

    clean_data = []
    for idx, row in data_df.iterrows():
        try:
            roi = row.iloc[0]
            if not pd.isna(roi) and roi.strip().isdigit():
                roi = int(roi)
                clean_data.append({
                    'Region_Number': roi,
                    'Network':       row.iloc[36],  # Suggested System
                    'MNI_X':         float(row.iloc[6]) if not pd.isna(row.iloc[6]) else None,
                    'MNI_Y':         float(row.iloc[7]) if not pd.isna(row.iloc[7]) else None,
                    'MNI_Z':         float(row.iloc[8]) if not pd.isna(row.iloc[8]) else None
                })
        except (ValueError, TypeError, IndexError) as e:
            print(f"Error processing row {idx}: {e}")
            continue

    region_info = pd.DataFrame(clean_data)
    # 0‑index the region numbers
    region_to_network = dict(zip(region_info['Region_Number'] - 1,
                                 region_info['Network']))

    # Build list of *all* networks seen (including 'Uncertain')
    all_networks = [net for net in region_info['Network'].unique() if not pd.isna(net)]
    # Sort them except 'Uncertain', then append 'Uncertain' at end if present
    networks = sorted([net for net in all_networks if net != 'Uncertain'])
    if 'Uncertain' in all_networks:
        networks.append('Uncertain')

    # Map each network to its list of zero‑indexed regions
    network_to_regions = {
        net: (region_info[region_info['Network'] == net]['Region_Number'].values - 1).tolist()
        for net in networks
    }

    return region_to_network, region_info, networks, network_to_regions

def _parse_cc200_atlas(atlas_path):
    """Parse CC200 atlas file"""
    try:
        # Adjust based on actual format
        df = pd.read_csv(atlas_path)
        
        # Assuming columns: Region_Number, Region_Name, Network, x, y, z
        region_info = df
        
        # Create mapping dict (0-indexed)
        region_to_network = dict(zip(region_info['Region_Number'] - 1, region_info['Network']))
        
        # Get unique networks
        networks = sorted([n for n in region_info['Network'].unique() if not pd.isna(n)])
        
        # Map networks to regions
        network_to_regions = {}
        for network in networks:
            regions = region_info[region_info['Network'] == network]['Region_Number'].values - 1
            network_to_regions[network] = regions.tolist()
        
        return region_to_network, region_info, networks, network_to_regions
    
    except Exception as e:
        print(f"Error parsing CC200 atlas: {e}")
        return {}, pd.DataFrame(), [], {}

def _parse_300roi_atlas(atlas_path):
    """Parse 300ROI atlas file"""
    try:
        # Adjust based on actual format
        df = pd.read_csv(atlas_path, sep=r'\s+', header=0)
        
        # Assuming columns: Region_Number, Region_Name, Network, x, y, z
        region_info = df
        
        # Create mapping dict (0-indexed)
        region_to_network = dict(zip(region_info['Region_Number'] - 1, region_info['Network']))
        
        # Get unique networks
        networks = sorted([n for n in region_info['Network'].unique() if not pd.isna(n)])
        
        # Map networks to regions
        network_to_regions = {}
        for network in networks:
            regions = region_info[region_info['Network'] == network]['Region_Number'].values - 1
            network_to_regions[network] = regions.tolist()
        
        return region_to_network, region_info, networks, network_to_regions
    
    except Exception as e:
        print(f"Error parsing 300ROI atlas: {e}")
        return {}, pd.DataFrame(), [], {}

def _parse_generic_atlas(atlas_path):
    """Generic atlas parser for standard format"""
    # Implementation as before, or simplified for your needs...
    try:
        # Try different file formats
        if str(atlas_path).endswith('.xlsx') or str(atlas_path).endswith('.xls'):
            df = pd.read_excel(atlas_path)
        elif str(atlas_path).endswith('.csv'):
            df = pd.read_csv(atlas_path)
        elif str(atlas_path).endswith('.tsv'):
            df = pd.read_csv(atlas_path, sep='\t')
        else:
            print(f"Unsupported atlas file format: {atlas_path}")
            return {}, pd.DataFrame(), [], {}
        
        # Try to identify key columns
        required_cols = ['Region_Number', 'Network']
        
        # Check if our required columns exist or try to map them
        col_mapping = {}
        for req_col in required_cols:
            # Check for exact match
            if req_col in df.columns:
                col_mapping[req_col] = req_col
            # Check for case-insensitive match
            elif any(col.lower() == req_col.lower() for col in df.columns):
                matching_col = next(col for col in df.columns if col.lower() == req_col.lower())
                col_mapping[req_col] = matching_col
            # Check for similar names
            elif req_col == 'Region_Number' and any(col in df.columns for col in ['ROI', 'Index', 'ID']):
                for alt in ['ROI', 'Index', 'ID']:
                    if alt in df.columns:
                        col_mapping[req_col] = alt
                        break
            elif req_col == 'Network' and any(col in df.columns for col in ['System', 'Module', 'Community']):
                for alt in ['System', 'Module', 'Community']:
                    if alt in df.columns:
                        col_mapping[req_col] = alt
                        break
        
        # If we couldn't find the required columns, try using positional mapping
        if len(col_mapping) < len(required_cols):
            print(f"Could not identify all required columns in atlas file: {atlas_path}")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"Using positional mapping for: {[col for col in required_cols if col not in col_mapping]}")
            
            # Assume first column is Region_Number if not mapped
            if 'Region_Number' not in col_mapping:
                col_mapping['Region_Number'] = df.columns[0]
            
            # Assume second column might be name, and third might be network
            if 'Network' not in col_mapping and len(df.columns) > 2:
                col_mapping['Network'] = df.columns[2]
            # If only two columns, assume second is network
            elif 'Network' not in col_mapping and len(df.columns) > 1:
                col_mapping['Network'] = df.columns[1]
        
        # Create standardized dataframe
        region_info = pd.DataFrame({
            'Region_Number': df[col_mapping['Region_Number']],
            'Network': df[col_mapping['Network']]
        })
        
        # Create mapping dict (0-indexed)
        region_to_network = dict(zip(region_info['Region_Number'] - 1, region_info['Network']))
        
        # Get unique networks
        networks = sorted([n for n in region_info['Network'].unique() if not pd.isna(n)])
        
        # Map networks to regions
        network_to_regions = {}
        for network in networks:
            regions = region_info[region_info['Network'] == network]['Region_Number'].values - 1
            network_to_regions[network] = regions.tolist()
        
        return region_to_network, region_info, networks, network_to_regions
    
    except Exception as e:
        print(f"Error parsing atlas with generic parser: {e}")
        return {}, pd.DataFrame(), [], {}
    

def standardize_network_names(features, networks):
    """
    Create mappings between feature name convention and atlas network names
    
    Parameters:
    -----------
    features : list
        List of feature names like 'network_strength_Dorsal_Attention'
    networks : list
        List of network names from atlas like 'Dorsal attention'
        
    Returns:
    --------
    dict
        Mapping from feature style names to atlas style names
    """
    # Extract network names from features
    feature_networks = set()
    for feature in features:
        if 'network_strength_' in feature:
            network = feature.replace('network_strength_', '')
            feature_networks.add(network)
        elif 'mean_connectivity_' in feature:
            network = feature.replace('mean_connectivity_', '')
            feature_networks.add(network)
    
    # Create potential standardized versions of atlas networks
    atlas_network_variants = {}
    for network in networks:
        # Original
        atlas_network_variants[network] = network
        # No spaces
        atlas_network_variants[network.replace(' ', '')] = network
        # Underscores
        atlas_network_variants[network.replace(' ', '_')] = network
        # Lowercase
        atlas_network_variants[network.lower()] = network
        # Lowercase with underscores
        atlas_network_variants[network.lower().replace(' ', '_')] = network
    
    # Create mapping between feature networks and atlas networks
    network_mapping = {}
    for feature_network in feature_networks:
        # Try direct lookup
        if feature_network in atlas_network_variants:
            network_mapping[feature_network] = atlas_network_variants[feature_network]
        # Try case-insensitive lookup
        elif feature_network.lower() in atlas_network_variants:
            network_mapping[feature_network] = atlas_network_variants[feature_network.lower()]
        # Try to replace with closest match
        else:
            closest = None
            max_similarity = 0
            for atlas_variant in atlas_network_variants:
                # Simple similarity: count matching characters
                similarity = sum(c1 == c2 for c1, c2 in zip(feature_network.lower(), atlas_variant.lower()))
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest = atlas_network_variants[atlas_variant]
            if closest:
                network_mapping[feature_network] = closest
                print(f"Mapped '{feature_network}' to '{closest}' (closest match)")
            else:
                print(f"WARNING: No mapping found for '{feature_network}'")
    
    return network_mapping