'''
Utility functions for preparing clinical and cluster dataframes for analysis.
'''
# Libraries
import pandas as pd
from src.utils_config import CLINICAL_DF_PATH, CLUSTER_DF_PATH


def load_clinical_data():
    """Load and preprocess the clinical data."""
    print(f"Loading clinical data from: {CLINICAL_DF_PATH}")
    try:
        df = pd.read_csv(CLINICAL_DF_PATH)
        df = df.drop_duplicates(subset=['id'])
        print(f"Clinical data shape: {df.shape}")
        print(f"Clinical data columns: {', '.join(df.columns.tolist())}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("\nMissing values in clinical data:")
        print(missing_values[missing_values > 0])
        
        # Check unique IDs
        print(f"\nNumber of unique IDs in clinical data: {df['id'].nunique()}")
        print(f"First 5 IDs in clinical data: {', '.join(df['id'].iloc[:5].tolist())}")
        
        return df
    except FileNotFoundError:
        print(f"ERROR: Could not find clinical data file at {CLINICAL_DF_PATH}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load clinical data: {str(e)}")
        raise


def load_cluster_data():
    """Load and preprocess the cluster data."""
    print(f"\nLoading cluster data from: {CLUSTER_DF_PATH}")
    try:
        df = pd.read_csv(CLUSTER_DF_PATH)
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
        
        # Fix the ID column - rename 'Unnamed: 0' to 'id' if it exists
        if 'Unnamed: 0' in df.columns:
            print("Renaming 'Unnamed: 0' column to 'id'")
            df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
        
        print(f"Cluster data shape: {df.shape}")
        print(f"Cluster data columns: {', '.join(df.columns.tolist())}")
        print("First few rows of cluster data:")
        print(df.head())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("\nMissing values in cluster data:")
        print(missing_values[missing_values > 0])
        
        # Check unique IDs
        print(f"\nNumber of unique IDs in cluster data: {df['id'].nunique()}")
        print(f"First 5 IDs in cluster data: {', '.join(df['id'].iloc[:5].tolist())}")
        
        # Analyze clusters
        print(f"\nUnique clusters: {df['cluster'].unique().tolist()}")
        cluster_counts = df['cluster'].value_counts().sort_index()
        print("Cluster distribution:")
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count} subjects ({count/len(df)*100:.1f}%)")
        
        return df
    except FileNotFoundError:
        print(f"ERROR: Could not find cluster data file at {CLUSTER_DF_PATH}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load cluster data: {str(e)}")
        raise


def check_id_consistency(clinical_df, cluster_df):
    """Check for inconsistencies in IDs between clinical and cluster data."""
    clinical_ids = set(clinical_df['id'])
    cluster_ids = set(cluster_df['id'])
    
    ids_in_clinical_not_cluster = clinical_ids - cluster_ids
    ids_in_cluster_not_clinical = cluster_ids - clinical_ids
    
    print(f"\nIDs in clinical data but not in cluster data: {len(ids_in_clinical_not_cluster)}")
    if len(ids_in_clinical_not_cluster) > 0 and len(ids_in_clinical_not_cluster) <= 10:
        print(f"  {', '.join(list(ids_in_clinical_not_cluster))}")
    
    print(f"IDs in cluster data but not in clinical data: {len(ids_in_cluster_not_clinical)}")
    if len(ids_in_cluster_not_clinical) > 0 and len(ids_in_cluster_not_clinical) <= 10:
        print(f"  {', '.join(list(ids_in_cluster_not_clinical))}")
    
    return ids_in_clinical_not_cluster, ids_in_cluster_not_clinical


def merge_dataframes(clinical_df, cluster_df):
    """Merge clinical and cluster dataframes."""
    print("\nMerging clinical and cluster data...")
    merged_df = pd.merge(cluster_df, clinical_df, on='id', how='inner')
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Number of subjects retained after merging: {merged_df['id'].nunique()}")
    
    # Check for any columns that might have been duplicated in the merge
    duplicate_columns = [col for col in merged_df.columns if col.endswith('_x') or col.endswith('_y')]
    if duplicate_columns:
        print("\nWarning: Duplicate columns found after merging:")
        print(f"  {', '.join(duplicate_columns)}")
    
    # Check cluster distribution after merging
    print("\nCluster distribution after merging:")
    merge_cluster_counts = merged_df['cluster'].value_counts().sort_index()
    for cluster, count in merge_cluster_counts.items():
        print(f"  Cluster {cluster}: {count} subjects ({count/len(merged_df)*100:.1f}%)")
    
    return merged_df


def preprocess_categorical_variables(df):
    """Preprocess categorical variables in the merged dataframe."""
    print("\nPreprocessing categorical variables...")
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Replace values in categorical columns
    if 'sev_ados_binary' in processed_df.columns:
        processed_df['sev_ados_binary'] = processed_df['sev_ados_binary'].replace({0: 'mild', 1: 'moderate-severe'})
    
    if 'sev_ados' in processed_df.columns:
        processed_df['sev_ados'] = processed_df['sev_ados'].replace({0: 'mild', 1: 'moderate', 2: 'severe'})
    
    # Convert to categorical data type
    categorical_columns = ['cluster', 'sex', 'sev_ados_binary', 'sev_ados']
    for col in categorical_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype('category')
    return processed_df