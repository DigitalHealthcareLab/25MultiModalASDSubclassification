'''
Helper functions for gene analysis
'''
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ----------------------------
# 1. Robust CSV Loader
# ----------------------------
def smart_load_csv(filepath):
    """Optimized CSV loader with better performance"""
    try:
        # Try reading with pandas' automatic parser first
        return pd.read_csv(filepath, low_memory=False)
    except Exception as e1:
        try:
            # Try tab delimiter
            return pd.read_csv(filepath, sep='\t', low_memory=False)
        except Exception as e2:
            # Use chunking for very large files as final attempt
            try:
                chunks = []
                for chunk in pd.read_csv(filepath, chunksize=10000):
                    chunks.append(chunk)
                return pd.concat(chunks)
            except Exception as e3:
                print(f"Failed to load {filepath}: {e3}")
                return None

# ----------------------------
# 2. Coding Variant Analysis
# ----------------------------
def analyze_coding_variants(participant_id, variant_type, base_dir, variant_types):
    """
    Analyze coding variants for a single participant and return both summary and detailed categorized data.
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    
    # Standardize ID format
    if not participant_id.startswith("asd"):
        participant_id = f"asd{participant_id}"
    id_num = int(participant_id[3:])
    participant_id = f"asd{id_num:03d}"

    # Build file path
    variant_info = variant_types[variant_type]
    preproc_dir = Path(base_dir) / 'preproc_data' / variant_type
    participant_file = preproc_dir / f"{participant_id}_{variant_info['variant_type1']}_{variant_info['variant_type2']}.csv"

    if not participant_file.exists():
        print(f"No file found for participant {participant_id} in {variant_type}")
        return None

    # Load the file with tab delimiter
    try:
        # First check if it's tab-delimited
        with open(participant_file, 'r') as f:
            first_line = f.readline().strip()
            
        if '\t' in first_line:
            print(f"DEBUG: Loading tab-delimited file for {participant_id}")
            df = pd.read_csv(participant_file, sep='\t')
        else:
            print(f"DEBUG: Loading comma-delimited file for {participant_id}")
            df = pd.read_csv(participant_file)
            
        # print(f"DEBUG: Successfully loaded file. Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"ERROR loading file: {e}")
        return None
    
    if df is None or df.empty:
        print(f"Empty or unreadable file for {participant_id}")
        return None
    
    # Check if REF:ALT exists, otherwise create it
    if 'REF:ALT' not in df.columns:
        # print(f"DEBUG: REF:ALT column not found for {participant_id}")
        if 'REF' in df.columns and 'ALT' in df.columns:
            # print(f"DEBUG: Creating REF:ALT from REF and ALT columns")
            df['REF:ALT'] = df['REF'] + ':' + df['ALT']
        else:
            print(f"DEBUG: Cannot find REF and ALT columns. Available columns: {df.columns.tolist()}")
            return None
    
    # Split alleles if needed
    if 'REF' not in df.columns or 'ALT' not in df.columns:
        # Create them if they don't exist
        df['REF'] = df['REF:ALT'].str.split(':', expand=True)[0]
        df['ALT'] = df['REF:ALT'].str.split(':', expand=True)[1]
    
    # Filter for SNVs immediately to reduce dataframe size
    snv_mask = (df['REF'].str.len() == 1) & (df['ALT'].str.len() == 1)
    df = df[snv_mask].copy()  # Keep only SNVs and assign back to df
    
    # Convert numeric values all at once
    numeric_cols = ['CADD_phred_hg19', 'PLI', 'DANN_score', 'Eigen_phred_score', 
                    'Polyphen2_HDIV_score', 'Polyphen2_HVAR_score', 'SIFT_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Use numpy for categorization - faster than string methods
    # Define conditions for variant categories
    ptv_patterns = ['frameshift_variant', 'stop_gained', 'splice_donor_variant', 
                    'splice_acceptor_variant', 'start_lost']
    
    # Create category column with vectorized operations
    df['Variant_Category'] = 'Other'
    for pattern in ptv_patterns:
        df.loc[df['Variant_Effect'].str.contains(pattern, case=False, na=False), 'Variant_Category'] = 'PTV'
    
    df.loc[df['Variant_Effect'].str.contains('missense_variant', case=False, na=False), 'Variant_Category'] = 'Missense'
    df.loc[df['Variant_Effect'].str.contains('synonymous_variant', case=False, na=False), 'Variant_Category'] = 'Synonymous'

    # Add genomic context - only needed for coding variants
    df['genomic_context'] = 'exonic'
    
    # Create CADD bins with numpy vectorization
    conditions = [
        df['Variant_Category'] == 'PTV',
        df['Variant_Category'] == 'Synonymous',
        (df['Variant_Category'] == 'Missense') & (df['CADD_phred_hg19'] > 30),
        (df['Variant_Category'] == 'Missense') & (df['CADD_phred_hg19'] > 20) & (df['CADD_phred_hg19'] <= 30),
        (df['Variant_Category'] == 'Missense') & (df['CADD_phred_hg19'] <= 20)
    ]
    choices = ['PTV', 'Synonymous', 'CADD>30', 'CADD20-30', 'CADD<20']
    df['cadd_bin'] = np.select(conditions, choices, default=None)
    
    # Create pLI constraint with numpy
    df['pli_constraint'] = np.where(df['PLI'] >= 0.9, 'pLI>=0.9', 'pLI<0.9')
    
    # Add regulatory flag (always False for coding)
    df['regulatory_flag'] = False
    
    # Create summary calculations with proper filtering
    sfari_variants = df[df['SFARI_gene_score'] != '.']
    
    # Create variant category summaries with a single groupby operation
    # This is much more efficient than multiple filters
    category_summaries = {}
    
    # Group by main categories
    for category, cat_group in df.groupby('Variant_Category'):
        category_summaries[category] = {}
        
        if category == 'Missense':
            # For missense, we need to further group by CADD bin
            for cadd_bin, cadd_group in cat_group.groupby('cadd_bin'):
                category_summaries[category][cadd_bin] = {}
                
                # Then by pLI constraint
                for pli_bin, pli_group in cadd_group.groupby('pli_constraint'):
                    genes = [g for g in pli_group['GeneName'].unique() 
                            if g not in ['.', '--'] and pd.notna(g)]
                    
                    category_summaries[category][cadd_bin][pli_bin] = {
                        'variant_count': len(pli_group),
                        'gene_count': len(genes),
                        'genes': genes
                    }
        else:
            # For other categories, we just group by pLI constraint
            for pli_bin, pli_group in cat_group.groupby('pli_constraint'):
                genes = [g for g in pli_group['GeneName'].unique() 
                        if g not in ['.', '--'] and pd.notna(g)]
                
                category_summaries[category][pli_bin] = {
                    'variant_count': len(pli_group),
                    'gene_count': len(genes),
                    'genes': genes
                }
    
    # Create SFARI category summaries with a single operation
    sfari_summaries = {}
    for score, score_group in sfari_variants.groupby('SFARI_gene_score'):
        if score == '.':
            continue
            
        genes = [g for g in score_group['GeneName'].unique() 
                if g not in ['.', '--'] and pd.notna(g)]
        
        sfari_summaries[score] = {
            'variant_count': len(score_group),
            'gene_count': len(genes),
            'genes': genes
        }
    
    # Return full summary
    summary = {
        'participant_id': participant_id,
        'total_variants': len(df),
        'sfari_variants': len(sfari_variants),
        'percent_sfari': (len(sfari_variants) / len(df) * 100) if len(df) > 0 else 0,
        'variant_categories': category_summaries,
        'sfari_categories': sfari_summaries,
        'categorized_df': df
    }
    
    return summary

# ----------------------------
# 2. Noncoding Variant Analysis
# ----------------------------
def get_genomic_context(genomic_region):
    g = str(genomic_region).lower()
    if 'intron' in g:
        return 'Intronic'
    if 'intergenic' in g:
        return 'Intergenic'
    return 'Other'  # Downstream, Upstream, UTRs, etc.

def get_regulatory_element(regulatory_region):
    r = str(regulatory_region).lower()
    if 'enhancer' in r and 'promoter' in r:
        return 'Enhancer,Promoter'
    if 'promoter' in r:
        return 'Promoter'
    if 'enhancer' in r:
        return 'Enhancer'
    return None

def categorize_noncoding_variants_optimized(df):
    """
    Categorizes rare noncoding SNVs based on Eigen, regulatory region, and SFARI score.
    Optimized for performance with vectorized operations.
    
    Args:
        df (pd.DataFrame): Raw variant DataFrame for a participant (rare noncoding).
    
    Returns:
        pd.DataFrame: Cleaned, categorized DataFrame with useful annotations added.
    """
    import pandas as pd
    import numpy as np
    
    # Copy the dataframe once
    df = df.copy()
    
    # Filter for SNVs in one step
    if 'REF' not in df.columns or 'ALT' not in df.columns:
        ref_alt_parts = df['REF:ALT'].str.split(':', expand=True)
        df['REF'] = ref_alt_parts[0]
        df['ALT'] = ref_alt_parts[1]
    
    # Keep only SNVs
    snv_mask = (df['REF'].str.len() == 1) & (df['ALT'].str.len() == 1)
    df = df[snv_mask]
    
    # Convert numeric columns in one step
    numeric_cols = ['Eigen_phred_score', 'CADD_phred_hg19']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create Eigen bins with numpy
    df['eigen_bin'] = np.where(df['Eigen_phred_score'] > 0, 'Eigen>0', 'Eigen<=0')
    
    # Map genomic regions with numpy select
    region_conditions = [
        df['Genomic_Region'].str.contains('intron', case=False, na=False),
        df['Genomic_Region'].str.contains('intergenic', case=False, na=False)
    ]
    region_choices = ['Intronic', 'Intergenic']
    df['genomic_context'] = np.select(region_conditions, region_choices, default='Other')
    
    # Map regulatory elements with numpy select
    reg_conditions = [
        (df['Regulatory_Region'].str.contains('enhancer', case=False, na=False) & 
         df['Regulatory_Region'].str.contains('promoter', case=False, na=False)),
        df['Regulatory_Region'].str.contains('promoter', case=False, na=False),
        df['Regulatory_Region'].str.contains('enhancer', case=False, na=False)
    ]
    reg_choices = ['Enhancer,Promoter', 'Promoter', 'Enhancer']
    df['regulatory_element'] = np.select(reg_conditions, reg_choices, default=None)
    
    # Create simplified region (prioritize regulatory elements)
    df['Simplified_Region'] = df['regulatory_element'].fillna(df['genomic_context'])
    df['Variant_Category'] = df['Simplified_Region']
    
    # Set regulatory flag (more efficient with numpy)
    regulatory_conditions = [
        df['Regulatory_Region'].str.contains('promoter|enhancer', case=False, na=False),
        df['Genomic_Region'].str.contains('5_prime_utr_variant|3_prime_utr_variant|upstream_gene_variant', 
                                        case=False, na=False)
    ]
    df['regulatory_flag'] = np.any(regulatory_conditions, axis=0)
    
    # Clean SFARI score and create category mapping
    df['SFARI_gene_score'] = df['SFARI_gene_score'].astype(str)
    
    # Map SFARI categories
    sfari_conditions = [
        df['SFARI_gene_score'] == '1',
        df['SFARI_gene_score'] == '2',
        df['SFARI_gene_score'] == '3',
        df['SFARI_gene_score'] == 'S'
    ]
    sfari_choices = ['SFARI 1', 'SFARI 2', 'SFARI 3', 'SFARI S']
    df['SFARI_Category'] = np.select(sfari_conditions, sfari_choices, default='Not SFARI')
    
    return df

def analyze_noncoding_variants(participant_id, variant_type, base_dir, variant_types):
    """
    Analyze rare noncoding variants for a single participant using Eigen, region, and SFARI scores.
    Optimized for performance with vectorized operations and reduced memory usage.

    Args:
        participant_id (str): Participant ID (e.g., 'asd011')
        variant_type (str): Variant type key (e.g., '3_3_Rare_Noncoding')
        base_dir (Path or str): Base directory for loading files
        variant_types (dict): Dictionary mapping variant type keys to file type info

    Returns:
        dict: Summary dictionary for this participant's noncoding variant burden including categorized_df
    """
    
    # Standardize ID format
    if not participant_id.startswith("asd"):
        participant_id = f"asd{participant_id}"
    id_num = int(participant_id[3:])
    participant_id = f"asd{id_num:03d}"

    # Build file path
    variant_info = variant_types[variant_type]
    preproc_dir = Path(base_dir) / 'preproc_data' / variant_type
    participant_file = preproc_dir / f"{participant_id}_{variant_info['variant_type1']}_{variant_info['variant_type2']}.csv"

    if not participant_file.exists():
        print(f"No file found for participant {participant_id} in {variant_type}")
        return None

    # Load with potential chunking for large files
    try:
        # First try to determine file size
        file_size = participant_file.stat().st_size
        if file_size > 100_000_000:  # >100MB
            print(f"Large file detected, processing in chunks: {participant_file}")
            chunks = []
            for chunk in pd.read_csv(participant_file, chunksize=100000):
                # Process each chunk to filter
                chunk['REF'] = chunk['REF:ALT'].str.split(':', expand=True)[0]
                chunk['ALT'] = chunk['REF:ALT'].str.split(':', expand=True)[1]
                chunk = chunk[(chunk['REF'].str.len() == 1) & (chunk['ALT'].str.len() == 1)]
                chunks.append(chunk)
            df_raw = pd.concat(chunks)
        else:
            df_raw = smart_load_csv(participant_file)
    except Exception as e:
        print(f"Error processing file: {e}")
        df_raw = smart_load_csv(participant_file)

    if df_raw is None or df_raw.empty:
        print(f"⚠️ Empty or unreadable file for {participant_id}")
        return None

    # Categorize variants with optimized function
    df = categorize_noncoding_variants_optimized(df_raw)
    
    # Calculate all statistics using groupby operations
    total_snvs = len(df)
    
    # SFARI stats
    sfari_mask = df['SFARI_gene_score'] != '.'
    sfari_variants = df[sfari_mask]
    sfari_count = len(sfari_variants)
    percent_sfari = (sfari_count / total_snvs * 100) if total_snvs > 0 else 0

    # Calculate distributions in one pass
    region_counts = df['Simplified_Region'].value_counts().to_dict()
    impact_distribution = df['eigen_bin'].value_counts().to_dict()
    
    # Regulatory variants
    regulatory_mask = df['regulatory_flag'] == True
    regulatory_variants = df[regulatory_mask]
    reg_total = len(regulatory_variants)
    reg_sfari = len(regulatory_variants[sfari_mask])
    reg_percent = (reg_total / total_snvs * 100) if total_snvs > 0 else 0

    # Create SFARI impact distribution with groupby
    sfari_impact = {}
    if not sfari_variants.empty:
        # Group by SFARI score and eigen bin
        sfari_grouped = sfari_variants.groupby(['SFARI_gene_score', 'eigen_bin']).size().unstack(fill_value=0)
        
        # Process each SFARI score
        for score in ['1', '2', '3', 'S']:
            if score in sfari_grouped.index:
                sfari_impact[score] = sfari_grouped.loc[score].to_dict()
                
                # Add gene counts
                score_genes = sfari_variants[sfari_variants['SFARI_gene_score'] == score]['GeneName'].unique()
                score_genes = [g for g in score_genes if g not in ['.', '--'] and pd.notna(g)]
                
                sfari_impact[score]['gene_count'] = len(score_genes)
                sfari_impact[score]['genes'] = score_genes

    # Create region impact matrix with groupby
    region_impact = {}
    
    # Group by region and eigen bin to get counts
    region_grouped = df.groupby(['Simplified_Region', 'eigen_bin']).size().unstack(fill_value=0)
    
    # Process each region
    for region in region_counts.keys():
        if region in region_grouped.index:
            region_impact[region] = region_grouped.loc[region].to_dict()
            
            # Add gene counts
            region_genes = df[df['Simplified_Region'] == region]['GeneName'].unique()
            region_genes = [g for g in region_genes if g not in ['.', '--'] and pd.notna(g)]
            
            region_impact[region]['gene_count'] = len(region_genes)
            region_impact[region]['genes'] = region_genes

    # Create summary
    summary = {
        'participant_id': participant_id,
        'total_variants': total_snvs,
        'sfari_variants': sfari_count,
        'percent_sfari': percent_sfari,
        'region_distribution': region_counts,
        'impact_distribution': impact_distribution,
        'regulatory_variants': {
            'total': reg_total,
            'percent': reg_percent,
            'sfari_count': reg_sfari,
            'sfari_percent': (reg_sfari / reg_total * 100) if reg_total > 0 else 0,
        },
        'sfari_impact_distribution': sfari_impact,
        'region_impact_distribution': region_impact,
        'categorized_df': df
    }

    return summary