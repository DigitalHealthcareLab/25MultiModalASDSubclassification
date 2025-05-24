'''
Utility functions for creating tables of demographic and clinical variables.
'''

# Library imports
import pandas as pd
from tableone import TableOne
from src.utils_config import OUTPUT_TABLES_DIR


def create_overall_table(df):
    """Create an overall table with demographic and clinical variables."""
    print("\nCreating overall characteristics table...")
    
    # Specify columns for the table
    columns = [
        'sex', 'age', 'ija_success', 'rja_low_success',
        'rja_high_success', 'ja_score', 'fsiq', 'vci', 'vsi', 'fri', 'wmi',
        'psi', 'ados_total', 'total_css', 'sev_ados',
        'sev_ados_binary', 'ados_sa_total', 'sa_css', 'sev_sa', 'rrb',
        'rrb_css', 'srs_total', 'srs_t_score', 'sev_srs', 'srs_awareness',
        'srs_cognition', 'srs_comm', 'srs_motivation', 'srs_rrb', 'scq_total',
        'cbcl_total', 'internal_sx', 'external_sx', 'emotional_problems',
        'anxiety', 'somatic_sx', 'withdrawal', 'sleep', 'attention',
        'aggression', 'affect', 'anxiety_sx', 'pdd', 'adhd', 'odd', 'vabs_comm',
        'vabs_daily', 'vabs_socialization', 'vabs_maladaptive', 'vabs_internal',
        'vabs_external', 'dcdq_control', 'dcdq_finemotor', 'dcdq_coord'
    ]
    
    # Filter for columns that actually exist in the dataframe
    columns = [col for col in columns if col in df.columns]
    
    # Specify categorical variables
    categorical = ['sex', 'sev_ados', 'sev_ados_binary', 'sev_srs', 'sev_sa']
    categorical = [col for col in categorical if col in df.columns]
    
    # Create TableOne for overall characteristics
    table1 = TableOne(df, columns=columns, categorical=categorical, pval=False)
    
    # Save the table
    output_path = OUTPUT_TABLES_DIR / "overall_participant_characteristics.csv"
    table1.to_csv(output_path)
    print(f"Overall table saved to: {output_path}")
    print(table1)
    
    return table1


def create_cluster_comparison_table(df):
    """Create a table comparing demographics and clinical measures across clusters."""
    print("\nCreating cluster comparison table...")
    
    # Specify columns for the table (same as overall table)
    columns = [
        'sex', 'age', 'ija_success', 'rja_low_success',
        'rja_high_success', 'ja_score', 'fsiq', 'vci', 'vsi', 'fri', 'wmi',
        'psi', 'ados_total', 'total_css', 'sev_ados',
        'sev_ados_binary', 'ados_sa_total', 'sa_css', 'sev_sa', 'rrb',
        'rrb_css', 'srs_total', 'srs_t_score', 'sev_srs', 'srs_awareness',
        'srs_cognition', 'srs_comm', 'srs_motivation', 'srs_rrb', 'scq_total',
        'cbcl_total', 'internal_sx', 'external_sx', 'emotional_problems',
        'anxiety', 'somatic_sx', 'withdrawal', 'sleep', 'attention',
        'aggression', 'affect', 'anxiety_sx', 'pdd', 'adhd', 'odd', 'vabs_comm',
        'vabs_daily', 'vabs_socialization', 'vabs_maladaptive', 'vabs_internal',
        'vabs_external', 'dcdq_control', 'dcdq_finemotor', 'dcdq_coord'
    ]
    
    # Filter for columns that actually exist in the dataframe
    columns = [col for col in columns if col in df.columns]
    
    # Specify categorical variables
    categorical = ['sex', 'sev_ados', 'sev_ados_binary', 'sev_srs', 'sev_sa']
    categorical = [col for col in categorical if col in df.columns]
    
    # Specify non-normal variables for which to use median and IQR
    # nonnormal = [
    #     'age', 'ija_success', 'rja_low_success', 'rja_high_success', 'ja_score',
    #     'fsiq', 'vci', 'vsi', 'fri', 'wmi', 'psi', 'ados_total', 'total_css',
    #     'sa_css', 'rrb_css', 'srs_total', 'srs_t_score', 'scq_total', 'cbcl_total',
    #     'internal_sx', 'external_sx', 'vabs_comm', 'vabs_daily', 'vabs_socialization',
    #     'dcdq_control', 'dcdq_finemotor', 'dcdq_coord'
    # ]
    # nonnormal = [col for col in nonnormal if col in df.columns]
    
    # Treat all continuous variables as non-normal due to small N
    all_continuous = [col for col in columns if col not in categorical]
    nonnormal = all_continuous

    # Create TableOne grouped by cluster
    table_cluster = TableOne(
        df, 
        columns=columns, 
        categorical=categorical, 
        groupby='cluster', 
        nonnormal=nonnormal,
        pval=True,  # Include p-values for comparisons
        pval_adjust='bonferroni'  # Apply Bonferroni correction for multiple comparisons
    )
    
    # Save the table
    output_path = OUTPUT_TABLES_DIR / "cluster_comparison_table.csv"
    table_cluster.to_csv(output_path)
    print(f"Cluster comparison table saved to: {output_path}")
    print(table_cluster)
    
    return table_cluster