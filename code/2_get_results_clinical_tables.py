'''
Date created: 04-10-2025
Author: Chanyoung Ko
Objective: Compare the clustering characteristics of demographics, gaze and fMRI features

Note: 
- For the 42 subjects that have both gaze & fMRI data, perform clustering analysis
- Only perform statistical analysis & create tables for demographics
- Will in a separate script develop code for creating figures for gaze & fMRI features to show cluster differences
'''
#%%
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Import config and utility modules
from src.utils_config import OUTPUT_TABLES_DIR, OUTPUT_FIGURES_DIR
from src.utils_results_df import (
    load_clinical_data, load_cluster_data, check_id_consistency, 
    merge_dataframes, preprocess_categorical_variables
)
from src.utils_results_table import create_overall_table, create_cluster_comparison_table
from src.utils_results_figure import create_visualizations


def summarize_key_variables(df):
    """Generate summary statistics for key variables."""
    print("\nSummary statistics for key variables:")
    key_vars = ['age', 'fsiq', 'ados_total', 'srs_total', 'scq_total']
    key_vars_available = [var for var in key_vars if var in df.columns]
    
    if key_vars_available:
        summary = df[key_vars_available].describe()
        print(summary)
        return summary
    else:
        print("No key variables found in the dataset.")
        return None


def analyze_continuous_variable(df, var):
    """Perform statistical tests for a continuous variable across clusters."""
    from scipy.stats import kruskal, mannwhitneyu
    
    # Check if there's enough data
    if df[var].notna().sum() < 10:
        print(f"  Skipping {var} due to insufficient data")
        return None, None
    
    # Kruskal-Wallis test (non-parametric ANOVA)
    groups = [df[df['cluster'] == cluster][var].dropna() for cluster in df['cluster'].unique()]
    if all(len(g) > 0 for g in groups):
        stat, p_value = kruskal(*groups)
        print(f"  {var}: H={stat:.2f}, p={p_value:.4f}")
        
        # If significant, perform post-hoc pairwise Mann-Whitney U tests
        posthoc_results = None
        if p_value < 0.05:
            print(f"    Significant difference found for {var}, performing post-hoc tests")
            posthoc_results = {}
            clusters = sorted(df['cluster'].unique())
            for i, c1 in enumerate(clusters):
                for c2 in clusters[i+1:]:
                    group1 = df[df['cluster'] == c1][var].dropna()
                    group2 = df[df['cluster'] == c2][var].dropna()
                    if len(group1) > 0 and len(group2) > 0:
                        stat, p_value = mannwhitneyu(group1, group2)
                        key = f"Cluster {c1} vs {c2}"
                        posthoc_results[key] = {'U': stat, 'p-value': p_value}
                        print(f"      {key}: U={stat:.2f}, p={p_value:.4f}")
        
        return (stat, p_value), posthoc_results
    return None, None


def analyze_categorical_variable(df, var):
    """Perform statistical tests for a categorical variable across clusters."""
    from scipy.stats import chi2_contingency
    
    # Create contingency table
    contingency_table = pd.crosstab(df['cluster'], df[var])
    
    # Only perform test if we have enough data
    if contingency_table.size > 0 and contingency_table.sum().sum() > 10:
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"  {var}: χ²={chi2:.2f}, p={p_value:.4f}")
        print(f"    Contingency table:\n{contingency_table}")
        return (chi2, p_value), contingency_table
    else:
        print(f"  Skipping {var} due to insufficient data")
        return None, None


def perform_detailed_statistical_tests(df):
    """Perform detailed statistical tests for key variables across clusters."""
    print("\nPerforming detailed statistical tests...")
    
    # Select key variables based on your research questions
    key_continuous_vars = [
        'age', 'fsiq', 'ados_total', 'total_css', 'rrb_css', 'sa_css',
        'srs_total', 'scq_total', 'ija_success', 'rja_low_success', 'rja_high_success',
        'vabs_socialization'
    ]
    key_continuous_vars = [var for var in key_continuous_vars if var in df.columns]
    
    key_categorical_vars = ['sex', 'sev_ados', 'sev_ados_binary', 'sev_srs']
    key_categorical_vars = [var for var in key_categorical_vars if var in df.columns]
    
    # Initialize results dataframes
    continuous_results = pd.DataFrame(columns=['Variable', 'Test', 'Statistic', 'p-value', 'Significant'])
    categorical_results = pd.DataFrame(columns=['Variable', 'Test', 'Statistic', 'p-value', 'Significant'])
    
    # Function to add result to dataframe
    def add_result(results_df, variable, test, statistic, p_value):
        significant = "Yes" if p_value < 0.05 else "No"
        return pd.concat([results_df, pd.DataFrame({
            'Variable': [variable],
            'Test': [test],
            'Statistic': [statistic],
            'p-value': [p_value],
            'Significant': [significant]
        })], ignore_index=True)
    
    # Perform statistical tests for continuous variables
    print("Analyzing continuous variables...")
    for var in key_continuous_vars:
        try:
            result, posthoc = analyze_continuous_variable(df, var)
            if result:
                stat, p_value = result
                continuous_results = add_result(continuous_results, var, 'Kruskal-Wallis', stat, p_value)
        except Exception as e:
            print(f"  Error analyzing {var}: {e}")
    
    # Perform statistical tests for categorical variables
    print("\nAnalyzing categorical variables...")
    for var in key_categorical_vars:
        try:
            result, contingency = analyze_categorical_variable(df, var)
            if result:
                chi2, p_value = result
                categorical_results = add_result(categorical_results, var, 'Chi-square', chi2, p_value)
        except Exception as e:
            print(f"  Error analyzing {var}: {e}")
    
    # Save results to CSV
    continuous_output_path = OUTPUT_TABLES_DIR / "continuous_variables_statistical_tests.csv"
    categorical_output_path = OUTPUT_TABLES_DIR / "categorical_variables_statistical_tests.csv"
    
    continuous_results.to_csv(continuous_output_path, index=False)
    categorical_results.to_csv(categorical_output_path, index=False)
    
    print(f"\nContinuous variable test results saved to: {continuous_output_path}")
    print(f"Categorical variable test results saved to: {categorical_output_path}")
    
    return continuous_results, categorical_results

#%%
def main():
    """Main function to execute the analysis pipeline."""
    print("Starting cluster characteristics analysis...")
    
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
    
    try:
        # Load data
        clinical_df = load_clinical_data()
        cluster_df = load_cluster_data()
        
        # Check ID consistency
        check_id_consistency(clinical_df, cluster_df)
        
        # Merge dataframes
        merged_df = merge_dataframes(clinical_df, cluster_df)
        
        # Preprocess categorical variables
        processed_df = preprocess_categorical_variables(merged_df)
        
        # Save the merged dataframe for future use
        merged_output_path = OUTPUT_TABLES_DIR / "participant_clinical_cluster_df.csv"
        merged_df.to_csv(merged_output_path, index=False)
        print(f"\nMerged clinical and cluster data saved to: {merged_output_path}")
        
        # Generate summary statistics
        summarize_key_variables(processed_df)
        
        # Create tables
        create_overall_table(processed_df)
        create_cluster_comparison_table(processed_df)
        
        # Perform detailed statistical tests
        perform_detailed_statistical_tests(processed_df)
        
        # Create visualizations
        create_visualizations(processed_df)
        
        print("\nAnalysis complete. Results saved to:")
        print(f"- Tables: {OUTPUT_TABLES_DIR}")
        print(f"- Figures: {OUTPUT_FIGURES_DIR}")
        
    except Exception as e:
        print(f"\nERROR: An error occurred during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# %%
## Read the saved merged_df 'participant_clinical_cluster_df.csv' and then check the number of unique 'id' values
# merged_df = pd.read_csv(OUTPUT_TABLES_DIR / "participant_clinical_cluster_df.csv")
# unique_ids = merged_df['id'].nunique()
# print(f"Number of unique IDs in the merged dataframe: {unique_ids}")
# unique_clusters = merged_df['cluster'].nunique()
# print(f"Number of unique clusters in the merged dataframe: {unique_clusters}")

    