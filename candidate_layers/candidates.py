#!/usr/bin/env python3
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def process_file(file_path, output_root):
    # Extract model name from filename, e.g. "H1-bert-base-uncased-Wic-results.csv"
    filename = os.path.basename(file_path)
    if not filename.endswith("-Wic-results.csv"):
        return  # Skip files that do not match the pattern
    model_name = filename[len("H1-") : -len("-Wic-results.csv")]

    # Create dedicated output folder for this model
    model_output_folder = os.path.join(output_root, model_name)
    os.makedirs(model_output_folder, exist_ok=True)
    
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # -------------------------------
    # 1) Visualization: Plot mean F1 vs. Layer with error bars
    # -------------------------------
    # Identify F1 columns (assuming columns like 'Run1 F1', 'Run2 F1', etc.)
    f1_columns = [col for col in df.columns if 'F1' in col]
    
    # Reshape the data into long format
    df_long = df.melt(id_vars=['Layer'], value_vars=f1_columns,
                      var_name='Run', value_name='F1_Score')
    # Extract run number (e.g., from "Run1 F1" get 1)
    df_long['Run'] = df_long['Run'].str.extract(r'Run(\d+)').astype(int)
    
    # Compute summary statistics per layer
    summary = df_long.groupby('Layer')['F1_Score'].agg(['mean', 'std']).reset_index()
    
    # Plot Mean F1 vs. Layer with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(summary['Layer'], summary['mean'], yerr=summary['std'],
                 fmt='-o', capsize=5)
    plt.xlabel('Layer')
    plt.ylabel('Mean F1 Score')
    plt.title(f'Mean F1 vs. Layer for {model_name}')
    mean_plot_path = os.path.join(model_output_folder, f"{model_name}_mean_f1_vs_layer.png")
    plt.savefig(mean_plot_path)
    plt.close()
    
    # -------------------------------
    # 2) One-Way ANOVA and Tukey's HSD Post-Hoc Test
    # -------------------------------
    # Fit the OLS model for ANOVA
    model_anova = smf.ols('F1_Score ~ C(Layer)', data=df_long).fit()
    anova_table = sm.stats.anova_lm(model_anova, typ=2)
    # Optionally, save the ANOVA table as CSV
    anova_path = os.path.join(model_output_folder, f"{model_name}_anova.csv")
    anova_table.to_csv(anova_path)
    
    # Run Tukey's HSD test
    tukey = pairwise_tukeyhsd(endog=df_long['F1_Score'],
                              groups=df_long['Layer'],
                              alpha=0.05)
    # Save the Tukey summary as text
    tukey_summary_str = tukey.summary().as_text()
    tukey_summary_path = os.path.join(model_output_folder, f"{model_name}_tukey_summary.txt")
    with open(tukey_summary_path, "w") as f:
        f.write(tukey_summary_str)
    
    # -------------------------------
    # 3) Visualization: Plotting the Tukey HSD Results
    # -------------------------------
    plt.figure(figsize=(12, 6))
    tukey.plot_simultaneous()
    plt.title(f"Tukey HSD: Mean differences between layers for {model_name}")
    plt.xlabel("F1 Score Difference")
    tukey_plot_path = os.path.join(model_output_folder, f"{model_name}_tukey_plot.png")
    plt.savefig(tukey_plot_path)
    plt.close()
    
    # -------------------------------
    # 4) Compute Beat Counts and Candidate Layers
    # -------------------------------
    # Extract Tukey HSD results as a DataFrame.
    tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    # Convert columns to appropriate types
    tukey_df['meandiff'] = tukey_df['meandiff'].astype(float)
    tukey_df['reject'] = tukey_df['reject'].astype(bool)
    
    # Get a sorted list of all layer labels (assumed numeric)
    all_layers = sorted(df_long['Layer'].unique())
    # Initialize dictionary to count how many layers each layer "beats"
    beat_counts = {layer: 0 for layer in all_layers}
    
    # Note: statsmodels returns meandiff as mean(group2) - mean(group1).
    # Thus, if meandiff is negative, group1's mean is higher.
    for _, row in tukey_df.iterrows():
        if not row['reject']:
            continue
        g1 = int(row['group1'])
        g2 = int(row['group2'])
        if row['meandiff'] < 0:
            # Group1 is better than Group2.
            beat_counts[g1] += 1
        elif row['meandiff'] > 0:
            # Group2 is better than Group1.
            beat_counts[g2] += 1
    
    # Define thresholds (non-mutually exclusive):
    # There are (n_layers - 1) comparisons per layer.
    n_layers = len(all_layers)
    n_other = n_layers - 1
    second_degree_threshold = math.ceil(0.75 * n_other)
    third_degree_threshold  = math.ceil(0.50 * n_other)
    
    first_degree = [L for L, count in beat_counts.items() if count == n_other]
    second_degree = [L for L, count in beat_counts.items() if count >= second_degree_threshold]
    third_degree = [L for L, count in beat_counts.items() if count >= third_degree_threshold]
    
    # Save beat counts and candidate lists to a text file
    candidates_path = os.path.join(model_output_folder, f"{model_name}_candidates.txt")
    with open(candidates_path, "w") as f:
        f.write("Beat counts per layer:\n")
        for layer in all_layers:
            f.write(f"Layer {layer}: {beat_counts[layer]}\n")
        f.write("\n")
        f.write(f"First-degree candidates (beats all {n_other} other layers): {first_degree}\n")
        f.write(f"Second-degree candidates (beats >= {second_degree_threshold} of other layers): {second_degree}\n")
        f.write(f"Third-degree candidates (beats >= {third_degree_threshold} of other layers): {third_degree}\n")
    
    print(f"Processed model {model_name} and saved outputs in {model_output_folder}")

def main():
    # Define directories relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to results folder (adjust as needed)
    results_folder = os.path.join(script_dir, '..', 'results')
    # Define output root folder (will be created if it doesn't exist)
    output_root = os.path.join(script_dir, '..', 'analysis_outputs')
    os.makedirs(output_root, exist_ok=True)
    
    # Iterate over each file in the results folder
    for filename in os.listdir(results_folder):
        if filename.endswith("-Wic-results.csv"):
            file_path = os.path.join(results_folder, filename)
            process_file(file_path, output_root)

if __name__ == "__main__":
    main()
