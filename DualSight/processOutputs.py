import pandas as pd

# Load the CSV file
file_path = "outputImages/evaluation_metrics_all.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# ------------------------------------------------------
# Modify your method_column_map to include TWO segrefiners
# ------------------------------------------------------
method_column_map = {
    "Base": [
        "base_mask_AP@50", "base_mask_AP@75", "base_mask_AP@95", "base_mask_AP@50:95",
        "base_mask_AR@50", "base_mask_AR@75", "base_mask_AR@95", "base_mask_AR@50:95"
    ],
    "SegRefiner - Small": [
        "segrefiner_small_mask_AP@50", "segrefiner_small_mask_AP@75", "segrefiner_small_mask_AP@95", "segrefiner_small_mask_AP@50:95",
        "segrefiner_small_mask_AR@50", "segrefiner_small_mask_AR@75", "segrefiner_small_mask_AR@95", "segrefiner_small_mask_AR@50:95"
    ],
    "SegRefiner - Large": [
        "segrefiner_large_mask_AP@50", "segrefiner_large_mask_AP@75", "segrefiner_large_mask_AP@95", "segrefiner_large_mask_AP@50:95",
        "segrefiner_large_mask_AR@50", "segrefiner_large_mask_AR@75", "segrefiner_large_mask_AR@95", "segrefiner_large_mask_AR@50:95"
    ],
    # "Zero Shot Refinement": [
    #     "zfr_mask_AP@50", "zfr_mask_AP@75", "zfr_mask_AP@95", "zfr_mask_AP@50:95",
    #     "zfr_mask_AR@50", "zfr_mask_AR@75", "zfr_mask_AR@95", "zfr_mask_AR@50:95"
    # ],
    "DualSight Old": [
        "dualsight_mask_AP@50", "dualsight_mask_AP@75", "dualsight_mask_AP@95", "dualsight_mask_AP@50:95",
        "dualsight_mask_AR@50", "dualsight_mask_AR@75", "dualsight_mask_AR@95", "dualsight_mask_AR@50:95"
    ],
    "DualSight (Ours)": [
        "sam_mask_AP@50", "sam_mask_AP@75", "sam_mask_AP@95", "sam_mask_AP@50:95",
        "sam_mask_AR@50", "sam_mask_AR@75", "sam_mask_AR@95", "sam_mask_AR@50:95"
    ]
}

# Extract and aggregate the data
processed_rows = []
for model in data["model_name"].unique():
    model_rows = []
    for method, columns in method_column_map.items():
        model_data = data[data["model_name"] == model]
        aggregated_values = model_data[columns].mean().values
        # Append the refinement method to the model name
        model_with_method = f"{model} + {method}" if method != "Base" else model
        model_rows.append([model_with_method] + list(aggregated_values))
    
    # Convert to DataFrame for processing
    model_df = pd.DataFrame(
        model_rows, 
        columns=["Model"] + ["AP@50", "AP@75", "AP@95", "AP@50:95", "AR@50", "AR@75", "AR@95", "AR@50:95"]
    )
    
    # Bold the largest value in each column
    for col in model_df.columns[1:]:
        max_value = model_df[col].max()
        # Format all values; bold the maximum
        model_df[col] = model_df[col].apply(
            lambda x: f"\\textbf{{{x:.4f}}}" if x == max_value else f"{x:.4f}"
        )
    
    # Append the processed rows for this model
    processed_rows.extend(model_df.values.tolist())
    # Add a separator line after each model's methods
    processed_rows.append(["\\midrule"] + [""] * 8)

# Remove the last separator line
if processed_rows[-1][0] == "\\midrule":
    processed_rows = processed_rows[:-1]

# Create a DataFrame for LaTeX formatting
columns = [
    "Model", 
    "AP$^{50}_{\\text{mask}}$", "AP$^{75}_{\\text{mask}}$", "AP$^{95}_{\\text{mask}}$", 
    "AP$^{50:95}_{\\text{mask}}$", "AR$^{50}_{\\text{mask}}$", "AR$^{75}_{\\text{mask}}$", 
    "AR$^{95}_{\\text{mask}}$", "AR$^{50:95}_{\\text{mask}}$"
]
latex_table_data = pd.DataFrame(processed_rows, columns=columns)

# Generate the LaTeX table
latex_code = r"""
\begin{table*}[h!]
\centering
\begin{adjustbox}{width=\textwidth}
\begin{tabular}{@{}l|c|c|c|c|c|c|c|c@{}}
\toprule
\textbf{Model} & \textbf{AP$^{50}_{\text{mask}}$} & \textbf{AP$^{75}_{\text{mask}}$} & \textbf{AP$^{95}_{\text{mask}}$} & \textbf{AP$^{50:95}_{\text{mask}}$} & \textbf{AR$^{50}_{\text{mask}}$} & \textbf{AR$^{75}_{\text{mask}}$} & \textbf{AR$^{95}_{\text{mask}}$} & \textbf{AR$^{50:95}_{\text{mask}}$} \\ \midrule
"""

# Add rows to the LaTeX table
for _, row in latex_table_data.iterrows():
    if row["Model"] == "\\midrule":
        # If it's a separator, just add it
        latex_code += row["Model"] + "\n"
    else:
        row_data = " & ".join(row)
        latex_code += row_data + r" \\ " + "\n"

# Close the LaTeX table
latex_code += r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Performance comparison of YOLO models with SegRefiner (Small), SegRefiner (Large), DualSight (Ours), and Base methods on segmentation metrics. Best results are highlighted in \textbf{bold}.}
\label{tab:performance_comparison}
\end{table*}
"""

# Save the LaTeX code to a file
with open("output_table.tex", "w") as f:
    f.write(latex_code)

# Print the LaTeX code (optional)
print(latex_code)
