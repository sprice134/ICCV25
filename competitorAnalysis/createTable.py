import pandas as pd

def csvs_to_latex_table(file_paths, method_names, output_file="output_table.tex"):
    """
    Given a list of CSV file paths and corresponding method names,
    read each CSV file and extract the performance metrics.
    
    Each CSV is assumed to have 3 rows (one per model) and the following columns:
      - "Run_Name" (the model name)
      - 8 metrics:
            refined_mask_AP@50, refined_mask_AP@75, refined_mask_AP@95, refined_mask_AP@50:95,
            refined_mask_AR@50, refined_mask_AR@75, refined_mask_AR@95, refined_mask_AR@50:95

    The table is built so that for each model (row in the CSV) the output
    contains one row per file. The first column is the model name (if the method is not "Base",
    the method name is appended with a " + "), followed by the 8 metric values (formatted with 4 decimals).
    
    After processing each model (i.e. all method rows for that model) a midrule is inserted.
    
    Additionally, for each model group the maximum value in each metric column is bolded.
    
    The resulting LaTeX table uses a tabular environment with a header and caption.
    """
    if len(file_paths) != len(method_names):
        raise ValueError("The number of file paths must match the number of method names.")

    # Define the list of metric columns (order matters)
    metrics = [
        'refined_mask_AP@50',
        'refined_mask_AP@75',
        'refined_mask_AP@95',
        'refined_mask_AP@50:95',
        'refined_mask_AR@50',
        'refined_mask_AR@75',
        'refined_mask_AR@95',
        'refined_mask_AR@50:95'
    ]

    # Read all CSV files
    dfs = [pd.read_csv(fp) for fp in file_paths]
    
    # Check that each file has 3 rows
    for i, df in enumerate(dfs):
        if df.shape[0] < 3:
            raise ValueError(f"File {file_paths[i]} has fewer than 3 rows.")

    # Get the list of model names from the first CSV file's "Run_Name" column.
    # (It is assumed that all CSVs have the same model order.)
    models = dfs[0]["Run_Name"].tolist()  # e.g., ['YOLOv8 Nano', 'YOLOv8 X-Large', 'Mask R-CNN']

    # Build the stacked rows. We will create a list of lists, where each inner list
    # corresponds to a row of the LaTeX table.
    # The table will have 9 columns: "Model" plus the 8 metrics.
    rows = []
    for model_idx, model in enumerate(models):
        # For each model, go over the CSVs/methods
        for df, method in zip(dfs, method_names):
            # Extract the row corresponding to this model.
            # We assume the CSV row order is consistent.
            metric_values = df.loc[model_idx, metrics].tolist()
            # Format the metric values to 4 decimal places
            try:
                formatted_metrics = [float(v) for v in metric_values]
            except ValueError:
                raise ValueError("Could not convert metric value to float; check CSV format.")
            
            # Build the first column. If the method is "Base", we leave the model name unchanged.
            # Otherwise, append the method name.
            model_label = model if method == "Base" else f"{model} + {method}"
            
            # Create a row: first column is the model label, then the 8 metric numbers.
            row = [model_label] + formatted_metrics
            rows.append(row)
        # After processing one model’s rows, add a separator row (except after the last model)
        if model_idx < len(models) - 1:
            rows.append(["\\midrule"] + [""] * 8)
    
    # Create a DataFrame from the rows to help with formatting and bolding.
    # The columns will be:
    columns = [
        "Model",
        "AP@50", "AP@75", "AP@95", "AP@50:95",
        "AR@50", "AR@75", "AR@95", "AR@50:95"
    ]
    table_df = pd.DataFrame(rows, columns=columns)
    
    # Process groups (each model’s block) to bold the maximum value in each metric column.
    # We scan through the rows, collecting contiguous rows until we hit a separator row ("\\midrule").
    processed_rows = []
    current_group = []
    
    def process_group(group):
        """ Bold-format the highest value (per metric) in the given group of rows. """
        if not group:
            return []
        group_df = pd.DataFrame(group, columns=columns)
        # Bold each metric column: find the maximum (as float) and then reformat each value.
        for col in columns[1:]:
            # Convert column values to float
            group_df[col] = group_df[col].astype(float)
            max_val = group_df[col].max()
            # Apply formatting: if value equals max_val, wrap in \textbf{}, otherwise format normally.
            group_df[col] = group_df[col].apply(
                lambda x: f"\\textbf{{{x:.4f}}}" if x == max_val else f"{x:.4f}"
            )
        # Convert the DataFrame back to a list of lists of strings.
        return group_df.astype(str).values.tolist()
    
    for row in rows:
        if row[0] == "\\midrule":
            # Process the current group and add a midrule row
            processed_rows.extend(process_group(current_group))
            processed_rows.append(["\\midrule"] + [""] * 8)
            current_group = []
        else:
            current_group.append(row)
    # Process the final group if any
    if current_group:
        processed_rows.extend(process_group(current_group))
    
    # Build the LaTeX code manually.
    # Here we use the same header, column names, and footer as in your sample.
    header = r"""\begin{table*}[h!]
\centering
\begin{adjustbox}{width=\textwidth}
\begin{tabular}{@{}l|c|c|c|c|c|c|c|c@{}}
\toprule
\textbf{Model} & \textbf{AP$^{50}_{\text{mask}}$} & \textbf{AP$^{75}_{\text{mask}}$} & \textbf{AP$^{95}_{\text{mask}}$} & \textbf{AP$^{50:95}_{\text{mask}}$} & \textbf{AR$^{50}_{\text{mask}}$} & \textbf{AR$^{75}_{\text{mask}}$} & \textbf{AR$^{95}_{\text{mask}}$} & \textbf{AR$^{50:95}_{\text{mask}}$} \\ \midrule
"""
    body = ""
    for row in processed_rows:
        if row[0] == "\\midrule":
            body += "\\midrule\n"
        else:
            # Join the columns with an ampersand and end with a LaTeX new-line.
            # (Make sure that every cell is a string.)
            row_cells = [str(cell) for cell in row]
            body += " & ".join(row_cells) + r" \\" + "\n"
    footer = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Performance comparison of models. Best results in each group are highlighted in \textbf{bold}.}
\label{tab:performance_comparison}
\end{table*}
"""
    latex_code = header + body + footer
    
    # Save the LaTeX table to file.
    with open(output_file, "w") as f:
        f.write(latex_code)
    
    print("LaTeX table written to", output_file)
    print(latex_code)


if __name__ == '__main__':
    # Example usage:
    # Suppose you have three CSV files corresponding to different methods.
    # Adjust the file paths and method names as needed.
    file_paths = [
        "metrics/powder_raw.csv",
        "metrics/powder_cascadePSP.csv",
        "metrics/powder_segrefinerSmall.csv",
        "metrics/powder_segrefinerLarge.csv",
        "metrics/powder_dualsight_v2.csv"
        # Add more files if needed.
    ]
    method_names = [
        "Base",
        "CascadePSP",
        "SegRefiner - Small",
        "SegRefiner - Large",
        "DualSight (Ours)"
        # The method names must correspond one-to-one with file_paths.
    ]
    
    csvs_to_latex_table(file_paths, method_names, output_file="output_table.tex")

