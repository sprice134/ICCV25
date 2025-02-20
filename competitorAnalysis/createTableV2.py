import pandas as pd

def csvs_to_latex_table(file_paths, method_names, output_file="output_table.tex"):
    """
    Given a list of CSV file paths and corresponding method names,
    read each CSV file and extract the performance metrics.

    Each CSV is assumed to have up to 3 rows (one per model) and the following columns:
      - "Run_Name" (the model name)
      - 10 metrics:
            refined_mask_AP@50, refined_mask_AP@75, refined_mask_AP@95, refined_mask_AP@50:95,
            refined_mask_AR@50, refined_mask_AR@75, refined_mask_AR@95, refined_mask_AR@50:95,
            refined_box_AP@50:95, refined_box_AR@50:95

    For each model (row in the CSV), the output contains one row per file.
    The first column is the model label. For the Base method the model name is printed in italics
    with a citation, and for other methods a “+ ” is prefixed along with a citation if available.
    The following 10 columns are the metric values (formatted with 4 decimals).

    After processing each model group a midrule is inserted.
    In each group, for each metric column the maximum value is bolded.

    The resulting LaTeX table uses a tabular environment with header and caption.
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
        'refined_mask_AR@50:95',
        'refined_box_AP@50:95',
        'refined_box_AR@50:95'
    ]

    # Define citation mappings
    model_citations = {
        "YOLOv8 Nano": "yolov8architecture",
        "YOLOv8 X-Large": "yolov8architecture",
        "Mask R-CNN": "he2017mask",
        "Mask2Former": "cheng2022masked"
    }
    method_citations = {
        "CascadePSP": "cheng2020cascadepsp",
        "SegRefiner - Small": "wang2023segrefiner",
        "SegRefiner - Large": "wang2023segrefiner"
        # For SAM (This Work) or other methods, no citation is added.
    }

    # Read all CSV files
    dfs = [pd.read_csv(fp) for fp in file_paths]

    # Get the list of models from the first CSV file's "Run_Name" column.
    # (It is assumed that all CSVs should correspond to the same models.)
    models = dfs[0]["Run_Name"].tolist()  # e.g., ['YOLOv8 Nano', 'YOLOv8 X-Large', 'Mask R-CNN']

    # For each CSV, if it has at least 2 rows but is missing some models, fill in the missing rows with zeros.
    for idx, df in enumerate(dfs):
        if df.shape[0] < 1:
            raise ValueError(f"File {file_paths[idx]} has fewer than 1 rows.")
        # If a file has fewer rows than the reference number of models, add rows.
        if df.shape[0] < len(models):
            # For every missing row, create a row with the proper model name and zeros for metrics.
            for j in range(df.shape[0], len(models)):
                new_row = {"Run_Name": models[j]}
                for m in metrics:
                    new_row[m] = 0
                # Append the new row.
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            dfs[idx] = df

    # Build the rows.
    # For each model, we will have one row per CSV (i.e. per method).
    rows = []
    for model_idx, model in enumerate(models):
        for df, method in zip(dfs, method_names):
            metric_values = df.loc[model_idx, metrics].tolist()
            try:
                formatted_metrics = [float(v) for v in metric_values]
            except ValueError:
                raise ValueError("Could not convert metric value to float; check CSV format.")
            
            # Build the model label:
            if method == "Base":
                # For the base method, italicize the model name and append its citation (if exists)
                citation = model_citations.get(model, "")
                model_label = f"\\textit{{{model}}}" + (f" \\cite{{{citation}}}" if citation else "")
            else:
                # For additional methods, prefix with a plus sign and add citation if available.
                citation = method_citations.get(method, "")
                model_label = f"+ {method}" + (f" \\cite{{{citation}}}" if citation else "")
            
            row = [model_label] + formatted_metrics
            rows.append(row)
        # Add a separator row after each model (except the last)
        if model_idx < len(models) - 1:
            rows.append(["\\midrule"] + [""] * len(metrics))
    
    # Define column headers for the table
    columns = [
        "Model",
        "AP$^{50}_{\\text{mask}}$", "AP$^{75}_{\\text{mask}}$", "AP$^{95}_{\\text{mask}}$", "AP$^{50:95}_{\\text{mask}}$",
        "AR$^{50}_{\\text{mask}}$", "AR$^{75}_{\\text{mask}}$", "AR$^{95}_{\\text{mask}}$", "AR$^{50:95}_{\\text{mask}}$",
        "AP$^{50:95}_{\\text{box}}$", "AR$^{50:95}_{\\text{box}}$"
    ]
    table_df = pd.DataFrame(rows, columns=columns)

    # Process groups (each model's block) to bold the maximum value in each metric column.
    processed_rows = []
    current_group = []

    def process_group(group):
        """ Bold-format the highest value (per metric) in the given group of rows. """
        if not group:
            return []
        group_df = pd.DataFrame(group, columns=columns)
        # For each metric column (skip the "Model" column)
        for col in columns[1:]:
            group_df[col] = group_df[col].astype(float)
            max_val = group_df[col].max()
            group_df[col] = group_df[col].apply(
                lambda x: f"\\textbf{{{x:.4f}}}" if x == max_val else f"{x:.4f}"
            )
        return group_df.astype(str).values.tolist()

    for row in rows:
        if row[0] == "\\midrule":
            processed_rows.extend(process_group(current_group))
            processed_rows.append(["\\midrule"] + [""] * len(metrics))
            current_group = []
        else:
            current_group.append(row)
    if current_group:
        processed_rows.extend(process_group(current_group))

    # Build the LaTeX code.
    header = r"""\begin{table*}[h!]
\centering
\begin{adjustbox}{width=\textwidth}
\begin{tabular}{@{}l|c|c|c|c|c|c|c|c|c|c@{}}
\toprule
\textbf{Model} & \textbf{AP$^{50}_{\text{mask}}$} & \textbf{AP$^{75}_{\text{mask}}$} & \textbf{AP$^{95}_{\text{mask}}$} & \textbf{AP$^{50:95}_{\text{mask}}$} & \textbf{AR$^{50}_{\text{mask}}$} & \textbf{AR$^{75}_{\text{mask}}$} & \textbf{AR$^{95}_{\text{mask}}$} & \textbf{AR$^{50:95}_{\text{mask}}$} & \textbf{AP$^{50:95}_{\text{box}}$} & \textbf{AR$^{50:95}_{\text{box}}$} \\ \midrule
"""
    body = ""
    for row in processed_rows:
        if row[0] == "\\midrule":
            body += "\\midrule\n"
        else:
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

    with open(output_file, "w") as f:
        f.write(latex_code)

    print("LaTeX table written to", output_file)
    print(latex_code)


if __name__ == '__main__':
    # Example usage: adjust file paths and method names as needed.
    file_paths = [
        "metrics/powder_raw.csv",
        "metrics/powder_cascadePSP.csv",
        "metrics/powder_segrefinerSmall.csv",
        "metrics/powder_segrefinerLarge.csv",
        "metrics/powder_dualsight_v2.csv"
    ]
    method_names = [
        "Base",
        "CascadePSP",
        "SegRefiner - Small",
        "SegRefiner - Large",
        "SAM (This Work)"
    ]
    
    csvs_to_latex_table(file_paths, method_names, output_file="output_table.tex")
