import os
import json
import pandas as pd

def escape_latex(s):
    """
    Escapes LaTeX special characters if needed.
    """
    special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
    }
    for char, escape_seq in special_chars.items():
        s = s.replace(char, escape_seq)
    return s

def determine_model_category(model_name):
    """
    Determines a canonical model category from the 'Model' string (case-insensitive).
    e.g. 'YOLOv8n' or 'yolov8 nano' → 'YOLOv8n'
         'YOLOv8x' or 'yolov8 xl'   → 'YOLOv8x'
         'mask r-cnn' or 'mask rcnn' → 'Mask R-CNN'
    """
    lower_name = str(model_name).lower()
    if "nano" in lower_name or "yolov8n" in lower_name:
        return "YOLOv8n"
    elif "xl" in lower_name or "yolov8x" in lower_name:
        return "YOLOv8x"
    elif "mask r-cnn" in lower_name or "mask rcnn" in lower_name:
        return "Mask R-CNN"
    return None

def summarize_and_build_tables(doi_csv_path, results_json_dir, output_csv_path):
    """
    1. Reads the main CSV (DOI.csv or processed CSV).
    2. Scans a directory of JSON results for matching config_ids.
    3. Extracts multiple metrics and merges them into the DataFrame as new columns.
    4. Prints summary info (collected count, remaining, etc.).
    5. Builds two LaTeX tables:
        - Full Table: Includes all specified metrics.
        - Mini Table: Includes a subset of metrics.
    6. Writes the updated DataFrame out to a new CSV.
    """
    # -- 1) Read the CSV
    df = pd.read_csv(doi_csv_path)
    
    # -- 2) Collect ID -> metrics from JSON files
    # Define the metrics to extract
    metrics_to_extract = [
        'sam_mask_AP@50', 'sam_mask_AP@75', 'sam_mask_AP@95', 'sam_mask_AP@50:95',
        'sam_mask_AR@50', 'sam_mask_AR@75', 'sam_mask_AR@95', 'sam_mask_AR@50:95'
    ]
    
    # Initialize a dictionary to hold all metrics
    id_to_metrics = {}
    for filename in os.listdir(results_json_dir):
        if not filename.endswith(".json"):
            continue
        full_path = os.path.join(results_json_dir, filename)
        
        try:
            with open(full_path, "r") as f:
                data = json.load(f)
            
            config_id = data.get("config_id", None)
            avg_metrics = data.get("average_metrics", {})
            
            if config_id is not None:
                # Initialize inner dictionary if not present
                if config_id not in id_to_metrics:
                    id_to_metrics[config_id] = {}
                for metric in metrics_to_extract:
                    metric_val = avg_metrics.get(metric, None)
                    id_to_metrics[config_id][metric] = metric_val
                    
        except Exception as e:
            print(f"[WARNING] Could not parse JSON file '{filename}': {e}")
    
    # -- 3) Merge the new metric columns into df
    for metric in metrics_to_extract:
        df[metric] = df["ID"].apply(lambda x: id_to_metrics.get(x, {}).get(metric, None))
    
    # -- 4) Print out progress information
    total_rows = len(df)
    collected_metrics = {metric: df[metric].notnull().sum() for metric in metrics_to_extract}
    remaining_metrics = {metric: total_rows - collected_metrics[metric] for metric in metrics_to_extract}
    
    print(f"Number of rows in CSV: {total_rows}")
    for metric in metrics_to_extract:
        print(f"Metric '{metric}': Collected {collected_metrics[metric]}, Remaining {remaining_metrics[metric]}, "
              f"Percent complete: {(collected_metrics[metric]/total_rows)*100:.2f}%")
    
    # -- 5) Define competitor IDs
    competitor_ids = {
        "ZFR": [770, 2114, 3458],        # YOLOv8n, YOLOv8x, Mask R-CNN
        "DualSight": [881, 2225, 3569],  # YOLOv8n, YOLOv8x, Mask R-CNN
    }
    all_competitor_ids = set(competitor_ids["ZFR"] + competitor_ids["DualSight"])
    
    # -- 6) Define specified DSv2 IDs
    dsv2_ids = {
        "YOLOv8n": 836,
        "YOLOv8x": 2179,
        "Mask R-CNN": 3578,
    }
    
    # -- 7) Initialize structured_rows
    structured_rows = {
        "YOLOv8n":   {"ZFR": None, "DualSight": None, "DSv2": None},
        "YOLOv8x":   {"ZFR": None, "DualSight": None, "DSv2": None},
        "Mask R-CNN":{"ZFR": None, "DualSight": None, "DSv2": None},
    }
    
    # -- 8) Define the format_row function for multiple metrics
    def format_row(row, label=None, bold=False, selected_metrics=None):
        """
        Return a list: [<ModelLabel>, <Metric1>, <Metric2>, ...]
        If bold is True, bold the metric values.
        selected_metrics is a list specifying the order of metrics to include.
        """
        if selected_metrics is None:
            selected_metrics = metrics_to_extract  # default to all metrics
        
        metric_vals = []
        for metric in selected_metrics:
            val = row.get(metric, "N/A")
            if pd.notnull(val):
                try:
                    val = f"{float(val):.4f}"
                except ValueError:
                    val = "N/A"
            else:
                val = "N/A"
            
            if bold and metric == 'sam_mask_AP@50:95':
                if val != "N/A":
                    val = f"\\textbf{{{val}}}"
            
            metric_vals.append(val)
        
        # if no label was given, build one from e.g. "ID" 
        if label is None:
            label = f"{row.get('ID', 'unknown')}"
        # Escape label for LaTeX
        label = escape_latex(str(label))
        
        return [label] + metric_vals
    
    # -- 9) Fill in ZFR and DualSight from competitor IDs
    for approach, ids in competitor_ids.items():
        for cid in ids:
            # Find that row in df
            row_match = df.loc[df["ID"] == cid]
            if row_match.empty:
                print(f"[WARNING] No row found for competitor ID={cid} ({approach})")
                continue
            row_data = row_match.iloc[0]
            # Determine which model category
            model_cat = determine_model_category(row_data["Model"])
            if model_cat not in structured_rows:
                print(f"[WARNING] Unknown model category for ID={cid}: {row_data['Model']}")
                continue
            
            # Build the row list with all metrics
            row_list = format_row(row_data, label=f"{approach} ({cid})", bold=False)
            structured_rows[model_cat][approach] = row_list
    
    # -- 10) Assign DSv2 entries using specified IDs
    for model_cat, ds_id in dsv2_ids.items():
        row_match = df.loc[df["ID"] == ds_id]
        if row_match.empty:
            print(f"[WARNING] No row found for DSv2 ID={ds_id} ({model_cat})")
            # Fill metrics with "N/A"
            na_metrics = ["N/A"] * len(metrics_to_extract)
            row_list = [escape_latex(f"DSv2 ({ds_id})")] + na_metrics
        else:
            row_data = row_match.iloc[0]
            # Bold the AP^{50:95}_mask metric
            row_list = format_row(row_data, label=f"DSv2 ({ds_id})", bold=True, 
                                  selected_metrics=['sam_mask_AP@50', 'sam_mask_AP@75', 
                                                    'sam_mask_AP@95', 'sam_mask_AP@50:95',
                                                    'sam_mask_AR@50', 'sam_mask_AR@75', 
                                                    'sam_mask_AR@95', 'sam_mask_AR@50:95'])
        structured_rows[model_cat]["DSv2"] = row_list
    
    # -- 11) Assign ModelCategory
    df["ModelCategory"] = df["Model"].apply(determine_model_category)
    
    # -- 12) Build the final table rows for the full table
    final_order = [
        ("YOLOv8n", "ZFR"),
        ("YOLOv8n", "DualSight"),
        ("YOLOv8n", "DSv2"),
        ("YOLOv8x", "ZFR"),
        ("YOLOv8x", "DualSight"),
        ("YOLOv8x", "DSv2"),
        ("Mask R-CNN", "ZFR"),
        ("Mask R-CNN", "DualSight"),
        ("Mask R-CNN", "DSv2"),
    ]
    final_rows_full = []
    for cat, approach in final_order:
        row_list = structured_rows[cat].get(approach)
        if row_list is None:
            # Fill with N/A if missing
            na_metrics = ["N/A"] * len(metrics_to_extract)
            row_list = [escape_latex(f"{approach} (N/A)")] + na_metrics
        final_rows_full.append(row_list)
    
    # -- 13) Generate LaTeX code for the full table
    # Define column headers
    full_table_columns = [
        r"\textbf{Model}",
        r"\textbf{AP$^{50}_{\text{mask}}$}",
        r"\textbf{AP$^{75}_{\text{mask}}$}",
        r"\textbf{AP$^{95}_{\text{mask}}$}",
        r"\textbf{AP$^{50:95}_{\text{mask}}$}",
        r"\textbf{AR$^{50}_{\text{mask}}$}",
        r"\textbf{AR$^{75}_{\text{mask}}$}",
        r"\textbf{AR$^{95}_{\text{mask}}$}",
        r"\textbf{AR$^{50:95}_{\text{mask}}$} \\ \midrule"
    ]
    
    latex_code_full = r"""
\begin{table}[h!]
\centering
\begin{tabular}{l|cccc|cccc}
\toprule
""" + " & ".join(full_table_columns) + "\n"
    
    for row_list in final_rows_full:
        model_label = row_list[0]
        metrics = row_list[1:]
        latex_code_full += f"{model_label} & " + " & ".join(metrics) + r" \\" + " \n"
    
    latex_code_full += r"""
\bottomrule
\end{tabular}
\caption{Comparison of multiple sam\_mask metrics for YOLOv8n, YOLOv8x, and Mask R-CNN. 
Best DSv2 experiments are shown with bolded AP$^{50:95}_{\text{mask}}$ metric.}
\label{tab:full_comparison}
\end{table}
"""
    
    # -- 14) Build the final table rows for the mini table
    # Define the subset of metrics for the mini table
    mini_selected_metrics = [
        'sam_mask_AP@50', 'sam_mask_AP@50:95',
        'sam_mask_AR@50', 'sam_mask_AR@50:95'
    ]
    
    
    # Re-initialize structured_rows for the mini table
    # Since the full table has already been populated, reuse it
    final_rows_mini = []
    for row_list in final_rows_full:
        # Extract only the required metrics
        # The indices are: 0 (Model), 1 (AP50), 4 (AP50:95), 5 (AR50), 8 (AR50:95)
        # Assuming the order in metrics_to_extract
        # metrics_to_extract = [
        #     'sam_mask_AP@50', 'sam_mask_AP@75', 'sam_mask_AP@95', 'sam_mask_AP@50:95',
        #     'sam_mask_AR@50', 'sam_mask_AR@75', 'sam_mask_AR@95', 'sam_mask_AR@50:95'
        # ]
        # Indices needed: 0, 3, 4, 7
        model_label = row_list[0]
        ap50 = row_list[1]
        ap50_95 = row_list[4]
        ar50 = row_list[5]
        ar50_95 = row_list[8]
        mini_row = [model_label, ap50, ap50_95, ar50, ar50_95]
        final_rows_mini.append(mini_row)
    
    # -- 15) Generate LaTeX code for the mini table
    mini_table_columns = [
        r"\textbf{Model}",
        r"\textbf{AP$^{50}_{\text{mask}}$}",
        r"\textbf{AP$^{50:95}_{\text{mask}}$}",
        r"\textbf{AR$^{50}_{\text{mask}}$}",
        r"\textbf{AR$^{50:95}_{\text{mask}}$} \\ \midrule"
    ]
    
    latex_code_mini = r"""
\begin{table}[h!]
\centering
\begin{tabular}{l|c|c|c|c}
\toprule
""" + " & ".join(mini_table_columns) + "\n"
    
    for row_list in final_rows_mini:
        model_label = row_list[0]
        ap50 = row_list[1]
        ap50_95 = row_list[2]
        ar50 = row_list[3]
        ar50_95 = row_list[4]
        latex_code_mini += f"{model_label} & {ap50} & {ap50_95} & {ar50} & {ar50_95} \\\\ \n"
    
    latex_code_mini += r"""
\bottomrule
\end{tabular}
\caption{Comparison of selected sam\_mask metrics for YOLOv8n, YOLOv8x, and Mask R-CNN. 
Best DSv2 experiments are shown with bolded AP$^{50:95}_{\text{mask}}$ metric.}
\label{tab:mini_comparison}
\end{table}
"""
    
    # -- 16) Save LaTeX tables to files
    with open("ablation_table.tex", "w") as f_full:
        f_full.write(latex_code_full)
    
    with open("ablation_table_mini.tex", "w") as f_mini:
        f_mini.write(latex_code_mini)
    
    print("\nLaTeX tables saved to 'ablation_table.tex' and 'ablation_table_mini.tex'.")
    
    # -- 17) Write out the updated DataFrame with all new metric columns to disk.
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV with all metrics saved to: {output_csv_path}")

if __name__ == "__main__":
    summarize_and_build_tables(
        doi_csv_path="/home/sprice/ICCV25/DualSight/ablationStudy/DOI.csv",
        results_json_dir="/home/sprice/ICCV25/DualSight/ablationStudy/ablation_outputs",
        output_csv_path="processedResults.csv"
    )
