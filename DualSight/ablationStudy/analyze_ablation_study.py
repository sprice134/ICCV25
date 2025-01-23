import os
import json
import pandas as pd

def summarize_sam_mask_results(doi_csv_path, results_json_dir, output_csv_path):
    """
    Reads the DOI CSV file, scans a directory of JSON results for matching config_ids,
    extracts the sam_mask_AP@50:95 value, and outputs an updated CSV with the new column.
    Also prints:
      - the number of collected results,
      - the number of remaining experiments,
      - the percent completion,
      - and, for each model, the top three parameter sets (by sam_mask_AP@50:95).
      
    :param doi_csv_path: Path to the DOI CSV file.
    :param results_json_dir: Directory containing the JSON result files.
    :param output_csv_path: Path to the output CSV file to be created.
    """
    # 1) Read the DOI CSV into a pandas DataFrame
    doi_df = pd.read_csv(doi_csv_path)
    
    # 2) Make a dict to store ID -> sam_mask_AP@50:95
    id_to_sam_mask_score = {}

    # 3) Iterate through JSON files in the results directory
    for filename in os.listdir(results_json_dir):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(results_json_dir, filename)
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                
            # We expect something like: data["config_id"] and data["average_metrics"]["sam_mask_AP@50:95"]
            config_id = data.get("config_id", None)
            avg_metrics = data.get("average_metrics", {})
            sam_mask_val = avg_metrics.get("sam_mask_AP@50:95", None)
            
            if config_id is not None and sam_mask_val is not None:
                id_to_sam_mask_score[config_id] = sam_mask_val
        
        except Exception as e:
            print(f"[WARNING] Could not parse JSON file '{filename}': {e}")

    # 4) Create a new column in doi_df to hold the average sam_mask_AP@50:95 value
    #    We'll name it "avg_sam_mask_AP_50_95" for convenience
    doi_df["avg_sam_mask_AP_50_95"] = doi_df["ID"].apply(lambda x: id_to_sam_mask_score.get(x, None))

    # 5) Print out progress information
    total_rows = len(doi_df)
    collected = doi_df["avg_sam_mask_AP_50_95"].notnull().sum()
    remaining = total_rows - collected
    pct_complete = (collected / total_rows) * 100 if total_rows > 0 else 0

    print(f"Number of collected ones: {collected}")
    print(f"Number of remaining experiments to go: {remaining}")
    print(f"Percent complete: {pct_complete:.2f}%")

    # 6) For each model, print the top three approaches and their scores
    #    (Approach = row with unique param sets, but we’ll just treat each row as a distinct approach.)
    #    We'll sort the DataFrame within each model group by the new score column, descending.
    
    # If your original CSV might have multiple distinct "Model" entries, let's group by "Model":
    grouped = doi_df.groupby("Model", dropna=False)

    for model_name, group_df in grouped:
        # Sort the group by avg_sam_mask_AP_50_95 descending
        sorted_group = group_df.sort_values(by="avg_sam_mask_AP_50_95", ascending=False, na_position="last")
        
        # Take the top three
        top_three = sorted_group.head(3)
        
        print(f"\n=== Top 3 for Model: {model_name} ===")
        for idx, row in top_three.iterrows():
            # We can show relevant parameters (or all from your CSV row).
            # For clarity, let's just show the major columns:
            rid = row["ID"]
            box_inclusion = row["BoxInclusion"]
            mask_inclusion = row["MaskInclusion"]
            num_pois = row["NumberOfPOIs"]
            poi_alg = row["POIPlacementAlgorithm"]
            perimeter_buf = row["PerimeterBuffer"]
            box_dist = row["BoundingBoxDistortion"]
            mask_dist = row["MaskDistortion"]
            score = row["avg_sam_mask_AP_50_95"]
            comp = rid in [770, 881, 2114, 2225, 3458, 3569]
            print(f"  ID={rid}, Competitor={comp}, Box={box_inclusion}, Mask={mask_inclusion}, "
                f"POIs={num_pois}, Algorithm={poi_alg}, PerimeterBuffer={perimeter_buf}, "
                f"BoxDist={box_dist}, MaskDist={mask_dist}, Score={score:.5f}")


    # 7) Write out the updated DataFrame to a new CSV
    doi_df.to_csv(output_csv_path, index=False)
    print(f"\nDone! Updated CSV saved to: {output_csv_path}")

if __name__ == '__main__':
    summarize_sam_mask_results('/home/sprice/ICCV25/DualSight/ablationStudy/DOI.csv', 
                               '/home/sprice/ICCV25/DualSight/ablationStudy/ablation_outputs',
                                'processedResults.csv')