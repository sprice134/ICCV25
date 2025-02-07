#!/bin/bash
#SBATCH --job-name=testingA             # Job name
#SBATCH --cpus-per-task=64               # Number of CPU cores
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH -C A100|V100|L40S               # A100 or V100 GPU
#SBATCH --mem=124000MB                   # Memory in MB
#SBATCH --time=24:00:00                 # Time limit (HH:MM:SS)
#SBATCH --partition=short               # Partition name


set -e

# Activate your virtual environment
source ../dsEnv/bin/activate

# Define paths (adjust as needed)
INFERENCE_PKL="../DualSight/ablationStudy/inference_outputs/yolov8n_inference.pkl"
IMAGES_DIR="../datasets/powder/test"
SAM_CHECKPOINT="../modelWeights/sam_vit_l.pth"

# Define base output directories
BASE_OUTPUT_DIR="dualsight_experiment_output"
BASE_CSV_DIR="csv_results"

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_CSV_DIR"

# Loop over the grid of parameters:
#   num_points: 2 to 7
#   ignore_border_percentage: 0, 5, 10, 15
#   box_expansion_rate: 0.9, 1.0, 1.1

for num_points in {2..7}; do
  for border in 0 5 10 15; do
    for bbox in 0.9 1.0 1.1; do
      # Construct a run name that encodes the parameters
      run_name="NP${num_points}_Border${border}_BBox${bbox}"
      # Each run gets its own CSV file inside the csv_results directory.
      output_csv="${BASE_CSV_DIR}/metrics_${run_name}.csv"
      
      # Remove any existing CSV file with the same name
      [ -f "$output_csv" ] && rm "$output_csv"
      
      echo "----------------------------------------"
      echo "Running experiment: $run_name"
      
      python sam.py \
         --inference_pickle_path "$INFERENCE_PKL" \
         --images_dir "$IMAGES_DIR" \
         --output_dir "$BASE_OUTPUT_DIR" \
         --sam_checkpoint "$SAM_CHECKPOINT" \
         --num_points "$num_points" \
         --ignore_border_percentage "$border" \
         --algorithm "Voronoi" \
         --use_box_input True \
         --use_mask_input False \
         --box_expansion_rate "$bbox" \
         --mask_expansion_rate 0.0 \
         --run_name "$run_name" \
         --output_csv "$output_csv"
      
      echo "Completed experiment: $run_name"
      echo "----------------------------------------"
    done
  done
done

# After all experiments have finished, search through the CSV files
# and determine which run has the highest average AP@50:95.
# (We assume each CSV contains a header row and a single row of results.)

best_run=""
best_value=0

echo ""
echo "Aggregating results..."

for csv in "${BASE_CSV_DIR}"/*.csv; do
  # Use awk to determine the column index for "AP@50:95" (header row),
  # then extract that value from the second row.
  value=$(awk -F',' '
    NR==1 {
      for(i=1;i<=NF;i++){
        if($i=="AP@50:95"){idx=i; break}
      }
    }
    NR==2 {print $idx}' "$csv")
  
  # Also extract the run name (assumed to be in the first column)
  run=$(awk -F',' 'NR==2 {print $1}' "$csv")
  
  echo "Run: $run, AP@50:95: $value"
  
  # Compare using bc (floating-point comparison)
  if (( $(echo "$value > $best_value" | bc -l) )); then
    best_value="$value"
    best_run="$run"
  fi
done

echo "----------------------------------------"
echo "Best experiment: $best_run with AP@50:95 = $best_value"
