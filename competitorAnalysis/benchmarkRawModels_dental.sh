#!/bin/bash

# --------------------------------------------
# run_raw_evaluation.sh
#
# This script evaluates raw model predictions without refinement.
# It runs evaluate_raw_models.py for different inference pickle files
# and aggregates their metrics into a single CSV.
# --------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e

# Activate the appropriate Python environment
# Replace 'your_env' with the actual environment name
source ../cascadeEnv/bin/activate

# Define paths to the inference pickle files
INFERENCE_PKL_1="../savedInference/dental_yolov8n_inference.pkl"
INFERENCE_PKL_2="../savedInference/dental_yolov8x_inference.pkl"

# Define other common parameters
IMAGES_DIR="../datasets/dental/valid/images"
OUTPUT_DIR="raw_dental_output"
OUTPUT_CSV="metrics_raw_dental.csv"

# Ensure the images directory exists
if [ ! -d "$IMAGES_DIR" ]; then
    echo "[ERROR] Images directory '$IMAGES_DIR' does not exist."
    exit 1
fi

# Remove the existing CSV if it exists to start fresh
if [ -f "$OUTPUT_CSV" ]; then
    rm "$OUTPUT_CSV"
    echo "Removed existing '$OUTPUT_CSV' to start fresh."
fi

# Function to run evaluate_raw_models.py
run_evaluation() {
    local pkl_path="$1"
    local run_name="$2"

    echo "----------------------------------------"
    echo "Running evaluation for run: $run_name"
    echo "Inference Pickle: $pkl_path"
    echo "----------------------------------------"

    python rawModels.py \
        --inference_pickle_path "$pkl_path" \
        --images_dir "$IMAGES_DIR" \
        --output_csv "$OUTPUT_CSV" \
        --run_name "$run_name"

    echo "Completed evaluation for run: $run_name"
    echo "----------------------------------------"
}

# Run evaluations for each model
run_evaluation "$INFERENCE_PKL_1" "YOLOv8 Nano"
run_evaluation "$INFERENCE_PKL_2" "YOLOv8 X-Large"

echo "All evaluations completed. Metrics have been aggregated in '$OUTPUT_CSV'."
