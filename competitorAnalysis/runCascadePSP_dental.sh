#!/bin/bash

# --------------------------------------------
# run_cascadePSP.sh
#
# This script runs cascadePSP.py three times with different
# inference pickle files and aggregates their metrics into a single CSV.
# --------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e

source ../cascadeEnv/bin/activate


# Define paths to the inference pickle files
INFERENCE_PKL_1="../savedInference/dental_yolov8n_inference.pkl"
INFERENCE_PKL_2="../savedInference/dental_yolov8x_inference.pkl"

# Define other common parameters
IMAGES_DIR="../datasets/dental/valid/images"
OUTPUT_DIR="cascade_output"
OUTPUT_CSV="metrics_cascadePSP_dental.csv"



# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Remove the existing CSV if it exists to start fresh
if [ -f "$OUTPUT_CSV" ]; then
    rm "$OUTPUT_CSV"
fi


# Function to run cascadePSP.py
run_cascadePSP() {
    local pkl_path="$1"
    local run_name="$2"

    echo "Running cascadePSP.py for run: $run_name with pickle: $pkl_path"

    python cascadePSP.py \
        --inference_pickle_path "$pkl_path" \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$run_name" \
        --output_csv "$OUTPUT_CSV"

    echo "Completed run: $run_name"
    echo "----------------------------------------"
}

# Run cascadePSP.py for each inference pickle
run_cascadePSP "$INFERENCE_PKL_1" "YOLOv8 Nano"
run_cascadePSP "$INFERENCE_PKL_2" "YOLOv8 X-Large"

echo "All runs completed. Metrics have been aggregated in $OUTPUT_CSV."
