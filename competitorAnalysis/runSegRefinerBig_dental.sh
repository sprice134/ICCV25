#!/bin/bash

# --------------------------------------------
# runSegrefiner_large.sh
#
# This script runs segrefiner.py using the big SegRefiner model
# and aggregates their metrics into a single CSV.
# --------------------------------------------

source ../segrefinerEnv/bin/activate

# ---------------------------------------------------------
# Modify these paths as needed for your environment
# ---------------------------------------------------------

# Paths to the inference pickle files
INFERENCE_PKL_1="../savedInference/dental_yolov8n_inference.pkl"
INFERENCE_PKL_2="../savedInference/dental_yolov8x_inference.pkl"

# Define other common parameters
IMAGES_DIR="../datasets/dental/valid/images"
OUTPUT_DIR="segrefiner_dental_output"
OUTPUT_CSV="metrics_segrefinerLarge_dental.csv"

# Path to SegRefiner config and big checkpoint
SEGREFINER_CONFIG="/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py"
SEGREFINER_CKPT="/home/sprice/ICCV25/modelWeights/segrefiner_hr_latest.pth"


# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Remove the existing CSV if you want to start fresh
if [ -f "$OUTPUT_CSV" ]; then
    rm "$OUTPUT_CSV"
fi

# Function to run segrefiner.py
run_segrefiner() {
    local pkl_path="$1"
    local run_name="$2"

    echo "Running segrefiner.py for run: $run_name with pickle: $pkl_path using BIG model"

    python segrefiner.py \
        --inference_pickle_path "$pkl_path" \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --segrefiner_config "$SEGREFINER_CONFIG" \
        --segrefiner_checkpoint "$SEGREFINER_CKPT" \
        --run_name "$run_name" \
        --output_csv "$OUTPUT_CSV"

    echo "Completed run: $run_name with BIG model"
    echo "----------------------------------------"
}

# Run segrefiner.py for each inference pickle using the big model
run_segrefiner "$INFERENCE_PKL_1" "YOLOv8 Nano"
run_segrefiner "$INFERENCE_PKL_2" "YOLOv8 X-Large"

echo "All runs with BIG model completed. Metrics have been aggregated in $OUTPUT_CSV."
