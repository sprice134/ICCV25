#!/bin/bash

# --------------------------------------------
# runSegrefiner.sh
#
# This script runs segrefiner.py multiple times with different
# inference pickle files and aggregates their metrics into a single CSV.
# --------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e

# (Optional) Activate your Python virtual environment
source ../segrefinerEnv/bin/activate

# ---------------------------------------------------------
# Modify these paths as needed for your environment
# ---------------------------------------------------------

# Paths to the inference pickle files
INFERENCE_PKL_1="../DualSight/ablationStudy/inference_outputs/yolov8n_inference.pkl"
INFERENCE_PKL_2="../DualSight/ablationStudy/inference_outputs/yolov8x_inference.pkl"
INFERENCE_PKL_3="../DualSight/ablationStudy/inference_outputs/maskrcnn_inference.pkl"

# Common parameters
IMAGES_DIR="../demo.v7i.coco/test"
OUTPUT_DIR="segrefiner_refined_outputs"

# Path to SegRefiner config and checkpoint
SEGREFINER_CONFIG="/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py"
SEGREFINER_CKPT="/home/sprice/ICCV25/modelWeights/segrefiner_lr_latest.pth"

# Output CSV file
OUTPUT_CSV="segrefinerMetrics.csv"

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

    echo "Running segrefiner.py for run: $run_name with pickle: $pkl_path"

    python segrefiner.py \
        --inference_pickle_path "$pkl_path" \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --segrefiner_config "$SEGREFINER_CONFIG" \
        --segrefiner_checkpoint "$SEGREFINER_CKPT" \
        --run_name "$run_name" \
        --output_csv "$OUTPUT_CSV"

    echo "Completed run: $run_name"
    echo "----------------------------------------"
}

# Run segrefiner.py for each inference pickle
run_segrefiner "$INFERENCE_PKL_1" "YOLOv8 Nano"
run_segrefiner "$INFERENCE_PKL_2" "YOLOv8 X-Large"
run_segrefiner "$INFERENCE_PKL_3" "Mask R-CNN"

echo "All runs completed. Metrics have been aggregated in $OUTPUT_CSV."
