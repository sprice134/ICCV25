#!/bin/bash
set -e
source ../dsEnv/bin/activate

INFERENCE_PKL_1="../savedInference/particle_yolov8n_inference.pkl"
INFERENCE_PKL_2="../savedInference/particle_yolov8x_inference.pkl"
INFERENCE_PKL_3="../savedInference/particle_maskrcnn_inference.pkl"
INFERENCE_PKL_4="../savedInference/particle_mask2former_inference.pkl"

IMAGES_DIR="../datasets/powder/test"
OUTPUT_DIR="dualsight_v2_output"
OUTPUT_CSV="metrics/powder_dualsight_v2.csv"

mkdir -p "$OUTPUT_DIR"
[ -f "$OUTPUT_CSV" ] && rm "$OUTPUT_CSV"

run_sam() {
    local pkl_path="$1"
    local run_name="$2"
    echo "Running SAM Ablation for run: $run_name with pickle: $pkl_path"
    python sam_v2.py \
        --inference_pickle_path "$pkl_path" \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --sam_checkpoint "../modelWeights/sam_vit_l.pth" \
        --num_points 3 \
        --ignore_border_percentage 10 \
        --algorithm "Voronoi" \
        --use_box_input True \
        --use_mask_input False \
        --box_expansion_rate 1.0 \
        --mask_expansion_rate 0.0 \
        --run_name "$run_name" \
        --output_csv "$OUTPUT_CSV"
    echo "Completed run: $run_name"
    echo "----------------------------------------"
}

run_sam "$INFERENCE_PKL_1" "YOLOv8 Nano"
run_sam "$INFERENCE_PKL_2" "YOLOv8 XL"
run_sam "$INFERENCE_PKL_3" "Mask R-CNN"
run_sam "$INFERENCE_PKL_4" "Mask2Former"

echo "All runs completed. Metrics have been aggregated in $OUTPUT_CSV."
