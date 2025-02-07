#!/usr/bin/env python3
"""
sam.py

This script loads a SAM model, reads an inference pickle file (which is expected
to contain predicted boxes, polygons, and masks per image), and then for each image:
  - Loads the image and its corresponding ground truth mask.
  - Processes the predicted masks.
  - Runs SAM inference using the provided SAM parameters.
  - Combines the SAM output masks into a single 16-bit mask.
  - Evaluates the SAM predictions using COCO metrics.
  - Writes aggregate metrics to a CSV file.

Usage example (to be run from a bash script):
    python samAblation.py \
        --inference_pickle_path "../DualSight/ablationStudy/inference_outputs/yolov8n_inference.pkl" \
        --images_dir "../datasets/powder/test" \
        --output_dir "sam_ablation_outputs" \
        --sam_checkpoint "/path/to/sam_checkpoint.pth" \
        --num_points 16 \
        --ignore_border_percentage 0.1 \
        --algorithm "your_algorithm" \
        --use_box_input \
        --use_mask_input \
        --box_expansion_rate 0.05 \
        --mask_expansion_rate 0.1 \
        --run_name "YOLOv8 Nano" \
        --output_csv "samAblationMetrics.csv"
"""

import os
import sys
import pickle
import csv
import json
import numpy as np
import time
import platform
import psutil
import torch
import math
import uuid
from PIL import Image
import argparse
import cv2

from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# Adjust these paths as needed so Python can locate your helper modules.
# sys.path.append('')
sys.path.append('../DualSight')  # Adjust as needed

# Import SAM helpers and evaluation tools
from sam_helper import load_sam_model, run_sam_inference




def combine_masks_16bit(list_of_binary_masks, output_path=None, return_array=False):
    """
    Combine a list of 0/1 or 0/255 binary masks into a single 16-bit instance mask.
    Each mask is assigned a unique ID (1, 2, 3, ...).
    """
    if not list_of_binary_masks:
        print("[WARNING] No masks provided to combine.")
        return None

    # Verify that all masks have the same shape
    first_shape = list_of_binary_masks[0].shape
    for idx, mask in enumerate(list_of_binary_masks):
        if mask.shape != first_shape:
            raise ValueError(f"All masks must have the same shape. Mask at index {idx} has shape {mask.shape}, expected {first_shape}.")

    height, width = first_shape
    combined_16bit = np.zeros((height, width), dtype=np.uint16)

    for idx, mask in enumerate(list_of_binary_masks, start=1):
        unique_vals = set(np.unique(mask))
        # Ensure mask is binary {0,1} or {0,255}
        if mask.dtype != bool and not (unique_vals <= {0, 1} or unique_vals <= {0, 255}):
            raise ValueError(
                f"Mask at index {idx-1} has unexpected unique values {unique_vals}."
                "Please provide 0/1 or 0/255 masks."
            )

        # Convert to boolean
        mask_bool = mask.astype(bool)
        # Assign unique ID
        combined_16bit[mask_bool] = idx

    # Optionally save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_16bit = combined_16bit.astype(np.uint16)
        success = cv2.imwrite(output_path, combined_16bit)
        if not success:
            raise IOError(f"Failed to write the combined mask to {output_path}")

    if return_array:
        return combined_16bit
    return None

def mask_to_coco_format(binary_mask, image_id, category_id=1, annotation_id=1, score=1.0):
    """
    Convert a binary mask (NumPy array) to COCO RLE format.
    """
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')  # decode bytes for JSON
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": float(maskUtils.area(rle)),
        "bbox": maskUtils.toBbox(rle).tolist(),
        "iscrowd": 0,
        "score": score  # Assign default score
    }

def generate_coco_annotations_from_multi_instance_masks_16bit(
    gt_mask_path,
    pred_mask_path,
    image_id=1,
    category_id=1
):
    """
    Generate COCO-style annotations for both ground truth and predicted masks.
    Assumes each pixel in the 16-bit mask corresponds to an instance ID (0 = background).
    """
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)

    if gt_mask is None or pred_mask is None:
        raise FileNotFoundError("One of the mask files was not found.")

    # Identify unique object IDs
    gt_ids = np.unique(gt_mask)
    pred_ids = np.unique(pred_mask)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = pred_ids[pred_ids > 0]

    gt_annotations = []
    pred_annotations = []

    # Create GT annotations
    for annotation_id, obj_id in enumerate(gt_ids, start=1):
        binary_mask = (gt_mask == obj_id).astype(np.uint8)
        gt_annotations.append(
            mask_to_coco_format(binary_mask, image_id, category_id, annotation_id, score=1.0)
        )

    # Create Predicted annotations
    for annotation_id, obj_id in enumerate(pred_ids, start=1):
        binary_mask = (pred_mask == obj_id).astype(np.uint8)
        pred_annotations.append(
            mask_to_coco_format(binary_mask, image_id, category_id, annotation_id, score=1.0)  # Assign score=1.0
        )

    gt_data = {
        "images": [{
            "id": image_id,
            "width": int(gt_mask.shape[1]),
            "height": int(gt_mask.shape[0]),
            "file_name": os.path.basename(gt_mask_path)
        }],
        "annotations": gt_annotations,
        "categories": [{"id": category_id, "name": "object"}]
    }

    return gt_data, pred_annotations

def compute_specific_metrics(coco_eval, max_dets=200):
    """
    Compute specific COCO evaluation metrics.
    """
    coco_eval.params.maxDets = [max_dets, max_dets, max_dets]
    coco_eval.params.iouThrs = [0.5, 0.75, 0.95]

    # Suppress output during evaluation
    saved_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        coco_eval.evaluate()
        coco_eval.accumulate()
    finally:
        sys.stdout.close()
        sys.stdout = saved_stdout

    # Extract precision and recall
    precision = coco_eval.eval['precision']  # shape: [T, R, K, A, M]
    recall = coco_eval.eval['recall']        # shape: [T, K, A, M]

    # Find indices for specific IoU thresholds
    iou_thr_indices = {
        0.5: np.where(np.isclose(coco_eval.params.iouThrs, 0.5))[0][0],
        0.75: np.where(np.isclose(coco_eval.params.iouThrs, 0.75))[0][0],
        0.95: np.where(np.isclose(coco_eval.params.iouThrs, 0.95))[0][0],
    }
    max_det_index = 0  # because we set maxDets list to identical values

    # Compute AP for each IoU threshold
    ap_values = {}
    for thr, idx in iou_thr_indices.items():
        valid_precisions = precision[idx, :, :, :, max_det_index]
        valid_precisions = valid_precisions[valid_precisions != -1]
        ap = np.mean(valid_precisions) if valid_precisions.size > 0 else float('nan')
        ap_values[f"AP@{int(thr*100)}"] = ap

    # Compute AP across IoU thresholds for all areas (AP@50:95)
    all_valid_precisions = precision[:, :, :, :, max_det_index]
    all_valid_precisions = all_valid_precisions[all_valid_precisions != -1]
    ap_values["AP@50:95"] = np.mean(all_valid_precisions) if all_valid_precisions.size > 0 else float('nan')

    # Compute AR for each IoU threshold
    ar_values = {}
    for thr, idx in iou_thr_indices.items():
        valid_recalls = recall[idx, :, :, max_det_index]
        ar = np.mean(valid_recalls) if valid_recalls.size > 0 else float('nan')
        ar_values[f"AR@{int(thr*100)}"] = ar

    # Compute AR across IoU thresholds for all areas (AR@50:95)
    all_valid_recalls = recall[:, :, :, max_det_index]
    ar_values["AR@50:95"] = np.mean(all_valid_recalls) if all_valid_recalls.size > 0 else float('nan')

    # Combine
    metrics = {}
    metrics.update(ap_values)
    metrics.update(ar_values)
    metrics["maxDets"] = max_dets

    return metrics

def evaluate_coco_metrics(gt_data, pred_data, iou_type="segm", max_dets=200):
    """
    Evaluate COCO metrics (AP, AR) given ground truth data and predicted data
    in COCO format. Returns a dictionary of metrics.
    """
    with open("temp_gt.json", "w") as gt_file:
        json.dump(gt_data, gt_file)

    with open("temp_pred.json", "w") as pred_file:
        json.dump(pred_data, pred_file)

    # Load into COCO
    coco_gt = COCO("temp_gt.json")
    coco_pred = coco_gt.loadRes("temp_pred.json")
    coco_eval = COCOeval(coco_gt, coco_pred, iouType=iou_type)

    # Compute metrics
    metrics = compute_specific_metrics(coco_eval, max_dets=max_dets)

    # Remove temporary files
    os.remove("temp_gt.json")
    os.remove("temp_pred.json")

    return metrics


# ------------------ Utility Functions ------------------ #
def safe_get(metrics_dict, key):
    """Return metrics_dict[key] if present, else None."""
    if metrics_dict is not None and key in metrics_dict:
        return metrics_dict[key]
    return None

def average_or_none(values):
    """Return the mean of a list if not empty, else None."""
    return float(np.mean(values)) if values else None

def write_metrics_to_csv(run_name, metrics, output_csv_path):
    """
    Writes the aggregate metrics to a CSV file along with the run_name.
    """
    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csvfile:
        fieldnames = ['Run_Name'] + list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {'Run_Name': run_name}
        row.update(metrics)
        writer.writerow(row)

# ------------------ Main SAM Ablation Function ------------------ #
def sam_inference_and_evaluate(
    inference_pickle_path,
    images_dir,
    output_dir,
    gt_mask_subdir="annotations",
    gt_suffix="_mask.png",
    sam_checkpoint='../modelWeights/sam_vit_l.pth',
    sam_model_type="vit_l",
    device="cuda",
    num_points=1,
    ignore_border_percentage=0.0,
    algorithm="default",
    use_box_input=False,
    use_mask_input=False,
    box_expansion_rate=0.0,
    mask_expansion_rate=0.0,
    iou_type="segm",
    max_dets=450
):
    """
    For each image in the inference pickle:
      - Loads the image and its GT mask.
      - Processes the predicted masks.
      - Runs SAM inference with the provided parameters:
            num_points, dropout_percentage (fixed at 0),
            ignore_border_percentage, algorithm,
            use_box_input, use_mask_input,
            box_expansion_rate, mask_expansion_rate.
      - Combines the resulting masks into a single 16-bit mask.
      - Evaluates the predictions via COCO metrics.
    Returns a summary dictionary including per-image and aggregate metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load inference data
    with open(inference_pickle_path, "rb") as pf:
        inference_data = pickle.load(pf)
    
    # Load SAM model
    if sam_checkpoint is None or not os.path.exists(sam_checkpoint):
        raise ValueError(f"SAM checkpoint path is invalid: {sam_checkpoint}")
    print(f"[INFO] Loading SAM model from {sam_checkpoint}")
    sam_predictor = load_sam_model(
        sam_checkpoint=sam_checkpoint,
        model_type=sam_model_type,
        device=device
    )
    
    per_image_results = []
    metrics_accumulators = {
        "SAM_AP@50": [],
        "SAM_AP@75": [],
        "SAM_AP@95": [],
        "SAM_AP@50:95": [],
        "SAM_AR@50": [],
        "SAM_AR@75": [],
        "SAM_AR@95": [],
        "SAM_AR@50:95": [],
    }
    missing_masks = []
    
    # Process each image in the inference data
    for image_name, image_data in inference_data.items():
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"[WARNING] Image file not found: {image_path}")
            continue
        
        # Construct ground truth mask path
        image_stem, ext = os.path.splitext(image_name)
        gt_mask_name = image_stem + gt_suffix
        gt_mask_path = os.path.join(images_dir, gt_mask_subdir, gt_mask_name)
        if not os.path.exists(gt_mask_path):
            print(f"[WARNING] GT mask not found for {image_name}: {gt_mask_path}")
            missing_masks.append(image_name)
            continue
        
        # Load image (convert to BGR for SAM inference)
        try:
            pil_image = Image.open(image_path)
            loop_image = np.array(pil_image)[:, :, ::-1].copy()
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            continue
        
        # Extract inference components (expecting keys: "boxes", "polygons", "masks")
        boxes = image_data.get("boxes", [])
        polygons = image_data.get("polygons", [])
        masks = image_data.get("masks", [])
        if not masks:
            print(f"[INFO] No masks found for {image_name}. Skipping.")
            continue
        
        # Process each mask: ensure binary format (uint8, thresholded if necessary)
        bin_masks = []
        for mask in masks:
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            if mask.dtype != bool:
                bin_mask = (mask > 127).astype(np.uint8)
            else:
                bin_mask = mask.astype(np.uint8)
            bin_masks.append(bin_mask)
        
        # Run SAM inference with the provided parameters
        try:
            sam_masks_list = run_sam_inference(
                predictor=sam_predictor,
                loop_image=loop_image,
                listOfPolygons=polygons,
                listOfBoxes=boxes,
                listOfMasks=bin_masks,
                image_width=pil_image.width,
                image_height=pil_image.height,
                num_points=num_points,
                dropout_percentage=0,
                ignore_border_percentage=ignore_border_percentage,
                algorithm=algorithm,
                use_box_input=use_box_input,
                use_mask_input=use_mask_input,
                box_expansion_rate=box_expansion_rate,
                mask_expansion_rate=mask_expansion_rate
            )
        except Exception as e:
            print(f"[ERROR] SAM inference failed for {image_name}: {e}")
            continue
        
        # Combine SAM masks into a single 16-bit mask
        try:
            sam_16bit = combine_masks_16bit(sam_masks_list, return_array=True)
        except Exception as e:
            print(f"[ERROR] Failed to combine SAM masks for {image_name}: {e}")
            continue
        
        # Save temporary SAM 16-bit mask image
        sam_mask_filename = f"sam_{image_stem}_{uuid.uuid4().hex}.png"
        sam_mask_path = os.path.join(output_dir, sam_mask_filename)
        Image.fromarray(sam_16bit).save(sam_mask_path)
        
        # Evaluate predictions using COCO metrics
        try:
            gt_data, pred_annotations = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, sam_mask_path, image_path
            )
            metrics = evaluate_coco_metrics(gt_data, pred_annotations, iou_type=iou_type, max_dets=max_dets)
        except Exception as e:
            print(f"[ERROR] Failed to evaluate metrics for {image_name}: {e}")
            metrics = {}
        
        # Build per-image result dictionary
        image_result = {
            "image_name": image_name,
            "SAM_AP@50": safe_get(metrics, "AP@50"),
            "SAM_AP@75": safe_get(metrics, "AP@75"),
            "SAM_AP@95": safe_get(metrics, "AP@95"),
            "SAM_AP@50:95": safe_get(metrics, "AP@50:95"),
            "SAM_AR@50": safe_get(metrics, "AR@50"),
            "SAM_AR@75": safe_get(metrics, "AR@75"),
            "SAM_AR@95": safe_get(metrics, "AR@95"),
            "SAM_AR@50:95": safe_get(metrics, "AR@50:95"),
        }
        print(f"\n[INFO] Metrics for {image_name}:")
        for k, v in image_result.items():
            if k != "image_name":
                print(f"   {k}: {v}")
        
        # Accumulate metrics for aggregate reporting
        for key in metrics_accumulators.keys():
            if image_result.get(key) is not None:
                metrics_accumulators[key].append(image_result.get(key))
        per_image_results.append(image_result)
        
        # Remove temporary SAM mask file
        if os.path.exists(sam_mask_path):
            os.remove(sam_mask_path)
    
    # Compute aggregate metrics over all images
    aggregate_metrics = {key: average_or_none(vals) for key, vals in metrics_accumulators.items()}
    summary = {
        "missing_masks": missing_masks,
        "aggregate_metrics": aggregate_metrics,
        "per_image_results": per_image_results
    }
    return summary

# ------------------ Argument Parsing ------------------ #
def parse_arguments():
    parser = argparse.ArgumentParser(description="SAM Ablation Inference and Evaluation")
    parser.add_argument('--inference_pickle_path', type=str, required=True,
                        help='Path to the pickle file containing model inferences.')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing the test images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to store outputs.')
    parser.add_argument('--gt_mask_subdir', type=str, default="annotations",
                        help='Subdirectory where GT masks are stored. Default: "annotations".')
    parser.add_argument('--gt_suffix', type=str, default="_mask.png",
                        help='Suffix for ground truth mask filenames. Default: "_mask.png".')
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                        help='Path to the SAM model checkpoint.')
    parser.add_argument('--sam_model_type', type=str, default="vit_l",
                        help='SAM model type. Default: "vit_l".')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to use for SAM inference. Default: "cuda".')
    parser.add_argument('--num_points', type=int, required=True,
                        help='Number of points for SAM inference.')
    parser.add_argument('--ignore_border_percentage', type=float, required=True,
                        help='Ignore border percentage for SAM inference.')
    parser.add_argument('--algorithm', type=str, required=True,
                        help='Algorithm to use for SAM inference.')
    parser.add_argument('--use_box_input', action='store_true',
                        help='Flag to use box input for SAM.')
    parser.add_argument('--use_mask_input', action='store_true',
                        help='Flag to use mask input for SAM.')
    parser.add_argument('--box_expansion_rate', type=float, required=True,
                        help='Box expansion rate for SAM inference.')
    parser.add_argument('--mask_expansion_rate', type=float, required=True,
                        help='Mask expansion rate for SAM inference.')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Identifier for the run (e.g., model name).')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to the output CSV file where metrics will be stored.')
    return parser.parse_args()

# ------------------ Main Execution ------------------ #
if __name__ == "__main__":
    args = parse_arguments()
    
    summary = sam_inference_and_evaluate(
        inference_pickle_path=args.inference_pickle_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        gt_mask_subdir=args.gt_mask_subdir,
        gt_suffix=args.gt_suffix,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        device=args.device,
        num_points=args.num_points,
        ignore_border_percentage=args.ignore_border_percentage,
        algorithm=args.algorithm,
        use_box_input=args.use_box_input,
        use_mask_input=args.use_mask_input,
        box_expansion_rate=args.box_expansion_rate,
        mask_expansion_rate=args.mask_expansion_rate,
        iou_type="segm",
        max_dets=450
    )
    
    aggregate_metrics = summary.get("aggregate_metrics", {})
    run_name = args.run_name
    write_metrics_to_csv(run_name, aggregate_metrics, args.output_csv)
    
    print("\n[SUMMARY] SAM Ablation Evaluation Completed.")
    if summary.get("missing_masks"):
        print("\n[WARNING] Missing GT masks for the following images:")
        for img in summary["missing_masks"]:
            print(f"   {img}")
    else:
        print("\n[INFO] All images have corresponding GT masks.")
    
    print("\n[SUMMARY] Average metrics across all evaluated images:")
    for k, v in aggregate_metrics.items():
        print(f"   {k}: {v}")
