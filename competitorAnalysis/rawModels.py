#!/usr/bin/env python3
import os
import pickle
import cv2
import numpy as np
import sys
import json
import argparse
import csv
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

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

def evaluate_models(inference_pickle_path, images_dir, output_csv, run_name):
    """
    Load predictions and evaluate COCO metrics.
    """
    with open(inference_pickle_path, "rb") as pf:
        inference_data = pickle.load(pf)

    gt_mask_dir = os.path.join(images_dir, "annotations")
    metrics_accumulators = []

    for image_id, (image_name, preds) in enumerate(inference_data.items(), start=1):
        image_stem, _ = os.path.splitext(image_name)
        gt_mask_path = os.path.join(gt_mask_dir, f"{image_stem}_mask.png")

        if not os.path.exists(gt_mask_path):
            print(f"[WARNING] GT mask not found for {image_name}. Skipping.")
            continue

        pred_masks = preds.get("masks", [])
        if not pred_masks:
            print(f"[INFO] No predicted masks found for {image_name}. Skipping.")
            continue

        # Combine predicted masks into a single 16-bit mask
        try:
            combined_16bit = combine_masks_16bit(pred_masks, return_array=True)
            if combined_16bit is None:
                print(f"[WARNING] No combined mask for {image_name}; skipping.")
                continue
        except ValueError as ve:
            print(f"[ERROR] Combining masks for {image_name} failed: {ve}")
            continue

        # Save the combined mask temporarily
        combined_mask_path = os.path.join(images_dir, "temp_pred_mask.png")
        cv2.imwrite(combined_mask_path, combined_16bit)

        # Generate COCO annotations
        try:
            gt_data, pred_data_coco = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, combined_mask_path, image_id=image_id, category_id=1
            )
        except Exception as e:
            print(f"[ERROR] Failed to generate COCO annotations for {image_name}: {e}")
            os.remove(combined_mask_path)
            continue

        # Evaluate COCO metrics
        try:
            segm_metrics = evaluate_coco_metrics(
                gt_data, pred_data_coco, iou_type="segm", max_dets=200
            )
        except Exception as e:
            print(f"[ERROR] Failed to evaluate COCO metrics for {image_name}: {e}")
            os.remove(combined_mask_path)
            continue

        # Remove the temporary combined mask
        os.remove(combined_mask_path)

        # Collect metrics
        metrics_accumulators.append(segm_metrics)

        print(f"[INFO] Evaluated {image_name}: AP@50={segm_metrics.get('AP@50', 'N/A')}, AP@75={segm_metrics.get('AP@75', 'N/A')}, AP@95={segm_metrics.get('AP@95', 'N/A')}, AP@50:95={segm_metrics.get('AP@50:95', 'N/A')}")

    if not metrics_accumulators:
        print("[WARNING] No metrics to aggregate. Exiting.")
        return

    # Aggregate metrics
    aggregate_metrics = {}
    keys = metrics_accumulators[0].keys()
    for key in keys:
        values = [m[key] for m in metrics_accumulators if m[key] is not None]
        if values:
            aggregate_metrics[key] = float(np.mean(values))
        else:
            aggregate_metrics[key] = float('nan')

    # Write metrics to CSV
    write_metrics_to_csv(run_name, aggregate_metrics, output_csv)

def write_metrics_to_csv(run_name, metrics, output_csv_path):
    """
    Write the metrics to a CSV file.
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate raw model predictions using COCO metrics.")
    parser.add_argument('--inference_pickle_path', type=str, required=True, help='Path to inference pickle file.')
    parser.add_argument('--images_dir', type=str, required=True, help='Path to images directory.')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file path.')
    parser.add_argument('--run_name', type=str, required=True, help='Name for this run.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_models(args.inference_pickle_path, args.images_dir, args.output_csv, args.run_name)
