#!/usr/bin/env python3
import os
import pickle
import cv2
import numpy as np
from PIL import Image
import json
import sys
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import argparse
import csv

sys.path.append('../')
sys.path.append('../../DualSight')  # Adjust as needed to locate your 'tools' and refine package
import segmentation_refinement as refine

def combine_masks_16bit(list_of_binary_masks, output_path=None, return_array=False):
    """
    Combine a list of 0/1 or 0/255 binary masks into a single 16-bit instance mask.
    Each mask is assigned a unique ID (1, 2, 3, ...).

    Parameters:
    - list_of_binary_masks (List[np.ndarray]): List of 0/1 or 0/255 binary masks (numpy arrays).
    - output_path (str, optional): Path to save the combined 16-bit PNG file.
    - return_array (bool, optional): If True, returns the combined 16-bit mask array.

    Returns:
    - combined_16bit (np.ndarray, optional): The combined 16-bit instance mask array if return_array is True.
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
        # Debug: Print unique values in the mask
        unique_vals = set(np.unique(mask))
        # print(f"Mask {idx-1} unique values: {unique_vals}")

        # Ensure mask is binary
        if mask.dtype != bool and not (unique_vals <= {0, 1} or unique_vals <= {0, 255}):
            raise ValueError(f"Mask at index {idx-1} has unexpected unique values {unique_vals}. Please provide 0/1 or 0/255 masks.")

        # Convert mask to boolean
        mask_bool = mask.astype(bool)

        # Assign unique ID to each mask
        combined_16bit[mask_bool] = idx

    # Save the combined mask as a 16-bit PNG if output_path is provided
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # OpenCV expects the image in the correct format
        # Ensure that combined_16bit is in uint16
        if combined_16bit.dtype != np.uint16:
            combined_16bit = combined_16bit.astype(np.uint16)

        success = cv2.imwrite(output_path, combined_16bit)
        if not success:
            raise IOError(f"Failed to write the combined mask to {output_path}")

    if return_array:
        return combined_16bit

    return None

def compute_specific_metrics(coco_eval, max_dets=200):
    # Adjust parameters as needed
    coco_eval.params.maxDets = [max_dets, max_dets, max_dets]
    # Set specific IoU thresholds
    coco_eval.params.iouThrs = [0.5, 0.75, 0.95]

    # Suppress output during evaluation
    saved_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
    finally:
        # Restore standard output
        sys.stdout.close()
        sys.stdout = saved_stdout

    # Extract precision and recall arrays after accumulation
    precision = coco_eval.eval['precision']  # shape: [T, R, K, A, M]
    recall = coco_eval.eval['recall']        # shape: [T, K, A, M]

    # Find indices for specific IoU thresholds
    iou_thr_indices = {
        0.5: np.where(np.isclose(coco_eval.params.iouThrs, 0.5))[0][0],
        0.75: np.where(np.isclose(coco_eval.params.iouThrs, 0.75))[0][0],
        0.95: np.where(np.isclose(coco_eval.params.iouThrs, 0.95))[0][0],
    }
    max_det_index = 0  # because we set maxDets list to have identical values

    # Compute AP for each IoU threshold over all areas
    ap_values = {}
    for thr, idx in iou_thr_indices.items():
        valid_precisions = precision[idx, :, :, :, max_det_index]
        valid_precisions = valid_precisions[valid_precisions != -1]  # Exclude invalid entries
        if valid_precisions.size > 0:
            ap = np.mean(valid_precisions)
        else:
            ap = float('nan')  # No valid precision values
        ap_values[f"AP@{int(thr*100)}"] = ap

    # Compute AP across IoU thresholds for all areas (AP@50:95)
    all_valid_precisions = precision[:, :, :, :, max_det_index]
    all_valid_precisions = all_valid_precisions[all_valid_precisions != -1]
    if all_valid_precisions.size > 0:
        ap_values["AP@50:95"] = np.mean(all_valid_precisions)
    else:
        ap_values["AP@50:95"] = float('nan')

    # Compute AR for each IoU threshold over all areas
    ar_values = {}
    for thr, idx in iou_thr_indices.items():
        valid_recalls = recall[idx, :, :, max_det_index]
        if valid_recalls.size > 0:
            ar = np.mean(valid_recalls)  # Recall values are valid; no need to filter
        else:
            ar = float('nan')
        ar_values[f"AR@{int(thr*100)}"] = ar

    # Compute AR across IoU thresholds for all areas (AR@50:95)
    all_valid_recalls = recall[:, :, :, max_det_index]
    if all_valid_recalls.size > 0:
        ar_values["AR@50:95"] = np.mean(all_valid_recalls)
    else:
        ar_values["AR@50:95"] = float('nan')

    # Combine AP and AR metrics into a single dictionary to return
    metrics = {}
    metrics.update(ap_values)
    metrics.update(ar_values)
    metrics["maxDets"] = max_dets  # include maxDets for reference if needed

    return metrics

def evaluate_coco_metrics(gt_data, pred_data, iou_type="segm", max_dets=200):
    """
    Evaluate COCO metrics (AP, AR) given ground truth data and predicted data
    in COCO format. Returns a dictionary of metrics.
    """
    # Create temporary JSON files for ground truth and predictions
    with open("temp_gt.json", "w") as gt_file:
        json.dump(gt_data, gt_file)

    pred_coco_format = []
    for pred in pred_data:
        # Ensure each prediction has a score; adjust for bbox if necessary
        pred_entry = {
            "image_id": pred["image_id"],
            "category_id": pred["category_id"],
            "score": pred.get("score", 1.0)  # default score 1.0 if missing
        }
        # Add segmentation or bbox depending on iou_type
        if iou_type == "segm":
            pred_entry["segmentation"] = pred["segmentation"]
        elif iou_type == "bbox":
            pred_entry["bbox"] = pred["bbox"]
        pred_coco_format.append(pred_entry)

    with open("temp_pred.json", "w") as pred_file:
        json.dump(pred_coco_format, pred_file)

    # Load ground truth and predictions into COCO API
    coco_gt = COCO("temp_gt.json")
    coco_pred = coco_gt.loadRes("temp_pred.json")
    coco_eval = COCOeval(coco_gt, coco_pred, iouType=iou_type)

    # Compute and retrieve metrics
    metrics = compute_specific_metrics(coco_eval, max_dets=max_dets)

    # Optionally delete temporary files after evaluation
    os.remove("temp_gt.json")
    os.remove("temp_pred.json")

    return metrics

def mask_to_coco_format(binary_mask, image_id, category_id=1, annotation_id=1):
    """
    Convert a binary mask (NumPy array) to COCO RLE format.
    """
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    # pycocotools uses bytes for 'counts', so we decode for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": float(maskUtils.area(rle)),
        "bbox": maskUtils.toBbox(rle).tolist(),
        "iscrowd": 0
    }

def generate_coco_annotations_from_multi_instance_masks_16bit(
    gt_mask_path,
    pred_mask_path,
    image_path,
    image_id=1,
    category_id=1
):
    """
    Generate COCO-style annotations for both ground truth and predicted masks.
    """
    # Read masks preserving the original bit depth (e.g., 16-bit)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)
    print(gt_mask_path)
    print('-'*100)
    if gt_mask is None or pred_mask is None:
        raise FileNotFoundError("One of the mask files was not found.")

    # Identify unique object IDs (works correctly for >255 values)
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
            mask_to_coco_format(binary_mask, image_id, category_id, annotation_id)
        )

    # Create prediction annotations
    for annotation_id, obj_id in enumerate(pred_ids, start=1):
        binary_mask = (pred_mask == obj_id).astype(np.uint8)
        pred_annotations.append(
            mask_to_coco_format(binary_mask, image_id, category_id, annotation_id)
        )

    # COCO GT data structure
    gt_data = {
        "images": [{
            "id": image_id,
            "width": int(gt_mask.shape[1]),
            "height": int(gt_mask.shape[0]),
            "file_name": os.path.basename(image_path)
        }],
        "annotations": gt_annotations,
        "categories": [{"id": category_id, "name": "object"}]
    }

    return gt_data, pred_annotations


# ------------------ Utility functions ------------------ #
def safe_get(metrics_dict, key):
    """Return metrics_dict[key] if present, else None."""
    if metrics_dict is not None and key in metrics_dict:
        return metrics_dict[key]
    return None

def average_or_none(values):
    """Return mean of a list if not empty, else None."""
    return float(np.mean(values)) if values else None

def write_metrics_to_csv(run_name, metrics, output_csv_path):
    """
    Writes the metrics dictionary to a CSV file with the given run_name.

    Args:
        run_name (str): Identifier for the run (e.g., model name).
        metrics (dict): Dictionary of metrics to write.
        output_csv_path (str): Path to the CSV file.
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

def write_instance_iou_to_csv(records, output_csv_path):
    """
    Writes instance-level IoU records to a CSV file.
    Each record includes: image_name, inference_file, object_id, initial_iou, refined_iou.
    """
    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csvfile:
        fieldnames = ['image_name', 'inference_file', 'object_id', 'initial_iou', 'refined_iou']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record)


# ------------------ Main refinement + evaluation logic ------------------ #
def refine_and_evaluate(
    inference_pickle_path,
    images_dir,
    output_dir,
    gt_mask_subdir="annotations",
    gt_suffix="_mask.png",
    binarize_threshold=128  # Threshold for binarization
):
    """
    1) Loads precomputed predictions from a pickle file.
    2) For each image:
       - Reads the image and predicted instance masks.
       - Computes a binarized version of the initial predictions and refines them using segmentation_refinement.
       - For each ground-truth instance (from the GT mask), computes the IoU with both the initial and refined predictions.
       - Evaluates them via COCO metrics against the ground truth mask.
    3) Returns image-level and aggregate metrics as well as per-instance IoU records.

    Args:
        inference_pickle_path (str): Path to the pickle file containing predictions.
                                     Expects a dict of { image_name: {"masks": [...], ... }, ... }.
        images_dir (str): Directory containing the original test images.
        output_dir (str): Directory to optionally store temporary or final outputs.
        gt_mask_subdir (str): Subdirectory under images_dir where GT masks are stored. Default: "annotations".
        gt_suffix (str): Suffix appended to the base image name to locate its ground-truth mask.
        binarize_threshold (int): Threshold value for binarizing refined masks. Default: 128.

    Returns:
        dict: A dictionary containing aggregate metrics, per-image results, and instance-level IoU records.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load inference data
    with open(inference_pickle_path, "rb") as pf:
        inference_data = pickle.load(pf)

    # Initialize the segmentation refiner (adjust device as needed)
    refiner = refine.Refiner(device='cuda:0')  # or 'cpu' if no GPU available

    # Metric accumulators (added additional keys for box metrics)
    metrics_accumulators = {
        "refined_mask_AP@50": [],
        "refined_mask_AP@75": [],
        "refined_mask_AP@95": [],
        "refined_mask_AP@50:95": [],
        "refined_mask_AR@50": [],
        "refined_mask_AR@75": [],
        "refined_mask_AR@95": [],
        "refined_mask_AR@50:95": [],
        "refined_box_AP@50": [],
        "refined_box_AP@75": [],
        "refined_box_AP@95": [],
        "refined_box_AP@50:95": [],
        "refined_box_AR@50": [],
        "refined_box_AR@75": [],
        "refined_box_AR@95": [],
        "refined_box_AR@50:95": [],
    }

    # Prepare directory where GT masks are located
    gt_mask_dir = os.path.join(images_dir, gt_mask_subdir)

    # Track missing masks
    missing_masks = []

    # List to store per-image results
    per_image_results = []

    # List to store instance-level IoU records
    instance_iou_records = []

    # Process each image in the inference data
    for image_name, preds in inference_data.items():
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            print(f"[WARNING] Image file not found: {image_path}")
            continue

        # Build the path to the ground-truth mask in 'annotations' subfolder
        image_stem, ext = os.path.splitext(image_name)
        gt_mask_name = image_stem + gt_suffix
        gt_mask_path = os.path.join(gt_mask_dir, gt_mask_name)

        if not os.path.exists(gt_mask_path):
            print(f"[WARNING] GT mask '{gt_mask_path}' not found. Skipping image '{image_name}'.")
            missing_masks.append(image_name)
            continue

        # Read input image
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            print(f"[ERROR] Could not read image {image_path}. Skipping.")
            continue

        # Read the ground truth mask (preserving bit depth)
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
        if gt_mask is None:
            print(f"[ERROR] Could not read GT mask {gt_mask_path}. Skipping.")
            continue

        # Identify ground truth instance IDs
        gt_ids = np.unique(gt_mask)
        gt_ids = gt_ids[gt_ids > 0]

        # Predicted masks (list of arrays). Typically each entry is a 2D array (H x W).
        predicted_masks = preds.get("masks", [])
        if not predicted_masks:
            print(f"[INFO] No predicted masks found for {image_name}. Skipping.")
            continue

        # Process each predicted mask to compute both the initial (binarized) and refined versions.
        initial_masks = []
        refined_masks = []
        for pmask in predicted_masks:
            # If the pickle data has the mask in torch tensor or boolean format, convert it
            if hasattr(pmask, "cpu"):  # e.g., torch tensor
                pmask = pmask.cpu().numpy()

            # Ensure mask is in uint8
            if pmask.dtype != np.uint8:
                pmask = pmask.astype(np.uint8)

            # Scale if mask is in {0,1}
            if pmask.max() == 1:
                pmask = pmask * 255

            # Compute initial binary mask using Otsu's thresholding
            _, init_mask_binary = cv2.threshold(pmask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            initial_masks.append(init_mask_binary)

            # Use the refiner to get the refined mask
            refined_mask = refiner.refine(
                image=bgr_image,
                mask=pmask,
                fast=False,   # or True, depending on your speed/quality trade-off
                L=900         # adjust as needed
            )
            # Binarize the refined mask using Otsu's thresholding
            _, refined_mask_binary = cv2.threshold(refined_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            refined_masks.append(refined_mask_binary)

        # ---------------- Compute instance-level IoU ---------------- #
        # For each ground truth instance, compute the best (maximum) IoU across all predicted masks.
        for gt_id in gt_ids:
            gt_instance = (gt_mask == gt_id)
            best_initial_iou = 0.0
            best_refined_iou = 0.0
            for init_mask in initial_masks:
                intersection = np.logical_and(gt_instance, init_mask > 0).sum()
                union = np.logical_or(gt_instance, init_mask > 0).sum()
                iou = intersection / union if union > 0 else 0
                best_initial_iou = max(best_initial_iou, iou)
            for ref_mask in refined_masks:
                intersection = np.logical_and(gt_instance, ref_mask > 0).sum()
                union = np.logical_or(gt_instance, ref_mask > 0).sum()
                iou = intersection / union if union > 0 else 0
                best_refined_iou = max(best_refined_iou, iou)
            instance_iou_records.append({
                "image_name": image_name,
                "inference_file": os.path.basename(inference_pickle_path),
                "object_id": int(gt_id),
                "initial_iou": best_initial_iou,
                "refined_iou": best_refined_iou
            })

        # Combine refined masks into a single 16-bit file for COCO-style evaluation
        try:
            refined_16bit = combine_masks_16bit(refined_masks, return_array=True)
        except ValueError as ve:
            print(f"[ERROR] Combining masks for {image_name} failed: {ve}")
            continue

        # Save the refined 16-bit mask temporarily
        refined_16bit_path = os.path.join(
            output_dir,
            f"refined_{image_stem}.png"
        )
        Image.fromarray(refined_16bit).save(refined_16bit_path)

        # Evaluate with COCO metrics (both segmentation and bbox)
        try:
            gt_data, pred_data_coco = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, refined_16bit_path, image_path
            )
            segm_metrics = evaluate_coco_metrics(
                gt_data, pred_data_coco,
                iou_type="segm",
                max_dets=450
            )
            bbox_metrics = evaluate_coco_metrics(
                gt_data, pred_data_coco,
                iou_type="bbox",
                max_dets=450
            )
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {image_name}: {e}")
            segm_metrics = {}
            bbox_metrics = {}

        # Collect metrics
        image_result = {
            "image_name": image_name,
            "refined_mask_AP@50":  safe_get(segm_metrics, "AP@50"),
            "refined_mask_AP@75":  safe_get(segm_metrics, "AP@75"),
            "refined_mask_AP@95":  safe_get(segm_metrics, "AP@95"),
            "refined_mask_AP@50:95": safe_get(segm_metrics, "AP@50:95"),
            "refined_mask_AR@50":  safe_get(segm_metrics, "AR@50"),
            "refined_mask_AR@75":  safe_get(segm_metrics, "AR@75"),
            "refined_mask_AR@95":  safe_get(segm_metrics, "AR@95"),
            "refined_mask_AR@50:95": safe_get(segm_metrics, "AR@50:95"),
            "refined_box_AP@50": safe_get(bbox_metrics, "AP@50"),
            "refined_box_AP@75": safe_get(bbox_metrics, "AP@75"),
            "refined_box_AP@95": safe_get(bbox_metrics, "AP@95"),
            "refined_box_AP@50:95": safe_get(bbox_metrics, "AP@50:95"),
            "refined_box_AR@50": safe_get(bbox_metrics, "AR@50"),
            "refined_box_AR@75": safe_get(bbox_metrics, "AR@75"),
            "refined_box_AR@95": safe_get(bbox_metrics, "AR@95"),
            "refined_box_AR@50:95": safe_get(bbox_metrics, "AR@50:95"),
        }

        # Print per-image results (optional)
        print(f"\n[INFO] Results for '{image_name}':")
        for k, v in image_result.items():
            if k != "image_name":
                print(f"   {k}: {v}")

        # Accumulate in metrics_accumulators
        for mkey in metrics_accumulators.keys():
            val = image_result.get(mkey, None)
            if val is not None:
                metrics_accumulators[mkey].append(val)

        per_image_results.append(image_result)

    # Compute aggregate metrics
    aggregate_metrics = {}
    for mkey, vals in metrics_accumulators.items():
        avg_val = average_or_none(vals)
        aggregate_metrics[mkey] = avg_val

    summary = {
        "missing_masks": missing_masks,
        "aggregate_metrics": aggregate_metrics,
        "per_image_results": per_image_results,
        "instance_iou_records": instance_iou_records
    }

    return summary

# ------------------ CSV Writing Function ------------------ #
def write_metrics_to_csv(run_name, metrics, output_csv_path):
    """
    Writes the metrics dictionary to a CSV file with the given run_name.

    Args:
        run_name (str): Identifier for the run (e.g., model name).
        metrics (dict): Dictionary of metrics to write.
        output_csv_path (str): Path to the CSV file.
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

# ------------------ Argument Parsing ------------------ #
def parse_arguments():
    parser = argparse.ArgumentParser(description="Refine masks and evaluate using COCO metrics.")
    parser.add_argument(
        '--inference_pickle_path', type=str, required=True,
        help='Path to the pickle file containing predictions.'
    )
    parser.add_argument(
        '--images_dir', type=str, required=True,
        help='Directory containing the original test images.'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to store temporary or final outputs.'
    )
    parser.add_argument(
        '--gt_mask_subdir', type=str, default="annotations",
        help='Subdirectory under images_dir where GT masks are stored. Default: "annotations".'
    )
    parser.add_argument(
        '--gt_suffix', type=str, default="_mask.png",
        help='Suffix appended to the base image name to locate its ground-truth mask. Default: "_mask.png".'
    )
    parser.add_argument(
        '--binarize_threshold', type=int, default=128,
        help='Threshold value for binarizing refined masks. Default: 128.'
    )
    parser.add_argument(
        '--run_name', type=str, required=True,
        help='Identifier for the run (e.g., model name). This will be used in the CSV output.'
    )
    parser.add_argument(
        '--output_csv', type=str, required=True,
        help='Path to the output CSV file where aggregate metrics will be stored.'
    )
    parser.add_argument(
        '--instance_csv', type=str, required=True,
        help='Path to the CSV file where instance-level IoU metrics will be stored.'
    )
    return parser.parse_args()

# ------------------ Main Execution ------------------ #
if __name__ == "__main__":
    args = parse_arguments()

    # Perform refinement and evaluation
    summary = refine_and_evaluate(
        inference_pickle_path=args.inference_pickle_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        gt_mask_subdir=args.gt_mask_subdir,
        gt_suffix=args.gt_suffix,
        binarize_threshold=args.binarize_threshold
    )

    # Prepare metrics for CSV (aggregate metrics)
    aggregate_metrics = summary.get("aggregate_metrics", {})
    run_name = args.run_name

    # Write aggregate metrics to CSV
    write_metrics_to_csv(run_name, aggregate_metrics, args.output_csv)

    # Write instance-level IoU metrics to CSV
    write_instance_iou_to_csv(summary.get("instance_iou_records", []), args.instance_csv)
    print(f"\n[INFO] Instance-level IoU metrics saved to {args.instance_csv}.")

    # Optionally, print summary
    print("\n[SUMMARY] Evaluation Completed.")
    if summary.get("missing_masks"):
        print("\n[WARNING] The following images are missing ground truth masks:")
        for img in summary["missing_masks"]:
            print(f"   {img}")
    else:
        print("\n[INFO] All images have corresponding ground truth masks.")

    print("\n[SUMMARY] Average metrics across all evaluated images:")
    for mkey, val in aggregate_metrics.items():
        print(f"   {mkey}: {val}")

'''
Example usage:
python cascadePSP.py \
        --inference_pickle_path "../DualSight/ablationStudy/inference_outputs/yolov8n_inference.pkl" \
        --images_dir "../datasets/powder/test" \
        --output_dir "cascade_refined_outputs" \
        --run_name "yolov8n" \
        --output_csv "cascadeMetrics.csv" \
        --instance_csv "instanceIoU.csv"

The accompanying shell script (run_cascadePSP.sh) would similarly need to pass the --instance_csv argument.
'''
