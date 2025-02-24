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
  - Writes instance-level (per-object) IoU metrics to a separate CSV file.

Usage example (from a bash script):
    python sam.py \
        --inference_pickle_path "../DualSight/ablationStudy/inference_outputs/yolov8n_inference.pkl" \
        --images_dir "../datasets/powder/test" \
        --output_dir "sam_ablation_outputs" \
        --sam_checkpoint "/path/to/sam_checkpoint.pth" \
        --num_points 16 \
        --ignore_border_percentage 0.1 \
        --algorithm "Distance Max" \
        --use_box_input True \
        --use_mask_input True \
        --box_expansion_rate 0.05 \
        --mask_expansion_rate 0.1 \
        --run_name "YOLOv8 Nano" \
        --output_csv "samAblationMetrics.csv" \
        --instance_csv "samAblationInstance.csv"
"""

import os
import sys
import pickle
import csv
import json
import numpy as np
import time
import random
import uuid
from PIL import Image
import argparse
import cv2

from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# Adjust these paths as needed so Python can locate your helper modules.
sys.path.append('../DualSight')  # Adjust as needed
from sam_helper import load_sam_model

# For reproducibility
random.seed(42)
np.random.seed(42)
import torch
if torch.cuda.is_available():
    torch.manual_seed(42)

# =============================================================================
#  Utility functions
# =============================================================================

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def combine_masks_16bit(list_of_binary_masks, output_path=None, return_array=False):
    if not list_of_binary_masks:
        print("[WARNING] No masks provided to combine.")
        return None
    first_shape = list_of_binary_masks[0].shape
    for idx, mask in enumerate(list_of_binary_masks):
        if mask.shape != first_shape:
            raise ValueError(f"All masks must have the same shape. Mask at index {idx} has shape {mask.shape}, expected {first_shape}.")
    height, width = first_shape
    combined_16bit = np.zeros((height, width), dtype=np.uint16)
    for idx, mask in enumerate(list_of_binary_masks, start=1):
        unique_vals = set(np.unique(mask))
        if mask.dtype != bool and not (unique_vals <= {0, 1} or unique_vals <= {0, 255}):
            raise ValueError(f"Mask at index {idx-1} has unexpected unique values {unique_vals}. Please provide 0/1 or 0/255 masks.")
        mask_bool = mask.astype(bool)
        combined_16bit[mask_bool] = idx
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, combined_16bit.astype(np.uint16)):
            raise IOError(f"Failed to write the combined mask to {output_path}")
    if return_array:
        return combined_16bit
    return None

def mask_to_coco_format(binary_mask, image_id, category_id=1, annotation_id=1, score=1.0):
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": float(maskUtils.area(rle)),
        "bbox": maskUtils.toBbox(rle).tolist(),
        "iscrowd": 0,
        "score": score
    }

def generate_coco_annotations_from_multi_instance_masks_16bit(gt_mask_path, pred_mask_path, image_path, image_id=1, category_id=1):
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)
    print(gt_mask_path)
    print('-'*100)
    if gt_mask is None or pred_mask is None:
        raise FileNotFoundError("One of the mask files was not found.")
    gt_ids = np.unique(gt_mask)
    pred_ids = np.unique(pred_mask)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = pred_ids[pred_ids > 0]
    gt_annotations = []
    pred_annotations = []
    for annotation_id, obj_id in enumerate(gt_ids, start=1):
        binary_mask = (gt_mask == obj_id).astype(np.uint8)
        gt_annotations.append(mask_to_coco_format(binary_mask, image_id, category_id, annotation_id))
    for annotation_id, obj_id in enumerate(pred_ids, start=1):
        binary_mask = (pred_mask == obj_id).astype(np.uint8)
        pred_annotations.append(mask_to_coco_format(binary_mask, image_id, category_id, annotation_id))
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

def compute_specific_metrics(coco_eval, max_dets=200):
    coco_eval.params.maxDets = [max_dets, max_dets, max_dets]
    coco_eval.params.iouThrs = [0.5, 0.75, 0.95]
    saved_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        coco_eval.evaluate()
        coco_eval.accumulate()
    finally:
        sys.stdout.close()
        sys.stdout = saved_stdout
    precision = coco_eval.eval['precision']
    recall = coco_eval.eval['recall']
    iou_thr_indices = {
        0.5: np.where(np.isclose(coco_eval.params.iouThrs, 0.5))[0][0],
        0.75: np.where(np.isclose(coco_eval.params.iouThrs, 0.75))[0][0],
        0.95: np.where(np.isclose(coco_eval.params.iouThrs, 0.95))[0][0],
    }
    max_det_index = 0
    ap_values = {}
    for thr, idx in iou_thr_indices.items():
        valid_precisions = precision[idx, :, :, :, max_det_index]
        valid_precisions = valid_precisions[valid_precisions != -1]
        ap_values[f"AP@{int(thr*100)}"] = np.mean(valid_precisions) if valid_precisions.size > 0 else float('nan')
    all_valid = precision[:, :, :, :, max_det_index]
    all_valid = all_valid[all_valid != -1]
    ap_values["AP@50:95"] = np.mean(all_valid) if all_valid.size > 0 else float('nan')
    ar_values = {}
    for thr, idx in iou_thr_indices.items():
        valid_recalls = recall[idx, :, :, max_det_index]
        ar_values[f"AR@{int(thr*100)}"] = np.mean(valid_recalls) if valid_recalls.size > 0 else float('nan')
    all_valid_recall = recall[:, :, :, max_det_index]
    ar_values["AR@50:95"] = np.mean(all_valid_recall) if all_valid_recall.size > 0 else float('nan')
    metrics = {}
    metrics.update(ap_values)
    metrics.update(ar_values)
    metrics["maxDets"] = max_dets
    return metrics

def evaluate_coco_metrics(gt_data, pred_data, iou_type="segm", max_dets=450):
    with open("temp_gt.json", "w") as f:
        json.dump(gt_data, f)
    with open("temp_pred.json", "w") as f:
        json.dump(pred_data, f)
    coco_gt = COCO("temp_gt.json")
    coco_pred = coco_gt.loadRes("temp_pred.json")
    coco_eval = COCOeval(coco_gt, coco_pred, iouType=iou_type)
    metrics = compute_specific_metrics(coco_eval, max_dets=max_dets)
    os.remove("temp_gt.json")
    os.remove("temp_pred.json")
    return metrics

def safe_get(metrics_dict, key):
    return metrics_dict.get(key, None)

def average_or_none(values):
    return float(np.mean(values)) if values else None

def write_metrics_to_csv(run_name, metrics, output_csv_path):
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
    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csvfile:
        fieldnames = ['image_name', 'inference_file', 'object_id', 'initial_iou', 'refined_iou']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record)

# =============================================================================
#  2) Point Selection and Mask Creation Helpers
# =============================================================================
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion
from itertools import combinations

def naiveMaximization(white_cells, num_points):
    if num_points == 1:
        centroid_x = np.mean([c[0] for c in white_cells])
        centroid_y = np.mean([c[1] for c in white_cells])
        closest = min(white_cells, key=lambda c: (c[0]-centroid_x)**2 + (c[1]-centroid_y)**2)
        return [closest]
    max_distance = 0
    best_set = None
    for point_set in combinations(white_cells, num_points):
        total = sum(np.sqrt((point_set[i][0]-point_set[j][0])**2 + (point_set[i][1]-point_set[j][1])**2)
                    for i in range(num_points) for j in range(i+1, num_points))
        if total > max_distance:
            max_distance = total
            best_set = point_set
    return list(best_set) if best_set is not None else []

def simulatedAnnealingMaximization(white_cells, num_points, initial_temp=1000, cooling_rate=0.995, max_iterations=1000, patience=300):
    def total_distance(points):
        return sum(np.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
                   for i in range(num_points) for j in range(i+1, num_points))
    current = random.sample(white_cells, num_points)
    best = current[:]
    best_distance = total_distance(best)
    temperature = initial_temp
    no_improve = 0
    for _ in range(max_iterations):
        candidate = current[:]
        idx = random.randint(0, num_points - 1)
        candidate[idx] = random.choice(white_cells)
        cand_distance = total_distance(candidate)
        if cand_distance > total_distance(current) or np.exp((cand_distance - total_distance(current))/temperature) > random.random():
            current = candidate
            if cand_distance > best_distance:
                best = candidate[:]
                best_distance = cand_distance
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        if no_improve >= patience:
            break
        temperature *= cooling_rate
        if temperature < 1e-10:
            break
    return best

def hillClimbingMaximization(white_cells, num_points, max_iterations=1000):
    def total_distance(points):
        return sum(np.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
                   for i in range(num_points) for j in range(i+1, num_points))
    current = random.sample(white_cells, num_points)
    best = current[:]
    best_distance = total_distance(current)
    for _ in range(max_iterations):
        improved = False
        for i in range(num_points):
            for candidate in white_cells:
                if candidate != current[i]:
                    test_points = current[:]
                    test_points[i] = candidate
                    if total_distance(test_points) > total_distance(current):
                        current = test_points
                        improved = True
                        break
            if improved:
                break
        if total_distance(current) > best_distance:
            best = current[:]
            best_distance = total_distance(current)
        if not improved:
            break
    return best

def clusterInitialization(white_cells, num_points):
    selected = [random.choice(white_cells)]
    while len(selected) < num_points:
        next_pt = max(white_cells, key=lambda pt: min(np.sqrt((pt[0]-s[0])**2 + (pt[1]-s[1])**2) for s in selected))
        selected.append(next_pt)
    return selected

def randomSelection(white_cells, num_points):
    return white_cells if len(white_cells) <= num_points else random.sample(white_cells, num_points)

def voronoi_optimization_from_coords(coords, num_points, iterations=50):
    coords = np.array(coords)
    init_idx = np.random.choice(len(coords), num_points, replace=False)
    points = coords[init_idx]
    def partition(coords, points):
        dists = np.linalg.norm(coords[:, None] - points[None, :], axis=2)
        return np.argmin(dists, axis=1)
    for _ in range(iterations):
        assign = partition(coords, points)
        new_points = []
        for i in range(num_points):
            region = coords[assign == i]
            new_points.append(region.mean(axis=0) if len(region) > 0 else coords[np.random.choice(len(coords))])
        new_points = np.array(new_points)
        if np.allclose(points, new_points, atol=1e-2):
            break
        points = new_points
    return points.tolist()

def select_point_placement(mask, num_points, dropout_percentage=0, ignore_border_percentage=0, algorithm="Voronoi", select_perimeter=False):
    rows, cols = np.where(mask > 0)
    white_cells = list(zip(rows, cols))
    labeled = label(mask)
    regions = regionprops(labeled)
    if len(regions) > 1:
        largest = max(regions, key=lambda r: r.area)
        white_cells = [(r, c) for (r, c) in white_cells if labeled[r, c] == largest.label]
    if select_perimeter:
        eroded = binary_erosion(mask)
        perimeter = mask & ~eroded
        rows, cols = np.where(perimeter > 0)
        white_cells = list(zip(rows, cols))
    if ignore_border_percentage > 0 and white_cells:
        min_r = min(r for r, _ in white_cells)
        max_r = max(r for r, _ in white_cells)
        min_c = min(c for _, c in white_cells)
        max_c = max(c for _, c in white_cells)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        ignore_h = int(height * (ignore_border_percentage / 100.0))
        ignore_w = int(width * (ignore_border_percentage / 100.0))
        white_cells = [(r, c) for (r, c) in white_cells if (min_r+ignore_h) <= r <= (max_r-ignore_h) and (min_c+ignore_w) <= c <= (max_c-ignore_w)]
    if white_cells and dropout_percentage > 0:
        keep = int(len(white_cells) * (1 - dropout_percentage/100))
        if keep < len(white_cells):
            white_cells = random.sample(white_cells, keep)
    white_norm = [(r/float(mask.shape[0]), c/float(mask.shape[1])) for r, c in white_cells]
    t0 = time.time()
    algo_map = {
        "Naive": naiveMaximization,
        "Simulated Annealing": simulatedAnnealingMaximization,
        "Hill Climbing": hillClimbingMaximization,
        "Cluster Initialization": clusterInitialization,
        "Random": randomSelection,
        "Voronoi": voronoi_optimization_from_coords,
    }
    selected = algo_map[algorithm](white_norm, num_points)
    t1 = time.time()
    selected_pixels = [(max(0, min(int(p[0]*mask.shape[0]), mask.shape[0]-1)),
                        max(0, min(int(p[1]*mask.shape[1]), mask.shape[1]-1)))
                       for p in selected]
    return selected_pixels, 0, (t1 - t0)

# =============================================================================
#  3) SAM Inference Helpers
# =============================================================================

def expand_bbox_within_border(x1, y1, x2, y2, width, height, expansion_rate=0.0):
    if expansion_rate <= 0:
        return [x1, y1, x2, y2]
    orig_w = x2 - x1
    orig_h = y2 - y1
    new_w = orig_w * expansion_rate
    new_h = orig_h * expansion_rate
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    new_x1 = max(0, cx - new_w/2)
    new_y1 = max(0, cy - new_h/2)
    new_x2 = min(width, cx + new_w/2)
    new_y2 = min(height, cy + new_h/2)
    return [new_x1, new_y1, new_x2, new_y2]

def adjust_mask_area(mask, target_percentage, max_iterations=50, kernel_size=(3,3)):
    if target_percentage == 0 or target_percentage == 100:
        return mask
    binary_mask = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    orig_area = np.sum(binary_mask)
    target_area = orig_area * (target_percentage / 100.0)
    op = 'erode' if target_percentage < 100 else 'dilate'
    new_mask = binary_mask.copy()
    for _ in range(max_iterations):
        curr_area = np.sum(new_mask)
        if op == 'erode' and curr_area <= target_area:
            break
        if op == 'dilate' and curr_area >= target_area:
            break
        new_mask = cv2.erode(new_mask, kernel, iterations=1) if op == 'erode' else cv2.dilate(new_mask, kernel, iterations=1)
    return new_mask

def prepare_mask_for_sam(mask, target_size=(256,256)):
    if len(mask.shape) != 2:
        raise ValueError("Mask must be 2D (H,W)")
    mask_float = mask.astype(np.float32)
    if mask_float.max() > 1:
        mask_float /= 255.0
    import torch.nn.functional as F
    mask_tensor = torch.tensor(mask_float, dtype=torch.float32)[None, None, :, :]
    mask_resized = F.interpolate(mask_tensor, size=target_size, mode="bilinear", align_corners=False).squeeze(0)
    return mask_resized

def run_sam_inference(sam_predictor, loop_image, listOfPolygons, listOfBoxes, listOfMasks,
                      image_width, image_height, num_points=4, dropout_percentage=0,
                      ignore_border_percentage=5, algorithm="Voronoi",
                      use_box_input=True, use_mask_input=False, box_expansion_rate=0.0,
                      mask_expansion_rate=0.0):
    sam_masks_list = []
    if algorithm == "Distance Max":
        algorithm = "Hill Climbing"
    for idx in range(len(listOfPolygons)):
        box = listOfBoxes[idx]
        if hasattr(box, 'cpu'):
            box = box.cpu().numpy()
        box = np.array(box, dtype=float)
        if use_box_input:
            x1, y1, x2, y2 = expand_bbox_within_border(box[0], box[1], box[2], box[3],
                                                        image_width, image_height, expansion_rate=box_expansion_rate)
            new_box = [x1, y1, x2, y2]
        else:
            new_box = None
        mask = listOfMasks[idx]
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        if np.sum(mask) == 0:
            print(f"[INFO] Skipping mask at index {idx} because its area is 0.")
            continue
        try:
            if algorithm in ["Hill Climbing"] and num_points != 1:
                selected_points, _, _ = select_point_placement(mask, num_points,
                                                                 dropout_percentage=dropout_percentage,
                                                                 ignore_border_percentage=ignore_border_percentage,
                                                                 algorithm=algorithm,
                                                                 select_perimeter=True)
            else:
                selected_points, _, _ = select_point_placement(mask, num_points,
                                                                 dropout_percentage=dropout_percentage,
                                                                 ignore_border_percentage=ignore_border_percentage,
                                                                 algorithm=algorithm,
                                                                 select_perimeter=False)
        except Exception as e:
            print(f"[ERROR] selecting points: {e} (mask sum: {np.sum(mask)})")
            continue
        mask = adjust_mask_area(mask, mask_expansion_rate * 100)
        sam_predictor.set_image(loop_image)
        if not selected_points:
            print(f"[INFO] No prompt points found for mask at index {idx}. Skipping.")
            continue
        py, px = zip(*selected_points)
        input_points = np.array(list(zip(px, py)))
        input_labels = np.ones(len(input_points), dtype=int)
        predict_kwargs = {
            'point_coords': input_points,
            'point_labels': input_labels,
            'multimask_output': True
        }
        if use_box_input and new_box is not None:
            x1, y1, x2, y2 = new_box
            if (x2 - x1 > 1) and (y2 - y1 > 1):
                predict_kwargs['box'] = np.array([new_box])
        if use_mask_input:
            mask_input = prepare_mask_for_sam(mask)
            predict_kwargs['mask_input'] = mask_input
        try:
            masks, scores, logits = sam_predictor.predict(**predict_kwargs)
            sam_masks_list.append(masks[0])
        except Exception as e:
            print(f"[ERROR] SAM predictor failed: {e}")
            continue
    return sam_masks_list

# =============================================================================
#  4) Main Inference and Evaluation Routine
# =============================================================================

def sam_inference_and_evaluate(inference_pickle_path, images_dir, output_dir,
                               gt_mask_subdir="annotations", gt_suffix="_mask.png",
                               sam_checkpoint='../modelWeights/sam_vit_l.pth', sam_model_type="vit_l",
                               device="cuda", num_points=1, ignore_border_percentage=0.0,
                               algorithm="default", use_box_input=False, use_mask_input=False,
                               box_expansion_rate=0.0, mask_expansion_rate=0.0, iou_type="segm",
                               max_dets=450):
    os.makedirs(output_dir, exist_ok=True)
    with open(inference_pickle_path, "rb") as pf:
        inference_data = pickle.load(pf)
    if sam_checkpoint is None or not os.path.exists(sam_checkpoint):
        raise ValueError(f"SAM checkpoint path is invalid: {sam_checkpoint}")
    print(f"[INFO] Loading SAM model from {sam_checkpoint}")
    sam_predictor = load_sam_model(sam_checkpoint, sam_model_type, device)
    per_image_results = []
    # Prepare accumulators for aggregate metrics
    metrics_accumulators = {
        "refined_mask_AP@50": [], "refined_mask_AP@75": [], "refined_mask_AP@95": [], "refined_mask_AP@50:95": [],
        "refined_mask_AR@50": [], "refined_mask_AR@75": [], "refined_mask_AR@95": [], "refined_mask_AR@50:95": [],
        "refined_box_AP@50": [], "refined_box_AP@75": [], "refined_box_AP@95": [], "refined_box_AP@50:95": [],
        "refined_box_AR@50": [], "refined_box_AR@75": [], "refined_box_AR@95": [], "refined_box_AR@50:95": []
    }
    missing_masks = []
    instance_iou_records = []  # List to store per-object IoU records

    for image_name, image_data in inference_data.items():
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"[WARNING] Image file not found: {image_path}")
            continue
        image_stem, ext = os.path.splitext(image_name)
        gt_mask_path = os.path.join(images_dir, gt_mask_subdir, image_stem + gt_suffix)
        if not os.path.exists(gt_mask_path):
            print(f"[WARNING] GT mask not found for {image_name}: {gt_mask_path}")
            missing_masks.append(image_name)
            continue
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            loop_image = np.array(pil_image)[:, :, ::-1].copy()  # Convert RGB to BGR
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            continue

        boxes = image_data.get("boxes", [])
        polygons = image_data.get("polygons", [])
        masks = image_data.get("masks", [])
        if not masks:
            print(f"[INFO] No masks found for {image_name}. Skipping.")
            continue
        bin_masks = []
        for mask in masks:
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            bin_mask = (mask > 127).astype(np.uint8) if mask.dtype != bool else mask.astype(np.uint8)
            bin_masks.append(bin_mask)
        base_16bit = combine_masks_16bit(bin_masks, return_array=True)
        try:
            sam_masks_list = run_sam_inference(sam_predictor, loop_image, polygons, boxes, bin_masks,
                                                 image_width=pil_image.width, image_height=pil_image.height,
                                                 num_points=num_points, dropout_percentage=0,
                                                 ignore_border_percentage=ignore_border_percentage,
                                                 algorithm=algorithm, use_box_input=use_box_input,
                                                 use_mask_input=use_mask_input,
                                                 box_expansion_rate=box_expansion_rate,
                                                 mask_expansion_rate=mask_expansion_rate)
            if not sam_masks_list:
                print("[WARNING] No SAM masks produced; using an empty mask.")
                sam_16bit = np.zeros_like(base_16bit, dtype=np.uint16)
            else:
                sam_16bit = combine_masks_16bit(sam_masks_list, return_array=True)
        except Exception as e:
            print(f"[ERROR] SAM inference failed for {image_name}: {e}")
            continue

        # ---------------- Compute instance-level IoU (per object) ---------------- #
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_UNCHANGED)
        if gt_mask is None:
            print(f"[ERROR] Could not read GT mask {gt_mask_path}. Skipping.")
            continue
        gt_ids = np.unique(gt_mask)
        gt_ids = gt_ids[gt_ids > 0]
        for gt_id in gt_ids:
            gt_instance = (gt_mask == gt_id)
            best_initial_iou = 0.0
            best_refined_iou = 0.0
            for init_mask in bin_masks:
                intersection = np.logical_and(gt_instance, init_mask > 0).sum()
                union = np.logical_or(gt_instance, init_mask > 0).sum()
                iou = intersection / union if union > 0 else 0
                best_initial_iou = max(best_initial_iou, iou)
            for refined_mask in sam_masks_list:
                intersection = np.logical_and(gt_instance, refined_mask > 0).sum()
                union = np.logical_or(gt_instance, refined_mask > 0).sum()
                iou = intersection / union if union > 0 else 0
                best_refined_iou = max(best_refined_iou, iou)
            instance_iou_records.append({
                "image_name": image_name,
                "inference_file": os.path.basename(inference_pickle_path),
                "object_id": int(gt_id),
                "initial_iou": best_initial_iou,
                "refined_iou": best_refined_iou
            })

        # Save the SAM refined 16-bit mask temporarily
        sam_mask_filename = f"sam_{image_stem}_{uuid.uuid4().hex}.png"
        sam_mask_path = os.path.join(output_dir, sam_mask_filename)
        Image.fromarray(sam_16bit).save(sam_mask_path)

        try:
            gt_data, pred_annotations = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, sam_mask_path, image_path
            )
            segm_metrics = evaluate_coco_metrics(gt_data, pred_annotations, iou_type=iou_type, max_dets=max_dets)
            bbox_metrics = evaluate_coco_metrics(gt_data, pred_annotations, iou_type="bbox", max_dets=max_dets)
        except Exception as e:
            print(f"[ERROR] Failed to evaluate metrics for {image_name}: {e}")
            segm_metrics = {}
            bbox_metrics = {}
        image_result = {
            "image_name": image_name,
            "refined_mask_AP@50": safe_get(segm_metrics, "AP@50"),
            "refined_mask_AP@75": safe_get(segm_metrics, "AP@75"),
            "refined_mask_AP@95": safe_get(segm_metrics, "AP@95"),
            "refined_mask_AP@50:95": safe_get(segm_metrics, "AP@50:95"),
            "refined_mask_AR@50": safe_get(segm_metrics, "AR@50"),
            "refined_mask_AR@75": safe_get(segm_metrics, "AR@75"),
            "refined_mask_AR@95": safe_get(segm_metrics, "AR@95"),
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
        print(f"\n[INFO] Metrics for {image_name}:")
        for k, v in image_result.items():
            if k != "image_name":
                print(f"   {k}: {v}")
        for key in metrics_accumulators.keys():
            val = image_result.get(key, None)
            if val is not None:
                metrics_accumulators[key].append(val)
        per_image_results.append(image_result)
        if os.path.exists(sam_mask_path):
            os.remove(sam_mask_path)
    aggregate_metrics = {k: average_or_none(v) for k, v in metrics_accumulators.items()}
    summary = {
        "missing_masks": missing_masks,
        "aggregate_metrics": aggregate_metrics,
        "per_image_results": per_image_results,
        "instance_iou_records": instance_iou_records
    }
    return summary

# =============================================================================
#  5) Argument Parsing
# =============================================================================

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
                        help='Percentage of bounding box to ignore as border (e.g., 10 means 10%).')
    parser.add_argument('--algorithm', type=str, required=True,
                        help='Point selection algorithm (e.g., "Distance Max", "Voronoi", "Random").')
    parser.add_argument('--use_box_input', type=lambda s: s.lower()=='true', required=True,
                        help='Use box input for SAM ("True" or "False").')
    parser.add_argument('--use_mask_input', type=lambda s: s.lower()=='true', required=True,
                        help='Use mask input for SAM ("True" or "False").')
    parser.add_argument('--box_expansion_rate', type=float, required=True,
                        help='Box expansion rate for SAM inference.')
    parser.add_argument('--mask_expansion_rate', type=float, required=True,
                        help='Mask expansion rate for SAM inference.')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Identifier for the run (e.g., model name).')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to the output CSV file where aggregate metrics will be stored.')
    parser.add_argument('--instance_csv', type=str, required=True,
                        help='Path to the CSV file where instance-level IoU metrics will be stored.')
    return parser.parse_args()

# =============================================================================
#  6) Main Execution
# =============================================================================

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
    write_instance_iou_to_csv(summary.get("instance_iou_records", []), args.instance_csv)
    print(f"\n[INFO] Instance-level IoU metrics saved to {args.instance_csv}.")
    print("\n[SUMMARY] SAM Ablation Evaluation Completed.")
    if summary.get("missing_masks"):
        print("\n[WARNING] Missing GT masks for the following images:")
        for m in summary["missing_masks"]:
            print(f"   {m}")
    else:
        print("\n[INFO] All images have corresponding GT masks.")
    print("\n[SUMMARY] Average metrics across all evaluated images:")
    for k, v in aggregate_metrics.items():
        print(f"   {k}: {v}")

'''
Example usage:
python sam.py \
        --inference_pickle_path "../DualSight/ablationStudy/inference_outputs/yolov8n_inference.pkl" \
        --images_dir "../datasets/powder/test" \
        --output_dir "sam_ablation_outputs" \
        --sam_checkpoint "/path/to/sam_checkpoint.pth" \
        --num_points 16 \
        --ignore_border_percentage 0.1 \
        --algorithm "Distance Max" \
        --use_box_input True \
        --use_mask_input True \
        --box_expansion_rate 0.05 \
        --mask_expansion_rate 0.1 \
        --run_name "YOLOv8 Nano" \
        --output_csv "samAblationMetrics.csv" \
        --instance_csv "samAblationInstance.csv"
'''
