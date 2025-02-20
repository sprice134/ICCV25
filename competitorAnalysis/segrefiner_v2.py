#!/usr/bin/env python3
import os
import pickle
import cv2
import numpy as np
import json
import sys
import argparse
import csv

# pycocotools imports
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# mmcv / mmdet imports for SegRefiner
import sys
sys.path.append('../SegRefiner/')
import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.utils import replace_cfg_vals, rfnext_init_model
from mmdet.core.mask import BitmapMasks

# ------------------ Utility Functions (for combination, COCO evaluation, etc.) ------------------ #
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

def compute_specific_metrics(coco_eval, max_dets=200):
    # Adjust parameters as needed
    coco_eval.params.maxDets = [max_dets, max_dets, max_dets]
    # Set specific IoU thresholds
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

    pred_coco_format = []
    for pred in pred_data:
        pred_entry = {
            "image_id": pred["image_id"],
            "category_id": pred["category_id"],
            "score": pred.get("score", 1.0)  # default score 1.0 if missing
        }
        if iou_type == "segm":
            pred_entry["segmentation"] = pred["segmentation"]
        elif iou_type == "bbox":
            pred_entry["bbox"] = pred["bbox"]
        pred_coco_format.append(pred_entry)

    with open("temp_pred.json", "w") as pred_file:
        json.dump(pred_coco_format, pred_file)

    # Load into COCO
    coco_gt = COCO("temp_gt.json")
    coco_pred = coco_gt.loadRes("temp_pred.json")
    coco_eval = COCOeval(coco_gt, coco_pred, iouType=iou_type)

    # Compute
    metrics = compute_specific_metrics(coco_eval, max_dets=max_dets)

    # Remove temp files
    os.remove("temp_gt.json")
    os.remove("temp_pred.json")

    return metrics


def mask_to_coco_format(binary_mask, image_id, category_id=1, annotation_id=1):
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
            mask_to_coco_format(binary_mask, image_id, category_id, annotation_id)
        )

    # Create Predicted annotations
    for annotation_id, obj_id in enumerate(pred_ids, start=1):
        binary_mask = (pred_mask == obj_id).astype(np.uint8)
        pred_annotations.append(
            mask_to_coco_format(binary_mask, image_id, category_id, annotation_id)
        )

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


# Simple helpers
def safe_get(metrics_dict, key):
    return metrics_dict[key] if (metrics_dict is not None and key in metrics_dict) else None

def average_or_none(values):
    return float(np.mean(values)) if values else None

# ------------------ SegRefiner Core ------------------ #
def load_segrefiner_model(segrefiner_config, segrefiner_checkpoint, device='cuda'):
    """
    Build and load the SegRefiner model from config and checkpoint.
    """
    cfg = Config.fromfile(segrefiner_config)
    cfg = replace_cfg_vals(cfg)

    # Device setup
    cfg.device = device
    cfg.gpu_ids = [0] if 'cuda' in device else []

    # If 'pretrained' is in cfg.model, set it to None to avoid confusion
    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None

    # Build the model (test mode)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    rfnext_init_model(model, cfg=cfg)

    # Load checkpoint
    load_checkpoint(model, segrefiner_checkpoint, map_location='cpu', strict=True)
    model = model.to(device)
    model.eval()
    return model, cfg


def refine_masks_with_segrefiner(bgr_image, coarse_mask_16bit, model, cfg, device='cuda'):
    """
    Given an original BGR image (np.ndarray) and a multi-instance 16-bit coarse mask,
    refine each instance using SegRefiner model. Returns a 16-bit refined mask.

    Steps:
      - Convert the BGR image to the normalized input the model expects.
      - Convert the 16-bit coarse mask to an 8-bit grayscale with distinct IDs.
      - For each unique ID > 0, refine separately and write back into a final mask.
    """
    # Convert BGR to the color format (e.g., RGB) if needed by your normalization
    img = bgr_image
    if 'img_norm_cfg' in cfg:
        mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
        to_rgb = cfg.img_norm_cfg.get('to_rgb', True)
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = mmcv.imnormalize(img, mean=mean, std=std, to_rgb=False)
    else:
        # Minimal handling if no norm is specified
        img = img.astype(np.float32)

    # Create torch tensor (C,H,W)
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Convert the 16-bit to 8-bit if the IDs do not exceed 255
    max_id = coarse_mask_16bit.max()
    if max_id > 255:
        raise ValueError(f"[ERROR] refine_masks_with_segrefiner: instance ID {max_id} > 255 found. "
                         "Please revise approach or reduce instance IDs.")
    coarse_mask_8bit = coarse_mask_16bit.astype(np.uint8)

    height, width = coarse_mask_8bit.shape
    unique_ids = np.unique(coarse_mask_8bit)
    unique_ids = unique_ids[unique_ids > 0]  # exclude background

    refined_mask_16bit = np.zeros((height, width), dtype=np.uint16)

    # Minimal meta info for the model
    img_metas = {
        'ori_filename': 'placeholder.jpg',
        'img_shape': (height, width, 3),
        'ori_shape': (height, width, 3),
        'pad_shape': (height, width, 3),
        'scale_factor': 1.0,
        'flip': False,
    }

    # Refine each instance
    for idx, obj_id in enumerate(unique_ids, start=1):
        single_obj_mask = (coarse_mask_8bit == obj_id).astype(np.uint8)
        single_obj_mask = single_obj_mask[np.newaxis, :, :]  # (1, H, W)

        coarse_bitmap_mask = BitmapMasks(single_obj_mask, height=height, width=width)
        data = {
            'img': img_tensor,
            'img_metas': [img_metas],
            'coarse_masks': [coarse_bitmap_mask],
        }

        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)
        # The model returns a refined mask in results[0][0] (depending on your config)
        refined = results[0][0]
        if isinstance(refined, torch.Tensor):
            refined = refined.cpu().numpy()
        # If the shape is (3,H,W), convert to (H,W,3) or handle accordingly
        if refined.ndim == 3 and refined.shape[0] == 3:
            refined = refined.transpose(1, 2, 0)
        refined = (refined * 255).astype(np.uint8)

        # Write the refined object ID back into final mask
        # any pixel > 0 belongs to the object
        refined_mask_16bit[refined > 0] = idx

    return refined_mask_16bit


# ------------------ Main refinement + evaluation logic ------------------ #
def refine_and_evaluate_segrefiner(
    inference_pickle_path,
    images_dir,
    output_dir,
    segrefiner_config,
    segrefiner_checkpoint,
    gt_mask_subdir="annotations",
    gt_suffix="_mask.png"
):
    """
    1) Loads precomputed predictions from a pickle file (with "masks" entries).
    2) Combines predicted masks for each image into a single multi-instance 16-bit mask.
    3) Uses SegRefiner to refine the multi-instance mask.
    4) Evaluates with COCO metrics vs. ground truth masks.
    5) Returns summary including aggregate and per-image metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load inference data
    with open(inference_pickle_path, "rb") as pf:
        inference_data = pickle.load(pf)

    # Initialize the SegRefiner model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    segrefiner_model, segrefiner_cfg = load_segrefiner_model(
        segrefiner_config,
        segrefiner_checkpoint,
        device=device
    )

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

    missing_masks = []
    per_image_results = []
    gt_mask_dir = os.path.join(images_dir, gt_mask_subdir)

    for image_name, preds in inference_data.items():
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"[WARNING] Image file not found: {image_path}")
            continue

        # Locate GT mask
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

        # Predicted masks
        predicted_masks = preds.get("masks", [])
        if not predicted_masks:
            print(f"[INFO] No predicted masks found for {image_name}. Skipping.")
            continue

        # 1) Combine the predicted masks into a single 16-bit coarse mask
        try:
            coarse_16bit = combine_masks_16bit(predicted_masks, return_array=True)
            if coarse_16bit is None:
                print(f"[WARNING] No combined mask for {image_name}; skipping.")
                continue
        except ValueError as ve:
            print(f"[ERROR] Combining masks for {image_name} failed: {ve}")
            continue

        # 2) Refine the coarse mask with SegRefiner
        try:
            refined_16bit = refine_masks_with_segrefiner(
                bgr_image=bgr_image,
                coarse_mask_16bit=coarse_16bit,
                model=segrefiner_model,
                cfg=segrefiner_cfg,
                device=device
            )
        except Exception as e:
            print(f"[ERROR] SegRefiner refinement failed for {image_name}: {e}")
            continue

        # 3) Save the refined mask
        refined_16bit_path = os.path.join(output_dir, f"refined_{image_stem}.png")
        cv2.imwrite(refined_16bit_path, refined_16bit)

        # 4) Evaluate with COCO metrics (both segmentation and bbox)
        try:
            gt_data, pred_data_coco = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, refined_16bit_path, image_path
            )
            segm_metrics = evaluate_coco_metrics(
                gt_data, pred_data_coco, iou_type="segm", max_dets=450
            )
            bbox_metrics = evaluate_coco_metrics(
                gt_data, pred_data_coco, iou_type="bbox", max_dets=450
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

        print(f"\n[INFO] Results for '{image_name}':")
        for k, v in image_result.items():
            if k != "image_name":
                print(f"   {k}: {v}")

        # Accumulate
        for mkey in metrics_accumulators.keys():
            val = image_result.get(mkey, None)
            if val is not None:
                metrics_accumulators[mkey].append(val)

        per_image_results.append(image_result)

    # Aggregate
    aggregate_metrics = {}
    for mkey, vals in metrics_accumulators.items():
        aggregate_metrics[mkey] = average_or_none(vals)

    summary = {
        "missing_masks": missing_masks,
        "aggregate_metrics": aggregate_metrics,
        "per_image_results": per_image_results
    }
    return summary


def write_metrics_to_csv(run_name, metrics, output_csv_path):
    """
    Writes the metrics dictionary to a CSV file with the given run_name.
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
    parser = argparse.ArgumentParser(description="Refine masks using SegRefiner and evaluate via COCO metrics.")
    parser.add_argument('--inference_pickle_path', type=str, required=True,
                        help='Path to the pickle file containing predictions.')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing the original test images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to store temporary or final outputs.')
    parser.add_argument('--segrefiner_config', type=str, required=True,
                        help='Path to the SegRefiner config file.')
    parser.add_argument('--segrefiner_checkpoint', type=str, required=True,
                        help='Path to the SegRefiner checkpoint file.')
    parser.add_argument('--gt_mask_subdir', type=str, default="annotations",
                        help='Subdirectory under images_dir where GT masks are stored. Default: "annotations".')
    parser.add_argument('--gt_suffix', type=str, default="_mask.png",
                        help='Suffix appended to the base image name to locate its ground-truth mask.')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Identifier for the run (e.g., model name). Used in CSV output.')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to the output CSV file where metrics will be stored.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Perform refinement and evaluation using SegRefiner
    summary = refine_and_evaluate_segrefiner(
        inference_pickle_path=args.inference_pickle_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        segrefiner_config=args.segrefiner_config,
        segrefiner_checkpoint=args.segrefiner_checkpoint,
        gt_mask_subdir=args.gt_mask_subdir,
        gt_suffix=args.gt_suffix,
    )

    # Prepare metrics for CSV
    aggregate_metrics = summary.get("aggregate_metrics", {})
    run_name = args.run_name

    # Write metrics to CSV
    write_metrics_to_csv(run_name, aggregate_metrics, args.output_csv)

    # Print summary
    print("\n[SUMMARY] SegRefiner Evaluation Completed.")
    missing_masks = summary.get("missing_masks", [])
    if missing_masks:
        print("\n[WARNING] The following images are missing ground truth masks:")
        for img in missing_masks:
            print(f"   {img}")
    else:
        print("\n[INFO] All images have corresponding ground truth masks.")

    print("\n[SUMMARY] Average metrics across all evaluated images:")
    for mkey, val in aggregate_metrics.items():
        print(f"   {mkey}: {val}")

'''
python segrefiner.py \
    --inference_pickle_path "../DualSight/ablationStudy/inference_outputs/yolov8n_inference.pkl" \
    --images_dir "../demo.v7i.coco/test" \
    --output_dir "segrefiner_refined_outputs" \
    --segrefiner_config "/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py" \
    --segrefiner_checkpoint "/home/sprice/ICCV25/modelWeights/segrefiner_lr_latest.pth" \
    --run_name "yolov8n" \
    --output_csv "segrefinerMetrics.csv"
'''
