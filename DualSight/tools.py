import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

def visualize_segmentations(image_path, mask_path, output_dir, title="Segmentation Overlay"):
    """
    Visualize segmentation masks by overlaying them on the original image.
    Saves a PNG file with the overlay in the output directory.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise FileNotFoundError(
            f"Image or mask not found: {image_path}, {mask_path}"
        )

    # Extract unique object IDs (excluding 0 which is background)
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]

    # Assign random colors for each object ID
    color_map = {obj_id: tuple(np.random.randint(0, 255, 3)) for obj_id in unique_ids}

    overlay = image.copy()
    for obj_id, color in color_map.items():
        binary_mask = (mask == obj_id).astype(np.uint8)
        colored_mask = np.stack([binary_mask * color[i] for i in range(3)], axis=-1)
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(overlay)
    plt.axis('off')

    os.makedirs(output_dir, exist_ok=True)
    output_overlay_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(output_overlay_path, bbox_inches='tight', pad_inches=0)
    plt.close()

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

def generate_coco_annotations_from_multi_instance_masks(
    gt_mask_path,
    pred_mask_path,
    image_path,
    image_id=1,
    category_id=1
):
    """
    Generate COCO-style annotations for both ground truth and predicted masks.
    """
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

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
            "width": gt_mask.shape[1],
            "height": gt_mask.shape[0],
            "file_name": os.path.basename(image_path)
        }],
        "annotations": gt_annotations,
        "categories": [{"id": category_id, "name": "object"}]
    }

    return gt_data, pred_annotations

def compute_specific_metrics(coco_eval, max_dets=200):
    """
    Helper function to accumulate COCO evaluation metrics and print
    out AP/AR for specific IOU thresholds (0.5, 0.75, 0.95).
    """
    # Adjust parameters as needed
    coco_eval.params.maxDets = [max_dets, max_dets, max_dets]
    # If you want specific IoU thresholds, you can set them here
    coco_eval.params.iouThrs = [0.5, 0.75, 0.95]

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()

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
        ap = np.mean(precision[idx, :, :, :, max_det_index])
        ap_values[f"AP@{int(thr*100)}"] = ap

    # Compute AP across IoU thresholds for all areas (AP@50:95)
    ap_values["AP@50:95"] = np.mean(precision[:, :, :, :, max_det_index])

    # Compute AR for each IoU threshold over all areas
    ar_values = {}
    for thr, idx in iou_thr_indices.items():
        ar = np.mean(recall[idx, :, :, max_det_index])
        ar_values[f"AR@{int(thr*100)}"] = ar

    # Compute AR across IoU thresholds for all areas (AR@50:95)
    ar_values["AR@50:95"] = np.mean(recall[:, :, :, max_det_index])

    print(f"Metrics for maxDets={max_dets}:")
    for metric, value in ap_values.items():
        print(f"{metric}: {value:.3f}")
    for metric, value in ar_values.items():
        print(f"{metric}: {value:.3f}")

def evaluate_coco_metrics(gt_data, pred_data, iou_type="segm", max_dets=200):
    """
    Evaluate COCO metrics (AP, AR) given ground truth data and predicted data
    in COCO format. Prints out results for each metric.
    """
    # Create temporary JSON files
    with open("temp_gt.json", "w") as gt_file:
        json.dump(gt_data, gt_file)

    pred_coco_format = []
    for pred in pred_data:
        # Defaulting the score to 1.0, can be updated if needed
        pred_coco_format.append({
            "image_id": pred["image_id"],
            "category_id": pred["category_id"],
            "segmentation": pred["segmentation"],
            "score": 1.0,
        })

    with open("temp_pred.json", "w") as pred_file:
        json.dump(pred_coco_format, pred_file)

    coco_gt = COCO("temp_gt.json")
    coco_pred = coco_gt.loadRes("temp_pred.json")
    coco_eval = COCOeval(coco_gt, coco_pred, iouType=iou_type)

    # Compute and print custom metrics
    compute_specific_metrics(coco_eval, max_dets=max_dets)
