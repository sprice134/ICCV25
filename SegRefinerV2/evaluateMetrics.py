# Install pycocotools if not already installed
# pip install pycocotools

import numpy as np
import cv2
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
import json
import matplotlib.pyplot as plt
import os

def save_individual_overlays(image_path, mask_path, output_dir, prefix="segmentation"):
    """
    Save each object segmentation as an overlay on the original image.
    """
    # Load the original image and mask
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for saving with Matplotlib
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise FileNotFoundError(f"Image or mask not found: {image_path}, {mask_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique object IDs from the mask
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]  # Exclude background

    # Save each object overlay
    for obj_id in unique_ids:
        binary_mask = (mask == obj_id).astype(np.uint8)
        colored_mask = np.zeros_like(image)
        color = np.random.randint(0, 255, size=3)  # Generate a random color for the object
        for i in range(3):  # Apply color to each channel
            colored_mask[:, :, i] = binary_mask * color[i]

        # Overlay the mask on the original image
        overlay = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

        # Save the overlay
        output_path = os.path.join(output_dir, f"{prefix}_object_{obj_id}.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved overlay for object {obj_id} at: {output_path}")



def visualize_segmentations(image_path, mask_path, title="Segmentation Overlay"):
    """
    Visualize segmentations as overlays on the original image.
    """
    # Load the original image and mask
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise FileNotFoundError(f"Image or mask not found: {image_path}, {mask_path}")

    # Generate a color map for unique object IDs in the mask
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]  # Exclude background
    color_map = {obj_id: tuple(np.random.randint(0, 255, 3)) for obj_id in unique_ids}

    # Overlay masks on the image
    overlay = image.copy()
    for obj_id, color in color_map.items():
        binary_mask = (mask == obj_id).astype(np.uint8)
        colored_mask = np.stack([binary_mask * color[i] for i in range(3)], axis=-1)
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

    # Visualize the overlay
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig(f'{title}.png')


def mask_to_coco_format(binary_mask, image_id, category_id=1, annotation_id=1):
    """
    Convert a binary mask to COCO RLE (Run-Length Encoding) format.
    """
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": float(maskUtils.area(rle)),
        "bbox": maskUtils.toBbox(rle).tolist(),
        "iscrowd": 0
    }
    return annotation

def compute_detailed_metrics(coco_eval, max_dets=500):
    """
    Compute detailed COCO metrics for a specific max_dets value:
    - AP 50, AP 75, AP 95, AP 50:95
    - AR 50, AR 75, AR 95, AR 50:95
    """
    p = coco_eval.params
    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
    max_det_index = p.maxDets.index(max_dets) if max_dets in p.maxDets else None

    if max_det_index is None:
        print(f"maxDets={max_dets} not found in params.maxDets.")
        return

    # Extract precision and recall arrays
    precision = coco_eval.eval['precision']
    recall = coco_eval.eval['recall']

    if precision.shape[0] == 0 or recall.shape[0] == 0:
        print("No precision or recall values available. Check evaluation setup.")
        return

    # Initialize metrics
    metrics = {
        "AP@50": None,
        "AP@75": None,
        "AP@95": None,
        "AP@50:95": None,
        "AR@50": None,
        "AR@75": None,
        "AR@95": None,
        "AR@50:95": None
    }

    # Compute AP metrics
    for i, iou_thr in enumerate(iou_thresholds):
        if iou_thr == 0.5:
            metrics["AP@50"] = np.mean(precision[i, :, :, :, max_det_index])
        elif iou_thr == 0.75:
            metrics["AP@75"] = np.mean(precision[i, :, :, :, max_det_index])
        elif iou_thr == 0.95:
            metrics["AP@95"] = np.mean(precision[i, :, :, :, max_det_index])

    # Compute mean AP@50:95
    metrics["AP@50:95"] = np.mean(precision[:, :, :, :, max_det_index])

    # Compute AR metrics
    for i, iou_thr in enumerate(iou_thresholds):
        if iou_thr == 0.5:
            metrics["AR@50"] = np.mean(recall[i, :, :, max_det_index])
        elif iou_thr == 0.75:
            metrics["AR@75"] = np.mean(recall[i, :, :, max_det_index])
        elif iou_thr == 0.95:
            metrics["AR@95"] = np.mean(recall[i, :, :, max_det_index])

    # Compute mean AR@50:95
    metrics["AR@50:95"] = np.mean(recall[:, :, :, max_det_index])

    # Print metrics
    print(f"Metrics for maxDets={max_dets}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}" if value is not None else f"{metric}: Not Computed")

    return metrics


def generate_coco_annotations_from_multi_instance_masks(gt_mask_path, pred_mask_path, image_id=1, category_id=1):
    """
    Generate COCO-style annotations for ground truth and predictions from multi-instance masks.
    """
    # Load ground truth and predicted masks
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

    if gt_mask is None or pred_mask is None:
        raise FileNotFoundError("One of the mask files was not found.")

    # Visualize the masks
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth Mask")
    plt.imshow(gt_mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="gray")
    plt.savefig('images.png')

    # Get unique object IDs in the ground truth and prediction masks
    gt_ids = np.unique(gt_mask)
    pred_ids = np.unique(pred_mask)

    # Remove the background ID (0)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = pred_ids[pred_ids > 0]

    # Create COCO annotations
    gt_annotations = []
    pred_annotations = []

    for annotation_id, obj_id in enumerate(gt_ids, start=1):
        binary_mask = (gt_mask == obj_id).astype(np.uint8)  # Extract binary mask for the object
        gt_annotations.append(mask_to_coco_format(binary_mask, image_id, category_id, annotation_id))

    for annotation_id, obj_id in enumerate(pred_ids, start=1):
        binary_mask = (pred_mask == obj_id).astype(np.uint8)  # Extract binary mask for the object
        pred_annotations.append(mask_to_coco_format(binary_mask, image_id, category_id, annotation_id))

    return {
        "images": [{
            "id": image_id,
            "width": gt_mask.shape[1],
            "height": gt_mask.shape[0],
            "file_name": "image.jpg"
        }],
        "annotations": gt_annotations,
        "categories": [{"id": category_id, "name": "object"}]
    }, pred_annotations

def evaluate_coco_metrics(gt_data, pred_data):
    """
    Evaluate COCO metrics using ground truth and prediction data in COCO format.
    Extracts and displays IoU values along with standard COCO metrics.
    """
    # Save the ground truth data in JSON format
    with open("temp_gt.json", "w") as gt_file:
        json.dump(gt_data, gt_file)

    # Save the predictions in the correct COCO format
    pred_coco_format = []
    for pred in pred_data:
        pred_coco_format.append({
            "image_id": pred["image_id"],
            "category_id": pred["category_id"],
            "segmentation": pred["segmentation"],
            "score": 1.0,  # Add a confidence score for COCO format
        })
    with open("temp_pred.json", "w") as pred_file:
        json.dump(pred_coco_format, pred_file)

    # Load ground truth and prediction data
    from pycocotools.coco import COCO
    coco_gt = COCO("temp_gt.json")
    coco_pred = coco_gt.loadRes("temp_pred.json")

    # Evaluate COCO metrics
    coco_eval = COCOeval(coco_gt, coco_pred, iouType="segm")
    coco_eval.params.maxDets = [5, 50, 500]
    coco_eval.evaluate()  # Populate eval dictionary
    coco_eval.accumulate()  # Summarize results
    coco_eval.summarize()  # Print results
    
    detailed_metrics = compute_detailed_metrics(coco_eval, max_dets=500)


    # Compute mean IoU (mIoU)
    if 'ious' in coco_eval.eval:
        iou_values = coco_eval.eval['ious']  # IoUs are stored as { (image_id, category_id): ndarray of IoUs }
        all_ious = []
        for key, iou_array in iou_values.items():
            # iou_array contains IoUs for different detections of the same instance
            all_ious.extend(iou_array)  # Collect all IoUs

        if all_ious:
            mean_iou = np.mean(all_ious)  # Compute the mean IoU
            print(f"\nMean IoU (mIoU): {mean_iou:.4f}")
        else:
            print("No IoU values were computed. Check the inputs.")
    else:
        print("IoU values could not be extracted. Ensure evaluation was successful.")

# Paths to ground truth and predicted 
# original_image_path = 'HP743_5S_500x_png.rf.9ff406796462449f85c2039537f32d6f.jpg'
original_image_path = 'S02_02_SE1_300X18_png.rf.1a16e8c5f4e008cb2fc48c98b35778fb.jpg'
# gt_mask_path = "/home/sprice/ICCV25/SegRefinerV2/test_masks/HP743_5S_500x_png.rf.9ff406796462449f85c2039537f32d6f_mask.png"  # Replace with your ground truth mask path
gt_mask_path = "/home/sprice/ICCV25/SegRefinerV2/test_masks/S02_02_SE1_300X18_png.rf.1a16e8c5f4e008cb2fc48c98b35778fb_mask.png"  # Replace with your ground truth mask path
pred_mask_path = "yoloMasks.png"  # Replace with your predicted mask path
refined_mask_path = "refined_grayscale.png"

# Save individual masks for ground truth

# save_individual_overlays(original_image_path, gt_mask_path, "ground_truth_masks", prefix="ground_truth")
# save_individual_overlays(original_image_path, pred_mask_path, "predicted_masks", prefix="predicted")
# save_individual_overlays(original_image_path, refined_mask_path, "refined_masks", prefix="refined")


# Visualize the segmentations
visualize_segmentations(original_image_path, gt_mask_path, title="Ground Truth Segmentation")
visualize_segmentations(original_image_path, pred_mask_path, title="Predicted Segmentation (YOLO)")
visualize_segmentations(original_image_path, refined_mask_path, title="Refined Segmentation")


gt_data, pred_data = generate_coco_annotations_from_multi_instance_masks(gt_mask_path, pred_mask_path)
evaluate_coco_metrics(gt_data, pred_data)

gt_data, refined_data = generate_coco_annotations_from_multi_instance_masks(gt_mask_path, refined_mask_path)
evaluate_coco_metrics(gt_data, refined_data)
