import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from samDemo import show_mask, show_box
from pointSelection import select_furthest_points_from_mask
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval

# Directories and Paths
output_dir = "outputImages/eval"
os.makedirs(output_dir, exist_ok=True)

# Initialize YOLO model
yolo_model_path = "/home/sprice/RQ/models/yolov8n-seg-train2/weights/best.pt"
model = YOLO(yolo_model_path)

# Initialize SAM model and predictor
sam_checkpoint = "/home/sprice/RQ/models/sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Input image path
image_path = '/home/sprice/RQ/demo.v7i.yolov8/test/images/TruForm174-2_00-37_500X16_png.rf.17ddf81ab4dd63c5ce6f66654a48c5b4.jpg'
image = Image.open(image_path)
image_rgba = image.convert("RGBA")
loop_image = cv2.imread(image_path)

def visualize_segmentations(image_path, mask_path, title="Segmentation Overlay"):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(f"Image or mask not found: {image_path}, {mask_path}")
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]
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
    output_overlay_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(output_overlay_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def mask_to_coco_format(binary_mask, image_id, category_id=1, annotation_id=1):
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
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

def generate_coco_annotations_from_multi_instance_masks(gt_mask_path, pred_mask_path, image_id=1, category_id=1):
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
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
    return {
        "images": [{
            "id": image_id,
            "width": gt_mask.shape[1],
            "height": gt_mask.shape[0],
            "file_name": os.path.basename(image_path)
        }],
        "annotations": gt_annotations,
        "categories": [{"id": category_id, "name": "object"}]
    }, pred_annotations

def compute_specific_metrics(coco_eval, max_dets=200):
    # Adjust parameters as needed
    coco_eval.params.maxDets = [max_dets, max_dets, max_dets]
    coco_eval.params.iouThrs = [0.5, 0.75, 0.95]  # Use desired IoU thresholds

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
        # Average over categories and area ranges for given IoU and maxDets. 
        # Recall array shape: [T, K, A, M] so we average over K and A.
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
    from pycocotools.coco import COCO
    with open("temp_gt.json", "w") as gt_file:
        json.dump(gt_data, gt_file)
    pred_coco_format = []
    for pred in pred_data:
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

    compute_specific_metrics(coco_eval, max_dets=max_dets)

# Data structures to store results
listOfPolygons = []
listOfBoxes = []
listOfMasks = []
sam_masks_list = []
yolo_masks_list = []

# Run YOLO inference
results = model(image)
count = 0
for result in results[0]:
    mask = result.masks.data
    if mask.sum() > 200:
        polygon_points = result.masks.xy[0]
        if len(polygon_points) > 2:
            listOfPolygons.append(polygon_points)
            listOfBoxes.append(result.boxes.xyxy[0])
            mask_image = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask_image)
            draw.polygon(polygon_points, outline=1, fill=1)
            mask_array = np.array(mask_image) * 255
            listOfMasks.append(mask_array)
            yolo_masks_list.append(mask_array)
            count += 1

# Sequential SAM processing
for INDEX in range(len(listOfPolygons)):
    box = listOfBoxes[INDEX]
    box = box.cpu().numpy() if isinstance(box, torch.Tensor) else np.array(box)
    mask = listOfMasks[INDEX]
    def expand_bbox_within_border(x1, y1, x2, y2, width, height, expansion_rate=0.0):
        return [x1, y1, x2, y2]
    box = np.array(expand_bbox_within_border(
        box[0], box[1], box[2], box[3], image.width, image.height, expansion_rate=0.0
    ))
    selected_points, _, _ = select_furthest_points_from_mask(
        mask=mask,
        num_points=4,
        dropout_percentage=0,
        ignore_border_percentage=5,
        algorithm="Voronoi"
    )
    op_y, op_x = zip(*selected_points)
    predictor.set_image(loop_image)
    input_point = np.array(list(zip(op_x, op_y)))
    input_label = np.array([1] * len(input_point))
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=box[None, :],
        multimask_output=True,
    )
    mask_out = masks[0]
    sam_masks_list.append(mask_out)

# Combine SAM masks into a single grayscale mask
if sam_masks_list:
    height, width = sam_masks_list[0].shape
    combined_sam_mask = np.zeros((height, width), dtype=np.uint8)
    for idx, mask in enumerate(sam_masks_list, start=1):
        combined_sam_mask[mask > 0] = idx
    combined_sam_mask_path = os.path.join(output_dir, 'combined_sam_mask.png')
    cv2.imwrite(combined_sam_mask_path, combined_sam_mask)
    print("Combined SAM mask saved to", combined_sam_mask_path)

# Combine YOLO masks into a single grayscale mask
if yolo_masks_list:
    height, width = yolo_masks_list[0].shape
    combined_yolo_mask = np.zeros((height, width), dtype=np.uint8)
    for idx, mask in enumerate(yolo_masks_list, start=1):
        combined_yolo_mask[mask > 0] = idx
    combined_yolo_mask_path = os.path.join(output_dir, 'combined_yolo_mask.png')
    cv2.imwrite(combined_yolo_mask_path, combined_yolo_mask)
    print("Combined YOLO mask saved to", combined_yolo_mask_path)

# Visualization
visualize_segmentations(image_path, combined_sam_mask_path, title="Combined SAM Segmentation")
visualize_segmentations(image_path, combined_yolo_mask_path, title="Combined YOLO Segmentation")

# Evaluate metrics for YOLO (Segmentation)
gt_mask_path = "/home/sprice/ICCV25/SegRefinerV2/test_masks/TruForm174-2_00-37_500X16_png.rf.17ddf81ab4dd63c5ce6f66654a48c5b4_mask.png"
gt_data, yolo_pred_data = generate_coco_annotations_from_multi_instance_masks(gt_mask_path, combined_yolo_mask_path)
print("Evaluating YOLO Segmentation Metrics:")
evaluate_coco_metrics(gt_data, yolo_pred_data, iou_type="segm", max_dets=350)

print("\nEvaluating YOLO Bounding Box Metrics:")
# For bounding boxes, you'll need annotations formatted for bounding boxes. This code currently uses segmentation annotations.
# You would normally generate box annotations differently. For demonstration, we'll use the same annotations but with iouType="bbox".
evaluate_coco_metrics(gt_data, yolo_pred_data, iou_type="bbox", max_dets=350)

print("\nEvaluating SAM Segmentation Metrics:")
gt_data, sam_pred_data = generate_coco_annotations_from_multi_instance_masks(gt_mask_path, combined_sam_mask_path)
evaluate_coco_metrics(gt_data, sam_pred_data, iou_type="segm", max_dets=350)

print("\nEvaluating SAM Bounding Box Metrics:")
# As above, for bounding boxes with SAM predictions, use iouType="bbox".
evaluate_coco_metrics(gt_data, sam_pred_data, iou_type="bbox", max_dets=350)