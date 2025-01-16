import os
import cv2
import glob
import csv
import numpy as np
from PIL import Image

# ---------------------------
# Local or custom modules
# ---------------------------
from models import load_trained_model, get_inference_predictions
from sam_helper import load_sam_model, run_sam_inference
from segrefiner_helper import load_segrefiner_model, run_segrefiner_inference
# NOTE: We will not use the old combine_masks; we’ll introduce combine_masks_16bit.
# from sam_helper import combine_masks

# SegRefiner-related imports
import sys
sys.path.append('../SegRefinerV2/')
import mmcv


# Tools or helper functions
from tools import (
    visualize_segmentations,
    generate_coco_annotations_from_multi_instance_masks,
    generate_coco_annotations_from_multi_instance_masks_16bit,
    evaluate_coco_metrics
)

# ---------------------------------------------------------------------
# 1) 16-bit "combine masks" function
# ---------------------------------------------------------------------
def combine_masks_16bit(list_of_binary_masks, output_path):
    """
    Given a list of binary masks (each shape [H, W], 0=bg,1=fg),
    combine them into a single 16-bit ID mask, assigning unique
    IDs for each mask: mask #1 => ID=1, #2 => ID=2, etc.
    """
    if not list_of_binary_masks:
        print("[WARNING] No masks to combine.")
        return

    height, width = list_of_binary_masks[0].shape
    combined_16bit = np.zeros((height, width), dtype=np.uint16)

    for idx, mask in enumerate(list_of_binary_masks, start=1):
        combined_16bit[mask > 0] = idx

    # Use mmcv to write a 16-bit single-channel PNG
    mmcv.imwrite(combined_16bit, output_path)
    # If you prefer, you can also use cv2:
    # cv2.imwrite(output_path, combined_16bit)
    return

# ---------------------------------------------------------------------
# 3) Main pipeline
# ---------------------------------------------------------------------
def main():
    # Directories and Paths
    output_dir = "outputImages"
    os.makedirs(output_dir, exist_ok=True)

    # Folders containing images and ground-truth masks
    image_dir = "/home/sprice/RQ/demo.v7i.yolov8/test/images/"
    gt_mask_dir = "/home/sprice/ICCV25/SegRefinerV2/output_masks_16bit/"

    # -------------------------------------------------------
    # 1) Load YOLO model (or other model)
    # -------------------------------------------------------
    chosen_model_type = "yolo"
    model_path = "/home/sprice/ICCV25/modelWeights/yolov8n-seg.pt"
    device = "cuda"
    model = load_trained_model(chosen_model_type, model_path, device=device)

    # -------------------------------------------------------
    # 2) Load SAM predictor
    # -------------------------------------------------------
    sam_model_type = "vit_l"
    sam_checkpoint = f"/home/sprice/ICCV25/modelWeights/sam_{sam_model_type}.pth"
    sam_predictor = load_sam_model(
        sam_checkpoint=sam_checkpoint,
        model_type=sam_model_type,
        device=device
    )

    # -------------------------------------------------------
    # 3) Load SegRefiner model
    # -------------------------------------------------------
    segrefiner_config_path = "/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py"
    segrefiner_checkpoint_path = "/home/sprice/ICCV25/modelWeights/segrefiner_lr_latest.pth"
    segrefiner_model, segrefiner_cfg = load_segrefiner_model(
        segrefiner_config_path,
        segrefiner_checkpoint_path,
        device=device
    )

    # -------------------------------------------------------
    # Prepare list to store per-image metrics
    # We'll store bounding-box + segmentation for:
    #   - YOLO
    #   - YOLO+SAM
    #   - YOLO+SegRefiner
    # => 2 (bbox/segm) * 8 (AP/AR) * 3 methods = 48 columns, plus image_name => 49
    # -------------------------------------------------------
    results = []

    # -------------------------------------------------------
    # 4) Get all .jpg images
    # -------------------------------------------------------
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

    for img_path in image_paths:
        image_name = os.path.basename(img_path)

        gt_mask_name = image_name.replace(".jpg", "_mask.png")
        gt_mask_path = os.path.join(gt_mask_dir, gt_mask_name)

        if not os.path.exists(gt_mask_path):
            print(f"[WARNING] GT mask not found for {image_name}, skipping...")
            continue

        # Load images
        loop_image = cv2.imread(img_path)
        pil_image = Image.open(img_path)
        if loop_image is None:
            print(f"[ERROR] Could not read image {img_path}")
            continue

        # -------------------------------------------------------
        # 5) YOLO Inference => listOfMasks (coarse)
        # -------------------------------------------------------
        listOfPolygons, listOfBoxes, listOfMasks = get_inference_predictions(
            model,
            chosen_model_type,
            img_path,
            device=device
        )
        # listOfMasks is typically a list of [H,W] binary arrays

        # -------------------------------------------------------
        # 6) YOLO => 16-bit multi-instance mask
        #    (So we can evaluate YOLO as-is, ignoring SAM/SegRefiner)
        # -------------------------------------------------------
        combined_yolo_mask_path = os.path.join(
            output_dir, f"{image_name.replace('.jpg', '_yolo_16bit.png')}"
        )
        combine_masks_16bit(listOfMasks, combined_yolo_mask_path)

        # -------------------------------------------------------
        # 7) YOLO+SAM => refine each mask individually with SAM
        # -------------------------------------------------------
        sam_masks_list = run_sam_inference(
            predictor=sam_predictor,
            loop_image=loop_image,
            listOfPolygons=listOfPolygons,
            listOfBoxes=listOfBoxes,
            listOfMasks=listOfMasks,
            image_width=pil_image.width,
            image_height=pil_image.height,
            num_points=4,
            dropout_percentage=0,
            ignore_border_percentage=5,
            algorithm="Voronoi",
            use_mask_input=False,
            box_expansion_rate=0.0
        )
        # sam_masks_list is also a list of binary masks

        combined_sam_mask_path = os.path.join(
            output_dir, f"{image_name.replace('.jpg', '_sam_16bit.png')}"
        )
        combine_masks_16bit(sam_masks_list, combined_sam_mask_path)

        # -------------------------------------------------------
        # 8) YOLO+SegRefiner => refine each YOLO mask with segrefiner
        # -------------------------------------------------------
        segrefiner_masks_list = run_segrefiner_inference(
            segrefiner_model, segrefiner_cfg,
            list_of_binary_masks=listOfMasks,
            loop_image_bgr=loop_image,
            device=device
        )
        segrefiner_mask_path = os.path.join(
            output_dir, f"{image_name.replace('.jpg', '_segrefiner_16bit.png')}"
        )
        combine_masks_16bit(segrefiner_masks_list, segrefiner_mask_path)

        # -------------------------------------------------------
        # 9) Optional Visualization
        # -------------------------------------------------------
        try:
            visualize_segmentations(
                image_path=img_path,
                mask_path=combined_yolo_mask_path,
                output_dir=output_dir,
                title="YOLO"
            )
            visualize_segmentations(
                image_path=img_path,
                mask_path=combined_sam_mask_path,
                output_dir=output_dir,
                title="YOLO+SAM"
            )
            visualize_segmentations(
                image_path=img_path,
                mask_path=segrefiner_mask_path,
                output_dir=output_dir,
                title="YOLO+SegRefiner"
            )
        except Exception as e:
            print(f"[WARNING] Visualization error for {image_name}: {e}")

        # -------------------------------------------------------
        # 10) Evaluate each approach
        # -------------------------------------------------------
        # Evaluate YOLO
        gt_data, yolo_pred_data = generate_coco_annotations_from_multi_instance_masks_16bit(
            gt_mask_path, combined_yolo_mask_path, img_path
        )
        # print("DEBUG: yolo_pred_data =", yolo_pred_data)
        yolo_segm_metrics = evaluate_coco_metrics(gt_data, yolo_pred_data, iou_type="segm", max_dets=350)
        yolo_bbox_metrics = evaluate_coco_metrics(gt_data, yolo_pred_data, iou_type="bbox", max_dets=350)

        # Evaluate YOLO+SAM
        gt_data, sam_pred_data = generate_coco_annotations_from_multi_instance_masks_16bit(
            gt_mask_path, combined_sam_mask_path, img_path
        )
        # print("DEBUG: sam_pred_data =", sam_pred_data)
        sam_segm_metrics = evaluate_coco_metrics(gt_data, sam_pred_data, iou_type="segm", max_dets=350)
        sam_bbox_metrics = evaluate_coco_metrics(gt_data, sam_pred_data, iou_type="bbox", max_dets=350)

        # Evaluate YOLO+SegRefiner
        gt_data, segrefiner_pred_data = generate_coco_annotations_from_multi_instance_masks_16bit(
            gt_mask_path, segrefiner_mask_path, img_path
        )
        # print("DEBUG: segrefiner_pred_data =", segrefiner_pred_data)
        segrefiner_segm_metrics = evaluate_coco_metrics(gt_data, segrefiner_pred_data, iou_type="segm", max_dets=350)
        segrefiner_bbox_metrics = evaluate_coco_metrics(gt_data, segrefiner_pred_data, iou_type="bbox", max_dets=350)

        # Safely retrieve metric dicts
        def safe_get(m, k):
            return m[k] if (m is not None and k in m) else None

        results.append({
            "image_name": image_name,

            # YOLO bounding boxes
            "yolo_box_AP@50": safe_get(yolo_bbox_metrics, "AP@50"),
            "yolo_box_AP@75": safe_get(yolo_bbox_metrics, "AP@75"),
            "yolo_box_AP@95": safe_get(yolo_bbox_metrics, "AP@95"),
            "yolo_box_AP@50:95": safe_get(yolo_bbox_metrics, "AP@50:95"),
            "yolo_box_AR@50": safe_get(yolo_bbox_metrics, "AR@50"),
            "yolo_box_AR@75": safe_get(yolo_bbox_metrics, "AR@75"),
            "yolo_box_AR@95": safe_get(yolo_bbox_metrics, "AR@95"),
            "yolo_box_AR@50:95": safe_get(yolo_bbox_metrics, "AR@50:95"),

            # YOLO masks
            "yolo_mask_AP@50": safe_get(yolo_segm_metrics, "AP@50"),
            "yolo_mask_AP@75": safe_get(yolo_segm_metrics, "AP@75"),
            "yolo_mask_AP@95": safe_get(yolo_segm_metrics, "AP@95"),
            "yolo_mask_AP@50:95": safe_get(yolo_segm_metrics, "AP@50:95"),
            "yolo_mask_AR@50": safe_get(yolo_segm_metrics, "AR@50"),
            "yolo_mask_AR@75": safe_get(yolo_segm_metrics, "AR@75"),
            "yolo_mask_AR@95": safe_get(yolo_segm_metrics, "AR@95"),
            "yolo_mask_AR@50:95": safe_get(yolo_segm_metrics, "AR@50:95"),

            # SAM bounding boxes
            "sam_box_AP@50": safe_get(sam_bbox_metrics, "AP@50"),
            "sam_box_AP@75": safe_get(sam_bbox_metrics, "AP@75"),
            "sam_box_AP@95": safe_get(sam_bbox_metrics, "AP@95"),
            "sam_box_AP@50:95": safe_get(sam_bbox_metrics, "AP@50:95"),
            "sam_box_AR@50": safe_get(sam_bbox_metrics, "AR@50"),
            "sam_box_AR@75": safe_get(sam_bbox_metrics, "AR@75"),
            "sam_box_AR@95": safe_get(sam_bbox_metrics, "AR@95"),
            "sam_box_AR@50:95": safe_get(sam_bbox_metrics, "AR@50:95"),

            # SAM masks
            "sam_mask_AP@50": safe_get(sam_segm_metrics, "AP@50"),
            "sam_mask_AP@75": safe_get(sam_segm_metrics, "AP@75"),
            "sam_mask_AP@95": safe_get(sam_segm_metrics, "AP@95"),
            "sam_mask_AP@50:95": safe_get(sam_segm_metrics, "AP@50:95"),
            "sam_mask_AR@50": safe_get(sam_segm_metrics, "AR@50"),
            "sam_mask_AR@75": safe_get(sam_segm_metrics, "AR@75"),
            "sam_mask_AR@95": safe_get(sam_segm_metrics, "AR@95"),
            "sam_mask_AR@50:95": safe_get(sam_segm_metrics, "AR@50:95"),

            # SegRefiner bounding boxes
            "segrefiner_box_AP@50": safe_get(segrefiner_bbox_metrics, "AP@50"),
            "segrefiner_box_AP@75": safe_get(segrefiner_bbox_metrics, "AP@75"),
            "segrefiner_box_AP@95": safe_get(segrefiner_bbox_metrics, "AP@95"),
            "segrefiner_box_AP@50:95": safe_get(segrefiner_bbox_metrics, "AP@50:95"),
            "segrefiner_box_AR@50": safe_get(segrefiner_bbox_metrics, "AR@50"),
            "segrefiner_box_AR@75": safe_get(segrefiner_bbox_metrics, "AR@75"),
            "segrefiner_box_AR@95": safe_get(segrefiner_bbox_metrics, "AR@95"),
            "segrefiner_box_AR@50:95": safe_get(segrefiner_bbox_metrics, "AR@50:95"),

            # SegRefiner masks
            "segrefiner_mask_AP@50": safe_get(segrefiner_segm_metrics, "AP@50"),
            "segrefiner_mask_AP@75": safe_get(segrefiner_segm_metrics, "AP@75"),
            "segrefiner_mask_AP@95": safe_get(segrefiner_segm_metrics, "AP@95"),
            "segrefiner_mask_AP@50:95": safe_get(segrefiner_segm_metrics, "AP@50:95"),
            "segrefiner_mask_AR@50": safe_get(segrefiner_segm_metrics, "AR@50"),
            "segrefiner_mask_AR@75": safe_get(segrefiner_segm_metrics, "AR@75"),
            "segrefiner_mask_AR@95": safe_get(segrefiner_segm_metrics, "AR@95"),
            "segrefiner_mask_AR@50:95": safe_get(segrefiner_segm_metrics, "AR@50:95"),
        })

    # -------------------------------------------------------
    # 11) Write results to CSV
    # -------------------------------------------------------
    csv_filename = os.path.join(output_dir, "evaluation_metrics.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nMetrics saved to {csv_filename}")
    else:
        print("\nNo results to save. Did not find any valid images or masks.")

if __name__ == "__main__":
    main()
