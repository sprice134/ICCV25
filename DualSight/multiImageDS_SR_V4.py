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

# Tools or helper functions
from tools import (
    visualize_segmentations,
    generate_coco_annotations_from_multi_instance_masks_16bit,
    evaluate_coco_metrics,
    combine_masks_16bit,
    compute_pixel_precision_recall
)

############################################################################
# 1) Main pipeline
############################################################################

def main():
    # --------------------------
    # Directories and Paths
    # --------------------------
    output_dir = "outputImages"
    os.makedirs(output_dir, exist_ok=True)

    image_dir = "/home/sprice/RQ/demo.v7i.yolov8/test/images/"
    gt_mask_dir = "/home/sprice/ICCV25/SegRefinerV2/output_masks_16bit/"

    # 1) Define the models you want to compare
    model_configs = [
        {
            "model_type": "yolo",
            "model_name": "yolov8n-seg",
            "model_path": "/home/sprice/ICCV25/modelWeights/yolov8n-seg.pt"
        },
        {
            "model_type": "yolo",
            "model_name": "yolov8x-seg",
            "model_path": "/home/sprice/ICCV25/modelWeights/yolov8x-seg.pt"
        },
        {
            "model_type": "maskrcnn",
            "model_name": "maskrcnn",
            "model_path": "/home/sprice/ICCV25/modelWeights/final_mask_rcnn_model.pth"
        }
        # Add more models if needed...
    ]


    # 2) Load SAM (only once, if you want to do YOLO+SAM on all YOLO predictions)
    device = "cuda"
    sam_model_type = "vit_l"
    sam_checkpoint = f"/home/sprice/ICCV25/modelWeights/sam_{sam_model_type}.pth"
    sam_predictor = load_sam_model(
        sam_checkpoint=sam_checkpoint,
        model_type=sam_model_type,
        device=device
    )
    
    # 3) Load SegRefiner (two models: small + large)
    # SMALL:
    segrefiner_config_path_sm = "/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py"
    segrefiner_checkpoint_path_sm = "/home/sprice/ICCV25/modelWeights/segrefiner_lr_latest.pth"
    segrefiner_model_small, segrefiner_cfg_small = load_segrefiner_model(
        segrefiner_config_path_sm,
        segrefiner_checkpoint_path_sm,
        device=device
    )
    
    # LARGE:
    segrefiner_config_path_lg = "/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py"
    segrefiner_checkpoint_path_lg = "/home/sprice/ICCV25/modelWeights/segrefiner_hr_latest.pth"
    segrefiner_model_large, segrefiner_cfg_large = load_segrefiner_model(
        segrefiner_config_path_lg,
        segrefiner_checkpoint_path_lg,
        device=device
    )

    # Prepare list to store per-image metrics across all models
    results = []

    # 4) Get all .jpg images
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not image_paths:
        print("[ERROR] No .jpg images found.")
        return

    # 5) Loop over each model config
    for config in model_configs:
        model_type = config["model_type"]
        model_name = config["model_name"]
        model_path = config["model_path"]

        print(f"\n--- Loading {model_name} ({model_type}) from {model_path} ---")
        model = load_trained_model(model_type, model_path, device=device)

        # 6) For each image, do inference + optional SAM + optional SegRefiner
        for img_path in image_paths:
            image_name = os.path.basename(img_path)

            gt_mask_name = image_name.replace(".jpg", "_mask.png")
            gt_mask_path = os.path.join(gt_mask_dir, gt_mask_name)

            if not os.path.exists(gt_mask_path):
                print(f"[WARNING] GT mask not found for {image_name}, skipping...")
                continue

            # Load the image in OpenCV for certain steps
            loop_image = cv2.imread(img_path)
            pil_image = Image.open(img_path)
            if loop_image is None:
                print(f"[ERROR] Could not read image {img_path}")
                continue

            # -------------------------------------------------------
            # 5-A) Model-only Inference => listOfMasks
            # -------------------------------------------------------
            listOfPolygons, listOfBoxes, listOfMasks = get_inference_predictions(
                model=model,
                model_type=model_type,
                image_path=img_path,
                device=device
            )
            # Convert to binary: [H,W] with 0 or 1
            bin_masks = []
            for mask in listOfMasks:
                # Some YOLO flows might give [0..255], ensure threshold
                bin_mask = (mask > 127).astype(np.uint8)
                bin_masks.append(bin_mask)

            # Combine into 16-bit (base)
            base_16bit_path = os.path.join(
                output_dir, f"{image_name.replace('.jpg', f'_{model_name}_16bit.png')}"
            )
            combine_masks_16bit(bin_masks, base_16bit_path)

            # -------------------------------------------------------
            # 5-B) YOLO+SAM => refine with SAM
            # -------------------------------------------------------
            sam_masks_list = run_sam_inference(
                predictor=sam_predictor,
                loop_image=loop_image,
                listOfPolygons=listOfPolygons,
                listOfBoxes=listOfBoxes,
                listOfMasks=bin_masks,
                image_width=pil_image.width,
                image_height=pil_image.height,
                num_points=4,
                dropout_percentage=0,
                ignore_border_percentage=5,
                algorithm="Voronoi",
                use_mask_input=False,
                box_expansion_rate=0.0
            )
            sam_16bit_path = os.path.join(
                output_dir, f"{image_name.replace('.jpg', f'_{model_name}_SAM_16bit.png')}"
            )
            combine_masks_16bit(sam_masks_list, sam_16bit_path)

            # -------------------------------------------------------
            # 5-C1) Model+SegRefiner (SMALL)
            # -------------------------------------------------------
            segrefiner_masks_list_small = run_segrefiner_inference(
                segrefiner_model_small,
                segrefiner_cfg_small,
                list_of_binary_masks=bin_masks,
                loop_image_bgr=loop_image,
                device=device
            )
            segrefiner_small_16bit_path = os.path.join(
                output_dir, f"{image_name.replace('.jpg', f'_{model_name}_SegRefinerSmall_16bit.png')}"
            )
            combine_masks_16bit(segrefiner_masks_list_small, segrefiner_small_16bit_path)

            # -------------------------------------------------------
            # 5-C2) Model+SegRefiner (LARGE)
            # -------------------------------------------------------
            segrefiner_masks_list_large = run_segrefiner_inference(
                segrefiner_model_large,
                segrefiner_cfg_large,
                list_of_binary_masks=bin_masks,
                loop_image_bgr=loop_image,
                device=device
            )
            segrefiner_large_16bit_path = os.path.join(
                output_dir, f"{image_name.replace('.jpg', f'_{model_name}_SegRefinerLarge_16bit.png')}"
            )
            combine_masks_16bit(segrefiner_masks_list_large, segrefiner_large_16bit_path)

            # -------------------------------------------------------
            # Optional Visualization
            # -------------------------------------------------------
            try:
                visualize_segmentations(
                    image_path=img_path,
                    mask_path=base_16bit_path,
                    output_dir=output_dir,
                    title=f"{model_name} (base)"
                )
                visualize_segmentations(
                    image_path=img_path,
                    mask_path=sam_16bit_path,
                    output_dir=output_dir,
                    title=f"{model_name} + SAM"
                )
                visualize_segmentations(
                    image_path=img_path,
                    mask_path=segrefiner_small_16bit_path,
                    output_dir=output_dir,
                    title=f"{model_name} + SegRefiner (Small)"
                )
                visualize_segmentations(
                    image_path=img_path,
                    mask_path=segrefiner_large_16bit_path,
                    output_dir=output_dir,
                    title=f"{model_name} + SegRefiner (Large)"
                )
            except Exception as e:
                print(f"[WARNING] Visualization error for {image_name}: {e}")

            # -------------------------------------------------------
            # 5-D) Evaluate each approach
            # -------------------------------------------------------
            # Utility for safe metric retrieval
            def safe_get(m, k):
                return m[k] if (m is not None and k in m) else None


            
            
            
            

            # Evaluate (base) => model-only
            gt_data, pred_data_base = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, base_16bit_path, img_path
            )
            base_segm_metrics = evaluate_coco_metrics(gt_data, pred_data_base, iou_type="segm", max_dets=350)
            base_bbox_metrics = evaluate_coco_metrics(gt_data, pred_data_base, iou_type="bbox", max_dets=350)
            base_precision, base_recall = compute_pixel_precision_recall(gt_mask_path, base_16bit_path)

            # Evaluate (model+SAM)
            gt_data, pred_data_sam = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, sam_16bit_path, img_path
            )
            sam_segm_metrics = evaluate_coco_metrics(gt_data, pred_data_sam, iou_type="segm", max_dets=350)
            sam_bbox_metrics = evaluate_coco_metrics(gt_data, pred_data_sam, iou_type="bbox", max_dets=350)
            sam_precision, sam_recall = compute_pixel_precision_recall(gt_mask_path, sam_16bit_path)

            # Evaluate (model+SegRefiner Small)
            gt_data, pred_data_seg_sm = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, segrefiner_small_16bit_path, img_path
            )
            segrefiner_small_segm_metrics = evaluate_coco_metrics(gt_data, pred_data_seg_sm, iou_type="segm", max_dets=350)
            segrefiner_small_bbox_metrics = evaluate_coco_metrics(gt_data, pred_data_seg_sm, iou_type="bbox", max_dets=350)
            seg_small_precision, seg_small_recall = compute_pixel_precision_recall(gt_mask_path, segrefiner_small_16bit_path)

            # Evaluate (model+SegRefiner Large)
            gt_data, pred_data_seg_lg = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, segrefiner_large_16bit_path, img_path
            )
            segrefiner_large_segm_metrics = evaluate_coco_metrics(gt_data, pred_data_seg_lg, iou_type="segm", max_dets=350)
            segrefiner_large_bbox_metrics = evaluate_coco_metrics(gt_data, pred_data_seg_lg, iou_type="bbox", max_dets=350)
            seg_large_precision, seg_large_recall = compute_pixel_precision_recall(gt_mask_path, segrefiner_large_16bit_path)

            # Build a single results dict for this image + model
            results.append({
                "model_name": model_name,
                "image_name": image_name,

                # Base bounding boxes
                "base_box_AP@50": safe_get(base_bbox_metrics, "AP@50"),
                "base_box_AP@75": safe_get(base_bbox_metrics, "AP@75"),
                "base_box_AP@95": safe_get(base_bbox_metrics, "AP@95"),
                "base_box_AP@50:95": safe_get(base_bbox_metrics, "AP@50:95"),
                "base_box_AR@50": safe_get(base_bbox_metrics, "AR@50"),
                "base_box_AR@75": safe_get(base_bbox_metrics, "AR@75"),
                "base_box_AR@95": safe_get(base_bbox_metrics, "AR@95"),
                "base_box_AR@50:95": safe_get(base_bbox_metrics, "AR@50:95"),
            
                # Base masks
                "base_mask_AP@50": safe_get(base_segm_metrics, "AP@50"),
                "base_mask_AP@75": safe_get(base_segm_metrics, "AP@75"),
                "base_mask_AP@95": safe_get(base_segm_metrics, "AP@95"),
                "base_mask_AP@50:95": safe_get(base_segm_metrics, "AP@50:95"),
                "base_mask_AR@50": safe_get(base_segm_metrics, "AR@50"),
                "base_mask_AR@75": safe_get(base_segm_metrics, "AR@75"),
                "base_mask_AR@95": safe_get(base_segm_metrics, "AR@95"),
                "base_mask_AR@50:95": safe_get(base_segm_metrics, "AR@50:95"),

                # Base Aggregate Metrics
                "base_precision": base_precision,
                "base_recall": base_recall,

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

                # Aggregate metrics for SAM
                "sam_precision": sam_precision,
                "sam_recall": sam_recall,

                # SegRefiner SMALL bounding boxes
                "segrefiner_small_box_AP@50": safe_get(segrefiner_small_bbox_metrics, "AP@50"),
                "segrefiner_small_box_AP@75": safe_get(segrefiner_small_bbox_metrics, "AP@75"),
                "segrefiner_small_box_AP@95": safe_get(segrefiner_small_bbox_metrics, "AP@95"),
                "segrefiner_small_box_AP@50:95": safe_get(segrefiner_small_bbox_metrics, "AP@50:95"),
                "segrefiner_small_box_AR@50": safe_get(segrefiner_small_bbox_metrics, "AR@50"),
                "segrefiner_small_box_AR@75": safe_get(segrefiner_small_bbox_metrics, "AR@75"),
                "segrefiner_small_box_AR@95": safe_get(segrefiner_small_bbox_metrics, "AR@95"),
                "segrefiner_small_box_AR@50:95": safe_get(segrefiner_small_bbox_metrics, "AR@50:95"),

                # SegRefiner SMALL masks
                "segrefiner_small_mask_AP@50": safe_get(segrefiner_small_segm_metrics, "AP@50"),
                "segrefiner_small_mask_AP@75": safe_get(segrefiner_small_segm_metrics, "AP@75"),
                "segrefiner_small_mask_AP@95": safe_get(segrefiner_small_segm_metrics, "AP@95"),
                "segrefiner_small_mask_AP@50:95": safe_get(segrefiner_small_segm_metrics, "AP@50:95"),
                "segrefiner_small_mask_AR@50": safe_get(segrefiner_small_segm_metrics, "AR@50"),
                "segrefiner_small_mask_AR@75": safe_get(segrefiner_small_segm_metrics, "AR@75"),
                "segrefiner_small_mask_AR@95": safe_get(segrefiner_small_segm_metrics, "AR@95"),
                "segrefiner_small_mask_AR@50:95": safe_get(segrefiner_small_segm_metrics, "AR@50:95"),

                # Aggregate metrics for SegRefiner Small
                "segrefiner_small_precision": seg_small_precision,
                "segrefiner_small_recall": seg_small_recall,

                # SegRefiner LARGE bounding boxes
                "segrefiner_large_box_AP@50": safe_get(segrefiner_large_bbox_metrics, "AP@50"),
                "segrefiner_large_box_AP@75": safe_get(segrefiner_large_bbox_metrics, "AP@75"),
                "segrefiner_large_box_AP@95": safe_get(segrefiner_large_bbox_metrics, "AP@95"),
                "segrefiner_large_box_AP@50:95": safe_get(segrefiner_large_bbox_metrics, "AP@50:95"),
                "segrefiner_large_box_AR@50": safe_get(segrefiner_large_bbox_metrics, "AR@50"),
                "segrefiner_large_box_AR@75": safe_get(segrefiner_large_bbox_metrics, "AR@75"),
                "segrefiner_large_box_AR@95": safe_get(segrefiner_large_bbox_metrics, "AR@95"),
                "segrefiner_large_box_AR@50:95": safe_get(segrefiner_large_bbox_metrics, "AR@50:95"),

                # SegRefiner LARGE masks
                "segrefiner_large_mask_AP@50": safe_get(segrefiner_large_segm_metrics, "AP@50"),
                "segrefiner_large_mask_AP@75": safe_get(segrefiner_large_segm_metrics, "AP@75"),
                "segrefiner_large_mask_AP@95": safe_get(segrefiner_large_segm_metrics, "AP@95"),
                "segrefiner_large_mask_AP@50:95": safe_get(segrefiner_large_segm_metrics, "AP@50:95"),
                "segrefiner_large_mask_AR@50": safe_get(segrefiner_large_segm_metrics, "AR@50"),
                "segrefiner_large_mask_AR@75": safe_get(segrefiner_large_segm_metrics, "AR@75"),
                "segrefiner_large_mask_AR@95": safe_get(segrefiner_large_segm_metrics, "AR@95"),
                "segrefiner_large_mask_AR@50:95": safe_get(segrefiner_large_segm_metrics, "AR@50:95"),

                # Aggregate metrics for SegRefiner Large
                "segrefiner_large_precision": seg_large_precision,
                "segrefiner_large_recall": seg_large_recall,
            })

    # -------------------------------------------------------
    # 7) Write results to CSV
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
