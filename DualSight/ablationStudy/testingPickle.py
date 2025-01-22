import os
import cv2
import glob
import csv
import pickle
import numpy as np
from PIL import Image

# ---------------------------
# Local or custom modules
# ---------------------------
import sys
sys.path.append('../')
from sam_helper import load_sam_model, run_sam_inference
from segrefiner_helper import load_segrefiner_model, run_segrefiner_inference

# Tools or helper functions
from tools import (
    visualize_segmentations,
    generate_coco_annotations_from_multi_instance_masks_16bit,
    evaluate_coco_metrics,
    combine_masks_16bit
)

############################################################################
# 1) Main pipeline using pre-saved inference files
############################################################################

def main():
    # --------------------------
    # Directories and Paths
    # --------------------------
    output_dir = "outputImages"
    os.makedirs(output_dir, exist_ok=True)

    image_dir = "/home/sprice/RQ/demo.v7i.yolov8/test/images/"
    gt_mask_dir = "/home/sprice/ICCV25/SegRefinerV2/output_masks_16bit/"
    inference_dir = "inference_outputs"  # Directory where inference pickle files are stored

    # 1) Define the models you want to compare and corresponding pickle filenames
    model_configs = [
        {
            "model_type": "yolo",
            "model_name": "yolov8n-seg",
            "inference_file": os.path.join(inference_dir, "yolov8n_inference.pkl")
        },
        {
            "model_type": "yolo",
            "model_name": "yolov8x-seg",
            "inference_file": os.path.join(inference_dir, "yolov8x_inference.pkl")
        },
        # Add more models with their respective inference pickle paths if needed...
    ]

    # 2) Load SAM (only once)
    device = "cuda"
    sam_model_type = "vit_l"
    sam_checkpoint = f"/home/sprice/ICCV25/modelWeights/sam_{sam_model_type}.pth"
    sam_predictor = load_sam_model(
        sam_checkpoint=sam_checkpoint,
        model_type=sam_model_type,
        device=device
    )
    
    # 3) Load SegRefiner (only once)
    segrefiner_config_path = f"/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py"
    segrefiner_checkpoint_path = f"/home/sprice/ICCV25/modelWeights/segrefiner_lr_latest.pth"
    segrefiner_model, segrefiner_cfg = load_segrefiner_model(
        segrefiner_config_path,
        segrefiner_checkpoint_path,
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
        inference_file = config["inference_file"]

        print(f"\n--- Loading pre-saved inference for {model_name} from {inference_file} ---")
        if not os.path.exists(inference_file):
            print(f"[ERROR] Inference file {inference_file} not found. Skipping {model_name}.")
            continue

        with open(inference_file, "rb") as pf:
            model_inference_data = pickle.load(pf)

        # 6) For each image, process inference results + optional SAM + SegRefiner
        for img_path in image_paths:
            image_name = os.path.basename(img_path)

            gt_mask_name = image_name.replace(".jpg", "_mask.png")
            gt_mask_path = os.path.join(gt_mask_dir, gt_mask_name)

            if not os.path.exists(gt_mask_path):
                print(f"[WARNING] GT mask not found for {image_name}, skipping...")
                continue

            loop_image = cv2.imread(img_path)
            pil_image = Image.open(img_path)
            if loop_image is None:
                print(f"[ERROR] Could not read image {img_path}")
                continue

            # Retrieve pre-saved inference results for the current image
            if image_name not in model_inference_data:
                print(f"[WARNING] No inference data for {image_name} in {model_name}, skipping...")
                continue

            image_results = model_inference_data[image_name]
            listOfPolygons = image_results["polygons"]
            listOfBoxes = image_results["boxes"]
            listOfMasks = image_results["masks"]

            # Convert masks to binary
            bin_masks = []
            for mask in listOfMasks:
                bin_mask = (mask > 127).astype(np.uint8)
                bin_masks.append(bin_mask)

            # Combine into 16-bit (base)
            base_16bit_path = os.path.join(
                output_dir, f"{image_name.replace('.jpg', f'_{model_name}_16bit.png')}"
            )
            combine_masks_16bit(bin_masks, base_16bit_path)

            # YOLO+SAM (Model+SAM) step using pre-saved predictions
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

            # Model+SegRefiner step
            segrefiner_masks_list = run_segrefiner_inference(
                segrefiner_model,
                segrefiner_cfg,
                list_of_binary_masks=bin_masks,
                loop_image_bgr=loop_image,
                device=device
            )
            segrefiner_16bit_path = os.path.join(
                output_dir, f"{image_name.replace('.jpg', f'_{model_name}_SegRefiner_16bit.png')}"
            )
            combine_masks_16bit(segrefiner_masks_list, segrefiner_16bit_path)

            # Optional Visualization
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
                    mask_path=segrefiner_16bit_path,
                    output_dir=output_dir,
                    title=f"{model_name} + SegRefiner"
                )
            except Exception as e:
                print(f"[WARNING] Visualization error for {image_name}: {e}")

            # 5-D) Evaluate each approach
            # Evaluate (base)
            gt_data, pred_data = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, base_16bit_path, img_path
            )
            base_segm_metrics = evaluate_coco_metrics(gt_data, pred_data, iou_type="segm", max_dets=350)
            base_bbox_metrics = evaluate_coco_metrics(gt_data, pred_data, iou_type="bbox", max_dets=350)

            # Evaluate (model+SAM)
            gt_data, sam_pred_data = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, sam_16bit_path, img_path
            )
            sam_segm_metrics = evaluate_coco_metrics(gt_data, sam_pred_data, iou_type="segm", max_dets=350)
            sam_bbox_metrics = evaluate_coco_metrics(gt_data, sam_pred_data, iou_type="bbox", max_dets=350)

            # Evaluate (model+SegRefiner)
            gt_data, segrefiner_pred_data = generate_coco_annotations_from_multi_instance_masks_16bit(
                gt_mask_path, segrefiner_16bit_path, img_path
            )
            segrefiner_segm_metrics = evaluate_coco_metrics(gt_data, segrefiner_pred_data, iou_type="segm", max_dets=350)
            segrefiner_bbox_metrics = evaluate_coco_metrics(gt_data, segrefiner_pred_data, iou_type="bbox", max_dets=350)

            def safe_get(m, k):
                return m[k] if (m is not None and k in m) else None

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
    # 7) Write results to CSV
    # -------------------------------------------------------
    csv_filename = os.path.join(output_dir, "evaluation_metrics_pickle1.csv")
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
