import os
import cv2
import glob
import csv
from PIL import Image

# Local imports from your modules
from models import load_trained_model, get_inference_predictions
from sam_helper import load_sam_model, run_sam_inference, combine_masks
from tools import (
    visualize_segmentations,
    generate_coco_annotations_from_multi_instance_masks,
    evaluate_coco_metrics
)

def main():
    # Directories and Paths
    output_dir = "outputImages"
    os.makedirs(output_dir, exist_ok=True)

    # Folders containing images and ground-truth masks
    image_dir = "/home/sprice/RQ/demo.v7i.yolov8/test/images/"
    gt_mask_dir = "/home/sprice/ICCV25/SegRefinerV2/test_masks/"

    # -------------------------------------------------------
    # 1) Load the model of your choice (YOLO / Mask R-CNN / MobileNetV3)
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
    # Prepare list to store our per-image metrics
    # We will store 8 metrics for each of 4 methods => 32 columns
    #  Plus an 'image_name' column = 33 columns total, for example.
    # -------------------------------------------------------
    results = []

    # -------------------------------------------------------
    # 3) Process all .jpg images in the directory
    # -------------------------------------------------------
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

    for img_path in image_paths:
        # Derive image name (e.g., "TruForm174-2_00-37_500X16_png.rf.17ddf81ab4dd63c5ce6f66654a48c5b4.jpg")
        image_name = os.path.basename(img_path)

        # Corresponding ground-truth mask file
        # Replace .jpg with _mask.png
        gt_mask_name = image_name.replace(".jpg", "_mask.png")
        gt_mask_path = os.path.join(gt_mask_dir, gt_mask_name)

        # Safety check in case the mask doesn't exist
        if not os.path.exists(gt_mask_path):
            print(f"[WARNING] GT mask not found for {image_name}, skipping...")
            continue

        # Load images via OpenCV and PIL
        loop_image = cv2.imread(img_path)
        pil_image = Image.open(img_path)

        # -------------------------------------------------------
        # 4) Model Inference (YOLO / mask-rcnn / etc.)
        # -------------------------------------------------------
        listOfPolygons, listOfBoxes, listOfMasks = get_inference_predictions(
            model,
            chosen_model_type,
            img_path,
            device=device
        )

        # -------------------------------------------------------
        # 5) Run SAM refinement
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

        # -------------------------------------------------------
        # 6) Combine SAM masks into one mask
        # -------------------------------------------------------
        combined_sam_mask_path = os.path.join(
            output_dir, f"{image_name.replace('.jpg', '_sam_combined.png')}"
        )
        combine_masks(sam_masks_list, combined_sam_mask_path)

        # -------------------------------------------------------
        # 7) Combine original model masks into one mask
        # -------------------------------------------------------
        combined_model_mask_path = os.path.join(
            output_dir, f"{image_name.replace('.jpg', '_yolo_combined.png')}"
        )
        combine_masks(listOfMasks, combined_model_mask_path)

        # -------------------------------------------------------
        # 8) Visualization (optional)
        #    Comment out if not needed or to speed things up.
        # -------------------------------------------------------
        visualize_segmentations(
            image_path=img_path,
            mask_path=combined_sam_mask_path,
            output_dir=output_dir,
            title="Combined SAM Segmentation"
        )
        visualize_segmentations(
            image_path=img_path,
            mask_path=combined_model_mask_path,
            output_dir=output_dir,
            title=f"Combined {chosen_model_type.upper()} Segmentation"
        )

        # -------------------------------------------------------
        # 9) Evaluation
        #    We'll get two sets of predictions:
        #      - YOLO predictions
        #      - SAM predictions
        # -------------------------------------------------------
        # Evaluate YOLO segmentation
        gt_data, model_pred_data = generate_coco_annotations_from_multi_instance_masks(
            gt_mask_path,
            combined_model_mask_path,
            img_path
        )
        yolo_segm_metrics = evaluate_coco_metrics(gt_data, model_pred_data, iou_type="segm", max_dets=350)

        # Evaluate YOLO bounding boxes
        yolo_bbox_metrics = evaluate_coco_metrics(gt_data, model_pred_data, iou_type="bbox", max_dets=350)

        # Evaluate SAM segmentation
        gt_data, sam_pred_data = generate_coco_annotations_from_multi_instance_masks(
            gt_mask_path,
            combined_sam_mask_path,
            img_path
        )
        sam_segm_metrics = evaluate_coco_metrics(gt_data, sam_pred_data, iou_type="segm", max_dets=350)

        # Evaluate SAM bounding boxes
        sam_bbox_metrics = evaluate_coco_metrics(gt_data, sam_pred_data, iou_type="bbox", max_dets=350)

        # -------------------------------------------------------
        # Assume each of these dictionaries returns something like:
        # {
        #   "AP@50": 0.901,
        #   "AP@75": 0.875,
        #   "AP@95": 0.251,
        #   "AP@50:95": 0.676,
        #   "AR@50": 0.904,
        #   "AR@75": 0.878,
        #   "AR@95": 0.374,
        #   "AR@50:95": 0.719
        # }
        # -------------------------------------------------------

        # Store the results for this image
        results.append({
            "image_name": image_name,

            "yolo_box_AP@50": yolo_bbox_metrics["AP@50"],
            "yolo_box_AP@75": yolo_bbox_metrics["AP@75"],
            "yolo_box_AP@95": yolo_bbox_metrics["AP@95"],
            "yolo_box_AP@50:95": yolo_bbox_metrics["AP@50:95"],
            "yolo_box_AR@50": yolo_bbox_metrics["AR@50"],
            "yolo_box_AR@75": yolo_bbox_metrics["AR@75"],
            "yolo_box_AR@95": yolo_bbox_metrics["AR@95"],
            "yolo_box_AR@50:95": yolo_bbox_metrics["AR@50:95"],

            "yolo_mask_AP@50": yolo_segm_metrics["AP@50"],
            "yolo_mask_AP@75": yolo_segm_metrics["AP@75"],
            "yolo_mask_AP@95": yolo_segm_metrics["AP@95"],
            "yolo_mask_AP@50:95": yolo_segm_metrics["AP@50:95"],
            "yolo_mask_AR@50": yolo_segm_metrics["AR@50"],
            "yolo_mask_AR@75": yolo_segm_metrics["AR@75"],
            "yolo_mask_AR@95": yolo_segm_metrics["AR@95"],
            "yolo_mask_AR@50:95": yolo_segm_metrics["AR@50:95"],

            "sam_box_AP@50": sam_bbox_metrics["AP@50"],
            "sam_box_AP@75": sam_bbox_metrics["AP@75"],
            "sam_box_AP@95": sam_bbox_metrics["AP@95"],
            "sam_box_AP@50:95": sam_bbox_metrics["AP@50:95"],
            "sam_box_AR@50": sam_bbox_metrics["AR@50"],
            "sam_box_AR@75": sam_bbox_metrics["AR@75"],
            "sam_box_AR@95": sam_bbox_metrics["AR@95"],
            "sam_box_AR@50:95": sam_bbox_metrics["AR@50:95"],

            "sam_mask_AP@50": sam_segm_metrics["AP@50"],
            "sam_mask_AP@75": sam_segm_metrics["AP@75"],
            "sam_mask_AP@95": sam_segm_metrics["AP@95"],
            "sam_mask_AP@50:95": sam_segm_metrics["AP@50:95"],
            "sam_mask_AR@50": sam_segm_metrics["AR@50"],
            "sam_mask_AR@75": sam_segm_metrics["AR@75"],
            "sam_mask_AR@95": sam_segm_metrics["AR@95"],
            "sam_mask_AR@50:95": sam_segm_metrics["AR@50:95"],
        })

    # -------------------------------------------------------
    # 10) Write results to CSV
    # -------------------------------------------------------
    csv_filename = os.path.join(output_dir, "evaluation_metrics.csv")
    if results:
        # Extract fieldnames from the first result
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
