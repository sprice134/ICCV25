import os
import cv2
from PIL import Image
import numpy as np

# Local imports from our new modules
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
    # 3) Input image
    # -------------------------------------------------------
    image_path = (
        "/home/sprice/RQ/demo.v7i.yolov8/test/images/"
        "Cu-Ni-Powder_250x_10_SE_png.rf.cd93ec4589ad8f4e412cb1ec0e805016.jpg"
    )
    gt_mask_path = (
        "/home/sprice/ICCV25/SegRefinerV2/test_masks/"
        "Cu-Ni-Powder_250x_10_SE_png.rf.cd93ec4589ad8f4e412cb1ec0e805016_mask.png"
    )
    # image_path = (
    #     "/home/sprice/ICCV25/demo.v7i.coco/test/"
    #     "S02_02_SE1_300X18_png.rf.1a16e8c5f4e008cb2fc48c98b35778fb.jpg"
    # )
    
    loop_image = cv2.imread(image_path)
    pil_image = Image.open(image_path)

    # -------------------------------------------------------
    # 4) Model Inference 
    # -------------------------------------------------------
    listOfPolygons, listOfBoxes, listOfMasks = get_inference_predictions(
        model,
        chosen_model_type,
        image_path,
        device=device
    )

    # -------------------------------------------------------
    # 5) Run SAM refinement
    #    - You can toggle use_mask_input = True to pass the mask to SAM
    #    - You can set box_expansion_rate > 0.0 to expand bounding boxes
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

    # ----- NEW: Save Each Refined SAM Mask as Binary ----- 
    # Create a subdirectory for individual SAM masks
    individual_sam_dir = os.path.join(output_dir, "individual_sam_masks")
    os.makedirs(individual_sam_dir, exist_ok=True)

    # Iterate over SAM masks and save each as a binary image
    for idx, mask in enumerate(sam_masks_list, start=1):
        # Convert mask to binary (0 or 255) - assuming mask is binary already
        binary_mask = (mask > 0).astype(np.uint8) * 255
        mask_path = os.path.join(individual_sam_dir, f"sam_mask_{idx:03d}.png")
        cv2.imwrite(mask_path, binary_mask)
    # -----------------------------------------------------

    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise FileNotFoundError(f"Ground truth mask not found: {gt_mask_path}")

    # Directory to store individual GT masks
    individual_gt_dir = os.path.join(output_dir, "individual_gt_masks")
    os.makedirs(individual_gt_dir, exist_ok=True)

    # Extract unique object IDs (excluding background assumed to be 0)
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]

    # Iterate over each object ID, create a binary mask, and save
    for obj_id in gt_ids:
        binary_mask = (gt_mask == obj_id).astype(np.uint8) * 255
        mask_filename = f"gt_mask_{int(obj_id):03d}.png"
        mask_path = os.path.join(individual_gt_dir, mask_filename)
        cv2.imwrite(mask_path, binary_mask)
    # -----------------------------------------------------

    # -------------------------------------------------------
    # 6) Combine SAM masks into one mask
    # -------------------------------------------------------
    combined_sam_mask_path = os.path.join(output_dir, 'combined_sam_mask.png')
    combine_masks(sam_masks_list, combined_sam_mask_path)

    # -------------------------------------------------------
    # 7) Combine original model masks into one mask
    # -------------------------------------------------------
    combined_model_mask_path = os.path.join(output_dir, 'combined_model_mask.png')
    combine_masks(listOfMasks, combined_model_mask_path)

    # -------------------------------------------------------
    # 8) Visualization
    # -------------------------------------------------------
    visualize_segmentations(
        image_path=image_path,
        mask_path=combined_sam_mask_path,
        output_dir=output_dir,
        title="Combined SAM Segmentation"
    )
    visualize_segmentations(
        image_path=image_path,
        mask_path=combined_model_mask_path,
        output_dir=output_dir,
        title=f"Combined {chosen_model_type.upper()} Segmentation"
    )

    # -------------------------------------------------------
    # 9) Evaluation with your original COCO logic
    # -------------------------------------------------------

    print(f"Evaluating {chosen_model_type.upper()} Segmentation Metrics:")
    gt_data, model_pred_data = generate_coco_annotations_from_multi_instance_masks(
        gt_mask_path, 
        combined_model_mask_path,
        image_path
    )
    evaluate_coco_metrics(gt_data, model_pred_data, iou_type="segm", max_dets=350)

    print(f"\nEvaluating {chosen_model_type.upper()} Bounding Box Metrics:")
    evaluate_coco_metrics(gt_data, model_pred_data, iou_type="bbox", max_dets=350)

    print("\nEvaluating SAM Segmentation Metrics:")
    gt_data, sam_pred_data = generate_coco_annotations_from_multi_instance_masks(
        gt_mask_path,
        combined_sam_mask_path,
        image_path
    )
    evaluate_coco_metrics(gt_data, sam_pred_data, iou_type="segm", max_dets=350)

    print("\nEvaluating SAM Bounding Box Metrics:")
    evaluate_coco_metrics(gt_data, sam_pred_data, iou_type="bbox", max_dets=350)

if __name__ == "__main__":
    main()
