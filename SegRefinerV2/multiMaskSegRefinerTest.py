import os
import glob
import csv
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append('../DualSight/')

# YOLO
from ultralytics import YOLO

# SegRefiner imports (as shown in your code)
import torch
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.utils import replace_cfg_vals, rfnext_init_model
from mmdet.core.mask import BitmapMasks

# Tools or helper functions (same names as in your code)
# Make sure these are accessible in your Python path, or place them in the same directory.
from tools import (
    visualize_segmentations,
    generate_coco_annotations_from_multi_instance_masks,
    evaluate_coco_metrics
)

def load_segrefiner_model(config_path, checkpoint_path, device='cuda'):
    """
    Load SegRefiner model based on your provided code.
    """
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)

    cfg.device = device
    cfg.gpu_ids = [0]

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None

    # Build the SegRefiner model
    cfg.model.train_cfg = None  # Ensure testing mode
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    rfnext_init_model(model, cfg=cfg)

    # Load checkpoint
    load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
    model = model.to(device)
    model.eval()
    return model, cfg


def refine_masks_segrefiner(model, cfg, coarse_mask_path, image_path, device='cuda'):
    """
    Given a coarse multi-instance mask (e.g., from YOLO),
    refine it with SegRefiner and return path to the output refined mask.
    """
    if not os.path.exists(coarse_mask_path):
        raise FileNotFoundError(f"Coarse mask file not found: {coarse_mask_path}")

    # Prepare output name
    base_name = os.path.basename(image_path)
    refined_mask_name = base_name.replace(".jpg", "_refined.png")
    refined_mask_path = os.path.join("refinedOutputs", refined_mask_name)

    os.makedirs("refinedOutputs", exist_ok=True)

    # Load and preprocess the image
    img = mmcv.imread(image_path)
    if 'img_norm_cfg' in cfg:
        mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
        to_rgb = cfg.img_norm_cfg.get('to_rgb', True)
        img = mmcv.imnormalize(img, mean=mean, std=std, to_rgb=to_rgb)

    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Load the coarse multi-instance mask
    coarse_masks = mmcv.imread(coarse_mask_path, flag='grayscale')
    unique_labels = np.unique(coarse_masks)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (0)

    mask_height, mask_width = coarse_masks.shape
    refined_grayscale_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # Process each object ID in the coarse mask
    for idx, obj_id in enumerate(unique_labels, start=1):
        single_mask = (coarse_masks == obj_id).astype(np.uint8)
        single_mask = single_mask[np.newaxis, :, :]
        coarse_bitmap_mask = BitmapMasks(single_mask, height=mask_height, width=mask_width)

        img_metas = {
            'ori_filename': base_name,
            'img_shape': img.shape[:2] + (3,),
            'ori_shape': img.shape[:2] + (3,),
            'pad_shape': img.shape[:2] + (3,),
            'scale_factor': 1.0,
            'flip': False,
        }

        data = {
            'img': img_tensor,
            'img_metas': [img_metas],
            'coarse_masks': [coarse_bitmap_mask],
        }

        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)

        refined_image = results[0][0]  # Adjust index based on your model’s output
        if isinstance(refined_image, torch.Tensor):
            refined_image = refined_image.cpu().numpy()

        if refined_image.ndim == 3 and refined_image.shape[0] == 3:
            refined_image = refined_image.transpose(1, 2, 0)

        refined_image = (refined_image * 255).astype(np.uint8)
        refined_grayscale_mask[refined_image > 0] = idx

    mmcv.imwrite(refined_grayscale_mask, refined_mask_path)
    return refined_mask_path


def main():
    # ------------------------------------------------------------------------------
    # 1) Setup directories and paths
    # ------------------------------------------------------------------------------
    image_dir = "/home/sprice/RQ/demo.v7i.yolov8/test/images/"  # Where your .jpg files are
    gt_mask_dir = "/home/sprice/ICCV25/SegRefinerV2/test_masks/"  # Where your *_mask.png files are
    output_dir = "outputImages"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------------------
    # 2) Load YOLO model
    # ------------------------------------------------------------------------------
    yolo_model_path = "/home/sprice/ICCV25/modelWeights/yolov8n-seg.pt"
    yolo_model = YOLO(yolo_model_path)

    # ------------------------------------------------------------------------------
    # 3) Load SegRefiner model
    # ------------------------------------------------------------------------------
    segrefiner_config_path = "/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py"
    segrefiner_checkpoint_path = "/home/sprice/ICCV25/modelWeights/segrefiner_lr_latest.pth"
    segrefiner_model, segrefiner_cfg = load_segrefiner_model(
        segrefiner_config_path, segrefiner_checkpoint_path, device='cuda'
    )

    # Prepare to store metrics
    results = []

    # ------------------------------------------------------------------------------
    # 4) Get all .jpg images in the directory
    # ------------------------------------------------------------------------------
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not image_paths:
        print(f"No .jpg images found in {image_dir}")
        return

    # ------------------------------------------------------------------------------
    # 5) Loop over images
    # ------------------------------------------------------------------------------
    for img_path in image_paths:
        image_name = os.path.basename(img_path)
        gt_mask_name = image_name.replace(".jpg", "_mask.png")
        gt_mask_path = os.path.join(gt_mask_dir, gt_mask_name)

        # Safety check for GT
        if not os.path.exists(gt_mask_path):
            print(f"[WARNING] GT mask not found for {image_name}, skipping...")
            continue

        print(f"\nProcessing {image_name} ...")

        # ------------------------------------------------------------------------------
        # 6) Run YOLO inference to get coarse masks
        # ------------------------------------------------------------------------------
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[ERROR] Could not read image: {img_path}")
            continue
        original_height, original_width = image_bgr.shape[:2]
        yolo_results = yolo_model(image_bgr)

        # Extract YOLO masks
        yolo_masks = yolo_results[0].masks.data  # shape: [num_instances, h, w]
        if yolo_masks is None or yolo_masks.numel() == 0:
            print("No masks detected by YOLO. Skipping.")
            continue

        yolo_masks = yolo_masks.cpu().numpy()
        combined_yolo_mask = np.zeros_like(yolo_masks[0], dtype=np.uint8)
        for idx, mask in enumerate(yolo_masks, start=1):
            combined_yolo_mask[mask > 0.5] = idx

        # Resize back to original dims
        combined_yolo_mask_resized = cv2.resize(
            combined_yolo_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Save YOLO combined mask
        coarse_mask_name = image_name.replace(".jpg", "_yolo_mask.png")
        coarse_mask_path = os.path.join(output_dir, coarse_mask_name)
        cv2.imwrite(coarse_mask_path, combined_yolo_mask_resized)

        # ------------------------------------------------------------------------------
        # 7) Refine with SegRefiner
        # ------------------------------------------------------------------------------
        refined_mask_path = refine_masks_segrefiner(
            segrefiner_model,
            segrefiner_cfg,
            coarse_mask_path,
            img_path,
            device='cuda'
        )

        # ------------------------------------------------------------------------------
        # 8) Visualize segmentations (optional)
        # ------------------------------------------------------------------------------
        try:
            visualize_segmentations(img_path, gt_mask_path, output_dir, title="Ground Truth")
            visualize_segmentations(img_path, coarse_mask_path, output_dir, title="Coarse YOLO Mask")
            visualize_segmentations(img_path, refined_mask_path, output_dir, title="Refined SegRefiner Mask")
        except Exception as e:
            print(f"[WARNING] Visualization error for {image_name}: {e}")

        # ------------------------------------------------------------------------------
        # 9) Evaluate with COCO metrics
        #    We'll do YOLO's coarse vs. GT, and SegRefiner refined vs. GT
        # ------------------------------------------------------------------------------
        # Evaluate YOLO coarse
        gt_data, yolo_pred_data = generate_coco_annotations_from_multi_instance_masks(
            gt_mask_path,
            coarse_mask_path,
            img_path,
            image_id=1  # or use a unique ID if you prefer
        )
        yolo_metrics = evaluate_coco_metrics(gt_data, yolo_pred_data, iou_type="segm", max_dets=350)

        # Evaluate SegRefiner
        # Re-use gt_data or re-generate for clarity
        gt_data, refined_pred_data = generate_coco_annotations_from_multi_instance_masks(
            gt_mask_path,
            refined_mask_path,
            img_path,
            image_id=1
        )
        segrefiner_metrics = evaluate_coco_metrics(gt_data, refined_pred_data, iou_type="segm", max_dets=350)

        # Suppose each returns a dictionary with keys like
        # ["AP@50", "AP@75", "AP@95", "AP@50:95", "AR@50", "AR@75", "AR@95", "AR@50:95"]
        # If your evaluate_coco_metrics prints them but does not return them,
        # you'd need to adapt your code in `tools/evaluate_coco_metrics` to return the dictionary.
        # For demonstration, we’ll assume you can capture those metrics:
        # yolo_metrics = {
        #    "AP@50": ...,
        #    "AP@75": ...,
        #    "AP@95": ...,
        #    "AP@50:95": ...,
        #    ...
        # }
        # segrefiner_metrics = {...}

        # For safety, we handle the case if yolo_metrics or segrefiner_metrics is None or incomplete:
        def safe_get(d, k):
            return d[k] if d and k in d else None

        # Collect results for CSV
        results.append({
            "image_name": image_name,

            "yolo_mask_AP@50": safe_get(yolo_metrics, "AP@50"),
            "yolo_mask_AP@75": safe_get(yolo_metrics, "AP@75"),
            "yolo_mask_AP@95": safe_get(yolo_metrics, "AP@95"),
            "yolo_mask_AP@50:95": safe_get(yolo_metrics, "AP@50:95"),
            "yolo_mask_AR@50": safe_get(yolo_metrics, "AR@50"),
            "yolo_mask_AR@75": safe_get(yolo_metrics, "AR@75"),
            "yolo_mask_AR@95": safe_get(yolo_metrics, "AR@95"),
            "yolo_mask_AR@50:95": safe_get(yolo_metrics, "AR@50:95"),

            "segrefiner_mask_AP@50": safe_get(segrefiner_metrics, "AP@50"),
            "segrefiner_mask_AP@75": safe_get(segrefiner_metrics, "AP@75"),
            "segrefiner_mask_AP@95": safe_get(segrefiner_metrics, "AP@95"),
            "segrefiner_mask_AP@50:95": safe_get(segrefiner_metrics, "AP@50:95"),
            "segrefiner_mask_AR@50": safe_get(segrefiner_metrics, "AR@50"),
            "segrefiner_mask_AR@75": safe_get(segrefiner_metrics, "AR@75"),
            "segrefiner_mask_AR@95": safe_get(segrefiner_metrics, "AR@95"),
            "segrefiner_mask_AR@50:95": safe_get(segrefiner_metrics, "AR@50:95"),
        })

    # ------------------------------------------------------------------------------
    # 10) Write results to CSV
    # ------------------------------------------------------------------------------
    csv_filename = os.path.join(output_dir, "segrefiner_evaluation_metrics.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nEvaluation metrics saved to {csv_filename}")
    else:
        print("\nNo results to save. Did not find any valid images or masks.")


if __name__ == "__main__":
    main()
