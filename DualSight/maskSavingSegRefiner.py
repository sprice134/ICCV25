#!/usr/bin/env python3
import os
import pickle
import time
import uuid
import numpy as np
from PIL import Image
import cv2
import torch
import sys
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.utils import replace_cfg_vals, rfnext_init_model
from mmdet.core.mask import BitmapMasks

# ------------------ Utility Functions ------------------ #
def combine_masks_16bit(list_of_binary_masks, output_path=None, return_array=False):
    """
    Combine a list of 0/1 or 0/255 binary masks into a single 16-bit instance mask.
    Each mask is assigned a unique ID (1, 2, 3, ...).
    """
    if not list_of_binary_masks:
        print("[WARNING] No masks provided to combine.")
        return None

    first_shape = list_of_binary_masks[0].shape
    for idx, mask in enumerate(list_of_binary_masks):
        if mask.shape != first_shape:
            raise ValueError(f"All masks must have the same shape. Mask at index {idx} has shape {mask.shape}, expected {first_shape}.")

    height, width = first_shape
    combined_16bit = np.zeros((height, width), dtype=np.uint16)

    for idx, mask in enumerate(list_of_binary_masks, start=1):
        unique_vals = set(np.unique(mask))
        if mask.dtype != bool and not (unique_vals <= {0, 1} or unique_vals <= {0, 255}):
            raise ValueError(
                f"Mask at index {idx-1} has unexpected unique values {unique_vals}. "
                "Please provide 0/1 or 0/255 masks."
            )
        mask_bool = mask.astype(bool)
        combined_16bit[mask_bool] = idx

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_16bit = combined_16bit.astype(np.uint16)
        success = cv2.imwrite(output_path, combined_16bit)
        if not success:
            raise IOError(f"Failed to write the combined mask to {output_path}")

    if return_array:
        return combined_16bit
    return None

def compute_iou(pred_mask, gt_mask):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

# ------------------ SegRefiner Functions ------------------ #
def load_segrefiner_model(segrefiner_config, segrefiner_checkpoint, device='cuda'):
    """
    Build and load the SegRefiner model from config and checkpoint.
    """
    cfg = Config.fromfile(segrefiner_config)
    cfg = replace_cfg_vals(cfg)

    # Device setup
    cfg.device = device
    cfg.gpu_ids = [0] if 'cuda' in device else []

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None

    # Build the model (in test mode)
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
    Given a BGR image and a multi-instance 16-bit coarse mask,
    refine each instance using the SegRefiner model and return a refined 16-bit mask.
    """
    # Normalize image as per config (if provided)
    img = bgr_image.copy()
    if 'img_norm_cfg' in cfg:
        mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
        to_rgb = cfg.img_norm_cfg.get('to_rgb', True)
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = mmcv.imnormalize(img, mean=mean, std=std, to_rgb=False)
    else:
        img = img.astype(np.float32)

    # Convert image to torch tensor (C,H,W)
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Ensure the coarse mask is in 8-bit (IDs assumed to be <=255)
    max_id = coarse_mask_16bit.max()
    if max_id > 255:
        raise ValueError(f"[ERROR] refine_masks_with_segrefiner: instance ID {max_id} > 255 found. Please reduce instance IDs.")
    coarse_mask_8bit = coarse_mask_16bit.astype(np.uint8)

    height, width = coarse_mask_8bit.shape
    unique_ids = np.unique(coarse_mask_8bit)
    unique_ids = unique_ids[unique_ids > 0]  # exclude background

    refined_mask_16bit = np.zeros((height, width), dtype=np.uint16)

    # Minimal image meta info for the model
    img_metas = {
        'ori_filename': 'placeholder.jpg',
        'img_shape': (height, width, 3),
        'ori_shape': (height, width, 3),
        'pad_shape': (height, width, 3),
        'scale_factor': 1.0,
        'flip': False,
    }

    # Refine each instance separately
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
        # Assuming the refined mask is in results[0][0]
        refined = results[0][0]
        if isinstance(refined, torch.Tensor):
            refined = refined.cpu().numpy()
        if refined.ndim == 3 and refined.shape[0] == 3:
            refined = refined.transpose(1, 2, 0)
        refined = (refined * 255).astype(np.uint8)

        # Any pixel >0 is assigned the instance ID (idx)
        refined_mask_16bit[refined > 0] = idx

    return refined_mask_16bit

# ------------------ Main Script ------------------ #
def main():
    # ------------------ Hardcoded Parameters ------------------ #
    # Paths to the image, inference pickle, and output directory
    image_path = "/home/sprice/ICCV25/datasets/powder/test/Cu-Ni-Powder_250x_10_SE_png.rf.cd93ec4589ad8f4e412cb1ec0e805016.jpg"
    inference_pickle_path = "/home/sprice/ICCV25/savedInference/particle_yolov8n_inference.pkl"
    output_dir = "single_masks"
    os.makedirs(output_dir, exist_ok=True)
    
    # SegRefiner model parameters (adjust paths as needed)
    segrefiner_config = "/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py"
    segrefiner_checkpoint = "/home/sprice/ICCV25/modelWeights/segrefiner_hr_latest.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ------------------ Load the Image ------------------ #
    # For SegRefiner, we use cv2 (BGR format)
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        print(f"[ERROR] Failed to load image '{image_path}'")
        return

    # ------------------ Load Inference Data ------------------ #
    if not os.path.exists(inference_pickle_path):
        print(f"[ERROR] Inference pickle file not found: {inference_pickle_path}")
        return
    with open(inference_pickle_path, "rb") as pf:
        inference_data = pickle.load(pf)
    
    image_filename = os.path.basename(image_path)
    if image_filename not in inference_data:
        print(f"[ERROR] Inference data for image '{image_filename}' not found in the pickle file.")
        return

    image_data = inference_data[image_filename]
    predicted_masks = image_data.get("masks", [])
    labels = image_data.get("labels", [])  # Optional labels

    if not predicted_masks:
        print(f"[ERROR] No valid masks found for image '{image_filename}'.")
        return

    # ------------------ Combine Predicted Masks ------------------ #
    try:
        coarse_16bit = combine_masks_16bit(predicted_masks, return_array=True)
    except ValueError as ve:
        print(f"[ERROR] {ve}")
        return

    # ------------------ Load SegRefiner Model ------------------ #
    print("Loading SegRefiner model...")
    try:
        segrefiner_model, segrefiner_cfg = load_segrefiner_model(segrefiner_config, segrefiner_checkpoint, device=device)
    except Exception as e:
        print(f"[ERROR] Failed to load SegRefiner model: {e}")
        return

    # ------------------ Refine the Combined Mask ------------------ #
    print("Running SegRefiner refinement on the combined mask...")
    try:
        refined_16bit = refine_masks_with_segrefiner(
            bgr_image=bgr_image,
            coarse_mask_16bit=coarse_16bit,
            model=segrefiner_model,
            cfg=segrefiner_cfg,
            device=device
        )
    except Exception as e:
        print(f"[ERROR] SegRefiner refinement failed: {e}")
        return

    unique_id = f"{int(time.time())}_{uuid.uuid4().hex}"
    # Save the entire refined 16-bit mask (for debugging or further processing)
    refined_mask_path = os.path.join(output_dir, f"refined_{os.path.splitext(image_filename)[0]}_{unique_id}.png")
    cv2.imwrite(refined_mask_path, refined_16bit)
    print(f"Saved refined 16-bit mask to: {refined_mask_path}")

    # ------------------ Load GT Mask ------------------ #
    image_dir = os.path.dirname(image_path)
    gt_mask_dir = os.path.join(image_dir, "annotations")
    gt_mask_filename = image_filename.replace(".jpg", "_mask.png")
    gt_mask_path = os.path.join(gt_mask_dir, gt_mask_filename)
    if not os.path.exists(gt_mask_path):
        print(f"[ERROR] GT mask file not found: {gt_mask_path}")
        return
    gt_pil = Image.open(gt_mask_path)
    gt_mask_arr = np.array(gt_pil)
    gt_ids = np.unique(gt_mask_arr)
    gt_ids = [gt_id for gt_id in gt_ids if gt_id != 0]

    # ------------------ Evaluate Each Refined Instance ------------------ #
    # For each unique instance ID (ignoring background) in the refined mask,
    # compute IoU against GT instances and save the mask images.
    instance_ids = np.unique(refined_16bit)
    instance_ids = instance_ids[instance_ids > 0]
    for idx, inst_id in enumerate(instance_ids):
        # Create binary mask for the current instance
        refined_inst_mask = (refined_16bit == inst_id).astype(np.uint8) * 255
        pred_binary = (refined_inst_mask > 127).astype(np.uint8)

        best_iou = 0.0
        best_gt_id = None
        best_gt_mask = None
        # Compare against each GT instance mask
        for gt_id in gt_ids:
            gt_inst_mask = (gt_mask_arr == gt_id).astype(np.uint8)
            iou = compute_iou(pred_binary, gt_inst_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
                best_gt_mask = gt_inst_mask * 255

        label_str = ""
        if labels and idx < len(labels):
            label_str = str(labels[idx])
        
        # Save the refined instance mask
        pred_mask_filename = f"{os.path.splitext(image_filename)[0]}_mask_{inst_id}_{unique_id}_{label_str}_iou_{best_iou:.5f}.png"
        pred_mask_path = os.path.join(output_dir, pred_mask_filename)
        cv2.imwrite(pred_mask_path, refined_inst_mask)
        print(f"Saved refined mask instance {inst_id} to: {pred_mask_path} with IoU: {best_iou:.5f}")
        
        # Save the corresponding GT mask if available
        if best_gt_mask is not None:
            gt_save_filename = f"{os.path.splitext(image_filename)[0]}_mask_{inst_id}_{unique_id}_{label_str}_gt_{best_gt_id}.png"
            gt_save_path = os.path.join(output_dir, gt_save_filename)
            cv2.imwrite(gt_save_path, best_gt_mask)
            print(f"Saved GT mask for instance {inst_id} to: {gt_save_path}")

if __name__ == "__main__":
    main()
