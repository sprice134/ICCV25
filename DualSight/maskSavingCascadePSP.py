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

# Adjust the system path to locate the helper modules if necessary.
sys.path.append('../')
sys.path.append('../../DualSight')  # Adjust as needed to locate your 'segmentation_refinement' module

import segmentation_refinement as refine

def compute_iou(pred_mask, gt_mask):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

def main():
    # -------------------------------------------------------------------------
    # Hardcoded parameters and paths
    # -------------------------------------------------------------------------
    # Path to the image you wish to process:
    image_path = "/home/sprice/ICCV25/datasets/powder/test/Cu-Ni-Powder_250x_10_SE_png.rf.cd93ec4589ad8f4e412cb1ec0e805016.jpg"
    
    # Path to the saved initial inference values (pickle file)
    # This pickle is assumed to be a dictionary mapping image filenames to a dict
    # containing a key "masks" (and optionally "labels").
    inference_pickle_path = "/home/sprice/ICCV25/savedInference/particle_yolov8n_inference.pkl"
    
    # Output directory where each refined mask will be saved individually
    output_dir = "single_masks"
    os.makedirs(output_dir, exist_ok=True)
    
    # CascadePSP refinement parameters (adjust as needed)
    refine_fast = False  # Use fast refinement? Set to False for higher quality.
    L = 900            # A parameter for the refinement (adjust based on your requirements)
    
    # -------------------------------------------------------------------------
    # Initialize the CascadePSP refiner
    # -------------------------------------------------------------------------
    print("Initializing CascadePSP refiner...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    refiner = refine.Refiner(device=device)
    
    # -------------------------------------------------------------------------
    # Load the saved inference values for initial segmentation
    # -------------------------------------------------------------------------
    if not os.path.exists(inference_pickle_path):
        print(f"[ERROR] Inference pickle file not found: {inference_pickle_path}")
        return

    with open(inference_pickle_path, "rb") as pf:
        inference_data = pickle.load(pf)
    
    # Use the basename of the image file to look up its initial inference values
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

    # -------------------------------------------------------------------------
    # Load the image and prepare for refinement
    # -------------------------------------------------------------------------
    # Read the image using cv2 (BGR format is expected by CascadePSP)
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        print(f"[ERROR] Failed to load image '{image_path}'")
        return

    # -------------------------------------------------------------------------
    # Refine each predicted mask using CascadePSP
    # -------------------------------------------------------------------------
    print("Running CascadePSP refinement on predicted masks...")
    refined_masks = []
    for idx, pmask in enumerate(predicted_masks):
        # Convert torch tensor to numpy array if necessary
        if hasattr(pmask, "cpu"):
            pmask = pmask.cpu().numpy()
        # Ensure mask is in uint8 format
        if pmask.dtype != np.uint8:
            pmask = pmask.astype(np.uint8)
        # If mask is binary (0 and 1), scale to 0 and 255
        if pmask.max() <= 1:
            pmask = pmask * 255

        # Use the CascadePSP refiner
        refined_mask = refiner.refine(
            image=bgr_image,
            mask=pmask,
            fast=refine_fast,
            L=L
        )
        # Binarize the refined mask using Otsu's thresholding
        _, refined_mask_binary = cv2.threshold(refined_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        refined_masks.append(refined_mask_binary)
    
    # -------------------------------------------------------------------------
    # Load the GT mask image and extract instance masks
    # -------------------------------------------------------------------------
    image_dir = os.path.dirname(image_path)
    gt_mask_dir = os.path.join(image_dir, "annotations")
    # Assumes GT mask filename is the image filename with ".jpg" replaced by "_mask.png"
    gt_mask_filename = image_filename.replace(".jpg", "_mask.png")
    gt_mask_path = os.path.join(gt_mask_dir, gt_mask_filename)
    
    if not os.path.exists(gt_mask_path):
        print(f"[ERROR] GT mask file not found: {gt_mask_path}")
        return

    gt_pil = Image.open(gt_mask_path)
    gt_mask_arr = np.array(gt_pil)
    gt_ids = np.unique(gt_mask_arr)
    # Exclude background (assumed to be 0)
    gt_ids = [gt_id for gt_id in gt_ids if gt_id != 0]
    
    # -------------------------------------------------------------------------
    # For each refined mask, compute IoU with GT instances and save both images
    # -------------------------------------------------------------------------
    unique_id = f"{int(time.time())}_{uuid.uuid4().hex}"
    for idx, mask in enumerate(refined_masks):
        # Ensure mask is a numpy array in uint8 format
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        mask = mask.astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255

        # Create a binary version for IoU calculation
        pred_binary = (mask > 127).astype(np.uint8)

        best_iou = 0.0
        best_gt_id = None
        best_gt_mask = None
        # Compare against each GT instance mask
        for gt_id in gt_ids:
            gt_instance_mask = (gt_mask_arr == gt_id).astype(np.uint8)
            iou = compute_iou(pred_binary, gt_instance_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
                best_gt_mask = gt_instance_mask * 255  # Scale to 0-255

        # Include label (if available) in the filename
        label_str = ""
        if labels and idx < len(labels):
            label_str = str(labels[idx])
        
        # Save the refined (predicted) mask
        pred_mask_filename = f"{os.path.splitext(image_filename)[0]}_mask_{idx}_{unique_id}_{label_str}_iou_{best_iou:.2f}.png"
        pred_mask_path = os.path.join(output_dir, pred_mask_filename)
        Image.fromarray(mask).save(pred_mask_path)
        print(f"Saved refined mask {idx} to: {pred_mask_path} with IoU: {best_iou:.5f}")
        
        # Save the corresponding GT mask if a match was found
        if best_gt_mask is not None:
            gt_save_filename = f"{os.path.splitext(image_filename)[0]}_mask_{idx}_{unique_id}_{label_str}_gt_{best_gt_id}.png"
            gt_save_path = os.path.join(output_dir, gt_save_filename)
            Image.fromarray(best_gt_mask.astype(np.uint8)).save(gt_save_path)
            print(f"Saved corresponding GT mask for refined mask {idx} to: {gt_save_path}")

if __name__ == "__main__":
    main()
