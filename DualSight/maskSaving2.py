#!/usr/bin/env python3
import os
import pickle
import time
import uuid
import numpy as np
from PIL import Image
import torch

# Adjust the system path to locate the helper modules if necessary
import sys
sys.path.append('../')  # Adjust if your sam_helper and tools modules are elsewhere

from sam_helper import load_sam_model, run_sam_inference
from tools import combine_masks_16bit  # Not used here, but imported in case you need it

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
    # containing keys "boxes", "polygons", "masks" and optionally "labels".
    inference_pickle_path = "/home/sprice/ICCV25/savedInference/particle_yolov8n_inference.pkl"
    
    # Output directory where each refined mask and corresponding GT mask will be saved individually
    output_dir = "single_masks"
    os.makedirs(output_dir, exist_ok=True)
    
    # SAM model parameters (adjust these as needed)
    sam_model_type = "vit_l"
    sam_checkpoint = "/home/sprice/ICCV25/modelWeights/sam_vit_l.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SAM inference parameters
    sam_params = {
        "num_points": 7,               # Number of points of interest per mask
        "algorithm": "Distance Max",   # e.g., "Random" or your chosen algorithm
        "ignore_border_percentage": 0,
        "use_box_input": True,
        "use_mask_input": True,
    }
    # These rates can be adjusted to expand the boxes/masks if needed
    box_expansion_rate = 1.1
    mask_expansion_rate = 1.1

    # -------------------------------------------------------------------------
    # Load the SAM model
    # -------------------------------------------------------------------------
    print("Loading SAM model...")
    sam_predictor = load_sam_model(
        sam_checkpoint=sam_checkpoint,
        model_type=sam_model_type,
        device=device
    )
    
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
    boxes = image_data.get("boxes", [])
    polygons = image_data.get("polygons", [])
    masks = image_data.get("masks", [])
    labels = image_data.get("labels", [])  # Optional labels

    if not boxes or not masks:
        print(f"[ERROR] No valid boxes/masks found for image '{image_filename}'.")
        return

    # -------------------------------------------------------------------------
    # Load the image and prepare the initial masks
    # -------------------------------------------------------------------------
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        print(f"[ERROR] Failed to load image '{image_path}': {e}")
        return

    # Convert the image to a NumPy array (BGR order expected by SAM)
    loop_image = np.array(pil_image)[:, :, ::-1].copy()

    # Ensure masks are in binary (uint8) format
    bin_masks = []
    for mask in masks:
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        # Convert to binary: assume mask pixel values > 127 indicate the object
        bin_mask = (mask > 127).astype(np.uint8) if mask.dtype != bool else mask.astype(np.uint8)
        bin_masks.append(bin_mask)

    # -------------------------------------------------------------------------
    # Run SAM to refine every mask
    # -------------------------------------------------------------------------
    print("Running SAM inference to refine masks...")
    start_time = time.time()
    refined_masks = run_sam_inference(
        predictor=sam_predictor,
        loop_image=loop_image,
        listOfPolygons=polygons,
        listOfBoxes=boxes,
        listOfMasks=bin_masks,
        image_width=pil_image.width,
        image_height=pil_image.height,
        num_points=sam_params["num_points"],
        dropout_percentage=0,  # if applicable; adjust if needed
        ignore_border_percentage=sam_params["ignore_border_percentage"],
        algorithm=sam_params["algorithm"],
        use_box_input=sam_params["use_box_input"],
        use_mask_input=sam_params["use_mask_input"],
        box_expansion_rate=box_expansion_rate,
        mask_expansion_rate=mask_expansion_rate
    )
    end_time = time.time()
    print(f"SAM inference completed in {end_time - start_time:.2f} seconds.")

    # -------------------------------------------------------------------------
    # Load the GT mask image and extract instance masks
    # -------------------------------------------------------------------------
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
    # Exclude background (assumed to be 0)
    gt_ids = [gt_id for gt_id in gt_ids if gt_id != 0]

    # -------------------------------------------------------------------------
    # For each refined mask, compute IoU with GT instances and save both images
    # -------------------------------------------------------------------------
    unique_id = f"{int(time.time())}_{uuid.uuid4().hex}"
    for idx, mask in enumerate(refined_masks):
        # Ensure the mask is a numpy array in uint8 format
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        mask = mask.astype(np.uint8)
        # If mask is binary (0 and 1) then scale to 0 and 255
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
        pred_mask_filename = f"{os.path.splitext(image_filename)[0]}_mask_{idx}_{unique_id}_{label_str}_iou_{best_iou:.5f}.png"
        pred_mask_path = os.path.join(output_dir, pred_mask_filename)
        Image.fromarray(mask).save(pred_mask_path)
        print(f"Saved predicted mask {idx} to: {pred_mask_path} with IoU: {best_iou:.5f}")
        
        # Save the corresponding GT mask if a match was found
        if best_gt_mask is not None:
            gt_save_filename = f"{os.path.splitext(image_filename)[0]}_mask_{idx}_{unique_id}_{label_str}_gt_{best_gt_id}.png"
            gt_save_path = os.path.join(output_dir, gt_save_filename)
            Image.fromarray(best_gt_mask.astype(np.uint8)).save(gt_save_path)
            print(f"Saved GT mask for predicted mask {idx} to: {gt_save_path}")

if __name__ == "__main__":
    main()
