"""
sam_refine_inference.py

Usage Example:
    python model_sam_inference.py \
        --inference-pickle "../../savedInference/particle_yolov8n_inference.pkl" \
        --images-dir "/home/sprice/ICCV25/datasets/powder/test" \
        --output-pickle "../../savedInference/particle_yolov8n_dualsight_inference.pkl" \
        --device "cuda"

Description:
    This script loads an existing inference pickle (produced by model_inference.py),
    loads each corresponding image, runs SAM to refine the annotations, and then
    saves a new pickle file with the same structure as the original. In each image’s
    dictionary, the key "masks" is replaced with the refined masks.
"""

import os
import pickle
import argparse
import numpy as np
import torch
from PIL import Image
import sys
sys.path.append('../')  # Adjust the path as needed to find your modules

# Import SAM helper functions. These should provide the SAM model loader and inference routine.
from sam_helper import load_sam_model, run_sam_inference

# -----------------------------------------------------------------------------
# Hardcoded SAM parameters -- adjust these as needed
# -----------------------------------------------------------------------------
SAM_MODEL_TYPE = "vit_l"
SAM_CHECKPOINT = "/home/sprice/ICCV25/modelWeights/sam_vit_l.pth"  # Change this to your checkpoint path
SAM_NUM_POINTS = 3
SAM_IGNORE_BORDER_PERCENTAGE = 0.1  # 10% perimeter buffer (0.1)
SAM_ALGORITHM = "Random"           # Change if needed (e.g. "Random" or another algorithm)
SAM_USE_BOX_INPUT = True
SAM_USE_MASK_INPUT = False
SAM_BOX_EXPANSION_RATE = 1.0        # 100% of the bounding box (using the full bounding box)
SAM_MASK_EXPANSION_RATE = 0.0

# -----------------------------------------------------------------------------
# Main function: load inference, run SAM, and save with the same structure.
# -----------------------------------------------------------------------------
def refine_inference_with_sam(inference_pickle, images_dir, output_pickle, device="cuda"):
    """
    Loads an inference pickle containing original model outputs (with keys: polygons, boxes, masks),
    then for each image uses SAM to refine the annotations. The refined masks replace the original
    "masks" key so that the output pickle maintains the same structure.
    
    Args:
        inference_pickle (str): Path to the pickle file with original inference.
        images_dir (str): Directory containing the source images.
        output_pickle (str): Path where the refined inference pickle will be saved.
        device (str): Device to use (default "cuda").
    """
    # Load the existing inference data.
    print(f"[INFO] Loading inference data from {inference_pickle}")
    with open(inference_pickle, "rb") as f:
        inference_data = pickle.load(f)
    
    # Load the SAM model using your helper function.
    print(f"[INFO] Loading SAM model ({SAM_MODEL_TYPE}) from {SAM_CHECKPOINT} on device {device}")
    sam_predictor = load_sam_model(
        sam_checkpoint=SAM_CHECKPOINT,
        model_type=SAM_MODEL_TYPE,
        device=device
    )
    
    refined_data = {}

    # Process each image in the inference pickle.
    for image_name, ann_data in inference_data.items():
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"[WARNING] Image '{image_path}' not found. Skipping...")
            continue

        try:
            # Open the image using PIL; convert to RGB.
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Could not load image '{image_path}': {e}")
            continue

        # Convert to a NumPy array. Some SAM routines expect BGR format so we flip the channels.
        loop_image = np.array(pil_image)[:, :, ::-1].copy()
        image_width, image_height = pil_image.width, pil_image.height

        # Retrieve the original annotations.
        # (The original inference structure contains the keys "polygons", "boxes", and "masks".)
        polygons = ann_data.get("polygons", [])
        boxes = ann_data.get("boxes", [])
        masks = ann_data.get("masks", [])

        # Process masks: if they are tensors, convert to NumPy arrays and binarize them.
        bin_masks = []
        for mask in masks:
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            # Binarize the mask (assuming values >127 indicate foreground).
            if mask.dtype != bool:
                bin_mask = (mask > 127).astype(np.uint8)
            else:
                bin_mask = mask.astype(np.uint8)
            bin_masks.append(bin_mask)

        # Run SAM to refine the annotations.
        refined_masks = run_sam_inference(
            predictor=sam_predictor,
            loop_image=loop_image,
            listOfPolygons=polygons,
            listOfBoxes=boxes,
            listOfMasks=bin_masks,
            image_width=image_width,
            image_height=image_height,
            num_points=SAM_NUM_POINTS,
            dropout_percentage=0,  # Hardcoded; adjust if needed.
            ignore_border_percentage=SAM_IGNORE_BORDER_PERCENTAGE,
            algorithm=SAM_ALGORITHM,
            use_box_input=SAM_USE_BOX_INPUT,
            use_mask_input=SAM_USE_MASK_INPUT,
            box_expansion_rate=SAM_BOX_EXPANSION_RATE,
            mask_expansion_rate=SAM_MASK_EXPANSION_RATE
        )

        # Replace the original "masks" key with the refined masks.
        ann_data["masks"] = refined_masks

        # Save the updated dictionary for this image (structure remains identical).
        refined_data[image_name] = ann_data
        print(f"[INFO] Processed image '{image_name}'")

    # Ensure the output directory exists and save the refined data.
    os.makedirs(os.path.dirname(output_pickle), exist_ok=True)
    with open(output_pickle, "wb") as f:
        pickle.dump(refined_data, f)
    print(f"[INFO] Saved refined inference data to '{output_pickle}'")

# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a saved inference pickle, refine its annotations with SAM, and save a new pickle with identical structure."
    )
    parser.add_argument(
        "--inference-pickle",
        type=str,
        required=True,
        help="Path to the pickle file with original model inference results."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing the original images corresponding to the inference."
    )
    parser.add_argument(
        "--output-pickle",
        type=str,
        required=True,
        help="Path to save the refined inference pickle."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run SAM on (default: cuda)."
    )
    args = parser.parse_args()

    refine_inference_with_sam(
        inference_pickle=args.inference_pickle,
        images_dir=args.images_dir,
        output_pickle=args.output_pickle,
        device=args.device
    )
