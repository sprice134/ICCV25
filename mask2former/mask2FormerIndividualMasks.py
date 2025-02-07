#!/usr/bin/env python3
"""
mask2FormerIndividualMasks.py

This script loads a Mask2Former model (fine-tuned on COCO instance segmentation),
runs inference on a given image, post-processes the outputs by applying a sigmoid to 
the raw mask logits, and then saves each predicted instance mask individually as a PNG 
file in the specified output directory.

Usage example:
    python mask2FormerIndividualMasks.py \
        --image_path "/home/sprice/ICCV25/datasets/powder/valid/HP743_12S_250x_png.rf.134ec61e32a9bd54ab364723ae7d41fb.jpg" \
        --checkpoint "mask2former_instance_seg.pth" \
        --output_dir "individual_masks" \
        --threshold 0.75
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
import argparse
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

def save_individual_masks(masks, output_dir, threshold=0.5):
    """
    Given a NumPy array of predicted masks (shape: [num_instances, H, W]),
    threshold each mask to obtain a binary mask and save each as a PNG file in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_instances = masks.shape[0]
    for i in range(num_instances):
        mask = masks[i]
        # Binarize the mask: if it's a float array, threshold and scale to [0,255]
        if mask.dtype in [np.float32, np.float64]:
            binary_mask = (mask > threshold).astype(np.uint8) * 255
        else:
            binary_mask = mask.astype(np.uint8)
        out_path = os.path.join(output_dir, f"mask_{i+1}.png")
        if not cv2.imwrite(out_path, binary_mask):
            raise IOError(f"Failed to save mask to {out_path}")
    print(f"[INFO] Saved {num_instances} individual masks to {output_dir}")

def main(args):
    # Load image
    image = Image.open(args.image_path).convert("RGB")
    
    # Load processor and model (same as your working code)
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-instance")
    
    # Load the trained checkpoint if available
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"[WARNING] Checkpoint {args.checkpoint} not found. Using default pretrained weights.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Instead of using post_process_instance_segmentation to merge masks,
    # we extract the raw mask logits and apply sigmoid to get probabilities.
    if not hasattr(outputs, "masks_queries_logits"):
        raise AttributeError("Model output does not have 'masks_queries_logits'.")
    instance_masks = torch.sigmoid(outputs.masks_queries_logits)[0].cpu().numpy()  # shape: [num_instances, H, W]
    
    print(f"[INFO] Detected {instance_masks.shape[0]} instance mask(s).")
    
    # Save each mask individually to the specified output directory.
    save_individual_masks(instance_masks, args.output_dir, threshold=args.threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mask2Former inference and save each predicted instance mask as an individual PNG file.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where individual mask PNG files will be saved.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for converting predicted masks to binary (default: 0.5).")
    args = parser.parse_args()
    main(args)
