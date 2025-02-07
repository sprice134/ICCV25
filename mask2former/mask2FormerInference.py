#!/usr/bin/env python3
"""
mask2FormerInference.py

This script loads a Mask2Former model (fine-tuned on COCO instance segmentation),
runs inference on a given image, post-processes the outputs using the processor’s
`post_process_instance_segmentation` method, extracts the combined instance segmentation map,
overlays it colorfully on the original image (without modifying the background), and saves
the result as a PNG. It also prints the number of detected objects.

Usage example:
    python mask2FormerInference.py \
        --image_path "/home/sprice/ICCV25/datasets/powder/valid/HP743_12S_250x_png.rf.134ec61e32a9bd54ab364723ae7d41fb.jpg" \
        --checkpoint "mask2former_instance_seg.pth" \
        --output_png "inference_output.png" \
        --threshold 0.95

    python mask2FormerInference.py \
        --image_path "/home/sprice/ICCV25/datasets/powder/valid/S02_03_SE1_500X23_png.rf.5e66a6f124b1c5950b099f853ca4e18c.jpg" \
        --checkpoint "mask2former_instance_seg.pth" \
        --output_png "inference_output.png" \
        --threshold 0.95

    python mask2FormerInference.py \
        --image_path "/home/sprice/ICCV25/datasets/powder/train/S03_02_SE1_3000X36_png.rf.609324ab2f6a741aefc6be2f96501ce7.jpg" \
        --checkpoint "mask2former_instance_seg.pth" \
        --output_png "inference_output.png" \
        --threshold 0.95
        
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
import argparse
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

def combine_instance_masks(instance_masks, threshold=0.5):
    """
    (Optional) Given a NumPy array of predicted masks (shape: [num_queries, H, W]),
    binarize each mask using the given threshold and combine them into a single
    16-bit label image where each instance is assigned a unique integer label.
    """
    if instance_masks.size == 0:
        return None
    H, W = instance_masks.shape[1:]
    combined = np.zeros((H, W), dtype=np.uint16)
    label = 1
    for mask in instance_masks:
        binary = (mask > threshold).astype(np.uint8)
        if np.sum(binary) > 0:
            combined[binary == 1] = label
            label += 1
    return combined

def colorize_segmentation(segmentation):
    """
    Given a 2D segmentation label image, return a color image (H, W, 3)
    where we assume that the background is the most frequent label.
    That background label is forced to black; all other labels are given random colors.
    
    Returns both the colorized image and the background label.
    """
    # Find all labels and their counts.
    labels, counts = np.unique(segmentation, return_counts=True)
    # Assume the most frequent label is background.
    background_label = labels[np.argmax(counts)]
    # Create a color map: force background to black.
    color_map = {background_label: np.array([0, 0, 0], dtype=np.uint8)}
    for lbl in labels:
        if lbl == background_label:
            continue
        color_map[lbl] = np.random.randint(0, 256, size=3, dtype=np.uint8)
    
    H, W = segmentation.shape
    color_image = np.zeros((H, W, 3), dtype=np.uint8)
    for lbl, color in color_map.items():
        color_image[segmentation == lbl] = color
    return color_image, background_label

def overlay_segmentation_on_image(original, segmentation_color, segmentation_labels, background_label, alpha=0.5):
    """
    Overlays segmentation_color on top of the original image only for pixels
    corresponding to foreground (where segmentation_labels != background_label).
    The background remains unchanged.
    
    Both original and segmentation_color should be in BGR format and of the same shape.
    """
    output = original.copy()
    # Create a boolean mask for foreground pixels (non-background).
    foreground_mask = segmentation_labels != background_label
    # Blend only the foreground pixels.
    output[foreground_mask] = cv2.addWeighted(
        original[foreground_mask], 1 - alpha,
        segmentation_color[foreground_mask], alpha, 0
    )
    return output

def main(args):
    # Load image using PIL and convert to BGR (OpenCV format).
    image = Image.open(args.image_path).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Load processor and model.
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-instance")
    
    # Load the trained checkpoint if available.
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"[WARNING] Checkpoint {args.checkpoint} not found. Using default pretrained weights.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Process the image for the model.
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference.
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the processor's post_process_instance_segmentation method.
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]
    
    if "segmentation" not in result:
        raise ValueError("Post-processed results do not contain the 'segmentation' key.")
    
    predicted_instance_map = result["segmentation"]
    print("Shape of Instance Map:", predicted_instance_map.shape)
    
    if not isinstance(predicted_instance_map, np.ndarray):
        if isinstance(predicted_instance_map, torch.Tensor):
            predicted_instance_map = predicted_instance_map.cpu().numpy()
        else:
            predicted_instance_map = np.array(predicted_instance_map)
    
    # Count and print the number of detected objects (excluding background).
    labels, counts = np.unique(predicted_instance_map, return_counts=True)
    # Assume the background label is the most frequent one.
    background_label = labels[np.argmax(counts)]
    num_objects = len(labels[labels != background_label])
    print(f"[INFO] Number of detected objects: {num_objects}")
    
    # Create a color version of the segmentation map.
    seg_color, bg_label = colorize_segmentation(predicted_instance_map)
    
    # Overlay segmentation on the original image (foreground only).
    overlay = overlay_segmentation_on_image(image_np, seg_color, predicted_instance_map, bg_label, alpha=0.5)
    
    # Save the overlay as a PNG.
    if not cv2.imwrite(args.output_png, overlay):
        raise IOError(f"Failed to write the output PNG to {args.output_png}")
    print(f"[INFO] Overlay inference result saved as PNG to {args.output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Mask2Former inference, overlay instance segmentation on the image (background unaltered), and print the number of detected objects."
    )
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--output_png", type=str, required=True,
                        help="Path where the output PNG file (overlay) will be saved.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="(Optional) Threshold for converting masks to binary (if needed).")
    args = parser.parse_args()
    main(args)

