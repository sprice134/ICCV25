#!/usr/bin/env python3

import os
import sys
import numpy as np
from PIL import Image

# 1) YOLOv8 import
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics (pip install ultralytics).")
    sys.exit(1)

import torch
import subprocess

# We'll assume `instance_segfix.py` is in the same directory:
INSTANCE_SEGFIX_SCRIPT = os.path.join(os.path.dirname(__file__), "instance_segfix.py")

def run_yolov8_inference(image_path, model_path="yolov8n-seg.pt"):
    """
    Runs YOLOv8 instance segmentation on a single image.
    Returns:
      - masks: a list of (H, W) NumPy arrays (0/1) for each detected instance
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.25)  # conf threshold example

    # YOLOv8 results object can contain multiple images; we assume just 1 here
    # results[0].masks.data is a list of torch Tensors, each shape = (H, W)
    yolo_masks = results[0].masks.data if hasattr(results[0].masks, 'data') else []

    masks = []
    for mask_tensor in yolo_masks:
        # Convert from torch Tensor to binary NumPy
        mask_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
        masks.append(mask_np)

    return masks

def save_masks_and_txt(masks, base_name, output_dir):
    """
    Saves each instance mask as <base_name>_0.png, <base_name>_1.png, ...
    Then writes <base_name>_pred.txt containing the list of mask filenames.

    The instance_segfix.py script expects a text file that references one or more
    mask images, one per line. We’ll place them in the same folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_filename = os.path.join(output_dir, f"{base_name}_pred.txt")

    with open(txt_filename, 'w') as f:
        for i, mask in enumerate(masks):
            mask_filename = f"{base_name}_{i}.png"
            mask_filepath = os.path.join(output_dir, mask_filename)
            Image.fromarray(mask * 255).save(mask_filepath)
            # Write this mask name to the text file
            f.write(mask_filename + "\n")

    return txt_filename

def run_segfix(in_dir, offset_dir, out_dir, scale=1.0, eval_only=False):
    """
    Calls instance_segfix.py with the provided arguments via subprocess.
    """
    cmd = [
        "python", INSTANCE_SEGFIX_SCRIPT,
        "--input", in_dir,
        "--offset", offset_dir,
        "--out", out_dir,
        "--scale", str(scale),
    ]
    if eval_only:
        cmd.append("--eval_only")

    print("Running SegFix with command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    # Example usage on one image
    image_path = "images/my_image.jpg"         # your input image
    offset_dir = "offsets"                     # directory containing .mat offsets
    pred_dir = "predictions"                   # directory for YOLO masks + text files
    segfix_output = "segfix_output"            # final post-processed masks

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # e.g. my_image

    print(f"Running YOLOv8 on {image_path} ...")
    masks = run_yolov8_inference(image_path, model_path="yolov8n-seg.pt")
    if not masks:
        print("No instances found by YOLOv8.")
        return

    print(f"Saving YOLOv8 instance masks and .txt file in {pred_dir} ...")
    txt_filename = save_masks_and_txt(masks, base_name, pred_dir)
    print(f"Created: {txt_filename}")

    # Now call SegFix:
    # The instance_segfix.py script expects:
    #   - in_dir: folder with <some>_pred.txt files
    #   - offset_dir: folder with .mat offset files (matching base name)
    #   - out_dir: where to put the post-processed masks
    #
    # We must ensure offset_dir has base_name+".mat" file, e.g. "my_image.mat"
    offset_path = os.path.join(offset_dir, f"{base_name}.mat")
    if not os.path.isfile(offset_path):
        print(f"[WARNING] Offset file not found: {offset_path}")
        print("SegFix will fail unless you have an offset .mat file named the same as your image.")
        return

    print("Running SegFix post-processing...")
    run_segfix(in_dir=pred_dir, offset_dir=offset_dir, out_dir=segfix_output, scale=1.0)
    # scale=1.0 is an example. Adjust as needed.

    print("Done. Check segfix_output/ for the corrected instance masks.")

if __name__ == "__main__":
    main()
