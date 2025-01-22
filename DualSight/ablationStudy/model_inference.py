"""
model_inference.py

Usage Example:
    python model_inference.py \
    --model "yolov8n-seg" \
    --model-type "yolo" \
    --model-path "/home/sprice/ICCV25/modelWeights/yolov8n-seg.pt" \
    --image-dir "/home/sprice/ICCV25/demo.v7i.coco/test" \
    --output-file "inference_outputs/yolov8n_inference.pkl" \
    --device "cuda"

    # For YOLOv8 XL
    python model_inference.py \
        --model "yolov8x-seg" \
        --model-type "yolo" \
        --model-path "/home/sprice/ICCV25/modelWeights/yolov8x-seg.pt" \
        --image-dir "/home/sprice/ICCV25/demo.v7i.coco/test" \
        --output-file "inference_outputs/yolov8x_inference.pkl" \
        --device "cuda"

    # For Mask R-CNN
    python model_inference.py \
        --model "maskrcnn" \
        --model-type "maskrcnn" \
        --model-path "/home/sprice/ICCV25/modelWeights/final_mask_rcnn_model.pth" \
        --image-dir "/home/sprice/ICCV25/demo.v7i.coco/test" \
        --output-file "inference_outputs/maskrcnn_inference.pkl" \
        --device "cuda"
"""

import os
import cv2
import glob
import pickle
import torch
import argparse
import numpy as np
import sys 
sys.path.append('../')
# Replace these imports with your actual modules
from models import load_trained_model, get_inference_predictions

def run_inference_for_model(model_type, model_name, model_path, image_dir, output_file, device="cuda"):
    """
    Runs inference for a single model on all .jpg images in `image_dir`,
    then saves a pickle with bounding boxes, polygons, and masks.
    This file is unique to the model, ensuring no interference with other models.
    """

    # 1) Gather images
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not image_paths:
        print(f"[ERROR] No .jpg images found in {image_dir}")
        return

    # 2) Load the model
    print(f"[INFO] Loading model: {model_name} ({model_type}) from {model_path}")
    model = load_trained_model(model_type, model_path, device=device)

    # 3) Create a dictionary to store results for each image
    #    inference_data = { "image1.jpg": {...}, "image2.jpg": {...}, ... }
    inference_data = {}

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[WARNING] Could not read {img_path}. Skipping...")
            continue

        # 4) Run the inference
        #    Returns polygons, boxes, and mask arrays (or Tensors).
        listOfPolygons, listOfBoxes, listOfMasks = get_inference_predictions(
            model=model,
            model_type=model_type,
            image_path=img_path,
            device=device
        )

        # 5) Convert to CPU / NumPy if you prefer. Pickle can handle tensors, but
        #    if you want them guaranteed CPU arrays, do something like:
        if torch.is_tensor(listOfBoxes):
            listOfBoxes = listOfBoxes.cpu().numpy()

        # 6) Store them in a dictionary entry for this image
        inference_data[img_name] = {
            "polygons": listOfPolygons,
            "boxes": listOfBoxes,
            "masks": listOfMasks
        }

    # 7) Save everything to a unique pickle for this model
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(inference_data, f)

    print(f"[INFO] Inference completed for {model_name}. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for a single model and save pickle.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. 'yolov8n')")
    parser.add_argument("--model-type", type=str, default="yolo", help="Model type (e.g. 'yolo' or 'maskrcnn')")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of .jpg images")
    parser.add_argument("--output-file", type=str, default="inference_outputs/model_inference.pkl",
                        help="Where to save the pickle file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")

    args = parser.parse_args()

    run_inference_for_model(
        model_type=args.model_type,
        model_name=args.model,
        model_path=args.model_path,
        image_dir=args.image_dir,
        output_file=args.output_file,
        device=args.device
    )
