#!/usr/bin/env python3
import os
import pickle
import random
import numpy as np
from PIL import Image
import torch
import argparse
import colorsys  # For converting HSV to RGB

# Import the SAM model helpers – adjust the path if necessary.
from sam_helper import load_sam_model, run_sam_inference

def generate_neon_colors(n=256):
    """
    Generate a list of n bright, neon-style RGB colors by sampling random hues
    at full (or near-full) saturation and value.
    """
    random.seed(42)  # for reproducibility
    colors = []
    for _ in range(n):
        # Random hue [0, 1), full saturation & brightness for neon look.
        hue = random.random()
        saturation = 1.0
        value = 1.0

        # Convert HSV to RGB in [0,1] range.
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        # Scale up to 0–255 and convert to int.
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        colors.append((r, g, b))
    
    return colors

def threshold_mask(mask):
    """
    Convert a mask to a binary mask using an appropriate threshold.
    If the mask values are in [0, 1], use 0.5; if in [0, 255], use 127.
    """
    if mask.dtype != bool:
        if mask.max() <= 1.0:
            return (mask > 0.5)
        else:
            return (mask > 127)
    return mask

def overlay_masks_on_image(image, masks, colors, alpha=0.5):
    """
    Overlay a list of masks (with corresponding colors) on the original image.
    - image: PIL.Image (RGB)
    - masks: list of binary numpy arrays (H, W)
    - colors: list of (R, G, B) tuples (should match the order of masks)
    - alpha: blending factor (0.0 to 1.0)
    Returns a new PIL.Image with the overlays.
    """
    image_np = np.array(image).copy()
    overlay = image_np.copy()

    for mask, color in zip(masks, colors):
        bin_mask = threshold_mask(mask)
        color_layer = np.zeros_like(image_np, dtype=np.uint8)
        color_layer[bin_mask] = color
        overlay[bin_mask] = (alpha * color_layer[bin_mask] + (1 - alpha) * overlay[bin_mask]).astype(np.uint8)
    return Image.fromarray(overlay)

def process_inference_file(inf_file, image_name, pil_image, loop_image, random_colors, sam_predictor, sam_params, output_dir):
    """
    Process a single inference file. For the target image, generate and save two overlays:
    one for unrefined masks and one for refined masks.
    """
    with open(inf_file, "rb") as f:
        inference_data = pickle.load(f)
    
    if image_name not in inference_data:
        print(f"[INFO] Image '{image_name}' not found in '{inf_file}'.")
        return

    result = inference_data[image_name]
    masks = result.get("masks", [])
    boxes = result.get("boxes", [])
    polygons = result.get("polygons", [])
    
    print(f"[INFO] Found {len(masks)} masks in '{inf_file}' for image '{image_name}'.")

    unrefined_masks = []
    unrefined_colors = []
    refined_masks_list = []
    refined_colors = []

    for idx, mask in enumerate(masks):
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        bin_mask = threshold_mask(mask)
        color = random_colors[idx % len(random_colors)]
        unrefined_masks.append(bin_mask)
        unrefined_colors.append(color)
    
    # Generate overlay for unrefined masks
    unrefined_overlay = overlay_masks_on_image(pil_image, unrefined_masks, unrefined_colors, alpha=0.5)
    base_name = os.path.splitext(image_name)[0]
    inf_id = os.path.splitext(os.path.basename(inf_file))[0]
    unrefined_overlay_filename = f"{base_name}_{inf_id}_unrefined_overlay.png"
    unrefined_overlay_path = os.path.join(output_dir, unrefined_overlay_filename)
    unrefined_overlay.save(unrefined_overlay_path)
    print(f"[INFO] Saved unrefined overlay image: {unrefined_overlay_path}")
    
    # Run SAM refinement.
    refined_masks = run_sam_inference(
        predictor=sam_predictor,
        loop_image=loop_image,
        listOfPolygons=polygons,
        listOfBoxes=boxes,
        listOfMasks=masks,
        image_width=pil_image.width,
        image_height=pil_image.height,
        num_points=sam_params.get("num_points", 1),
        dropout_percentage=sam_params.get("dropout_percentage", 0),
        ignore_border_percentage=sam_params.get("ignore_border_percentage", 0.0),
        algorithm=sam_params.get("algorithm", "default"),
        use_box_input=sam_params.get("use_box_input", True),
        use_mask_input=sam_params.get("use_mask_input", True),
        box_expansion_rate=sam_params.get("box_expansion_rate", 0.0),
        mask_expansion_rate=sam_params.get("mask_expansion_rate", 0.0)
    )
    
    if len(refined_masks) != len(masks):
        print(f"[WARNING] Number of refined masks ({len(refined_masks)}) does not match original ({len(masks)}) in '{inf_file}'.")
    
    for idx, refined_mask in enumerate(refined_masks):
        if torch.is_tensor(refined_mask):
            refined_mask = refined_mask.cpu().numpy()
        refined_bin_mask = threshold_mask(refined_mask)
        color = random_colors[idx % len(random_colors)]
        refined_masks_list.append(refined_bin_mask)
        refined_colors.append(color)
    
    # Generate overlay for refined masks.
    refined_overlay = overlay_masks_on_image(pil_image, refined_masks_list, refined_colors, alpha=0.5)
    refined_overlay_filename = f"{base_name}_{inf_id}_refined_overlay.png"
    refined_overlay_path = os.path.join(output_dir, refined_overlay_filename)
    refined_overlay.save(refined_overlay_path)
    print(f"[INFO] Saved refined overlay image: {refined_overlay_path}")

def refine_and_save_masks(inference_files, image_name, output_dir, image_dir, sam_model_params, sam_params):
    """
    For each inference file, if the provided image name is found, process the masks
    separately and generate two overlay images (unrefined and refined).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original image.
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f"[ERROR] Original image '{image_path}' not found.")
        return
    pil_image = Image.open(image_path).convert("RGB")
    loop_image = np.array(pil_image)[:, :, ::-1].copy()  # For SAM inference (BGR)

    # Load SAM model once.
    sam_predictor = load_sam_model(
        sam_checkpoint=sam_model_params["checkpoint"],
        model_type=sam_model_params["model_type"],
        device=sam_model_params["device"]
    )
    
    # Use the new neon color generator
    random_colors = generate_neon_colors(256)
    
    for inf_file in inference_files:
        if not os.path.exists(inf_file):
            print(f"[WARNING] Inference file '{inf_file}' not found. Skipping.")
            continue
        process_inference_file(inf_file, image_name, pil_image, loop_image, random_colors, sam_predictor, sam_params, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For a given image, load saved inference results from multiple files and create separate overlay images for each."
    )
    parser.add_argument("--inference_files", nargs="+", required=True,
                        help="List of paths to inference pickle files.")
    parser.add_argument("--image_name", type=str, required=True,
                        help="Name of the image to process (e.g., sample.jpg).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output overlay images.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory where the original image is located.")
    # SAM model parameters:
    parser.add_argument("--sam_checkpoint", type=str, required=True,
                        help="Path to the SAM model checkpoint file.")
    parser.add_argument("--sam_model_type", type=str, default="vit_l",
                        help="Type of SAM model (e.g., vit_l).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (e.g., cuda or cpu).")
    # Additional SAM inference parameters:
    parser.add_argument("--num_points", type=int, default=1, help="Number of points for SAM inference.")
    parser.add_argument("--dropout_percentage", type=float, default=0.0, help="Dropout percentage for SAM inference.")
    parser.add_argument("--ignore_border_percentage", type=float, default=0.0,
                        help="Ignore border percentage for SAM inference.")
    parser.add_argument("--algorithm", type=str, default="default", help="SAM algorithm to use.")
    parser.add_argument("--use_box_input", action="store_true", help="Use box input for SAM inference.")
    parser.add_argument("--use_mask_input", action="store_true", help="Use mask input for SAM inference.")
    parser.add_argument("--box_expansion_rate", type=float, default=0.0, help="Box expansion rate for SAM inference.")
    parser.add_argument("--mask_expansion_rate", type=float, default=0.0, help="Mask expansion rate for SAM inference.")
    
    args = parser.parse_args()
    
    sam_model_params = {
        "checkpoint": args.sam_checkpoint,
        "model_type": args.sam_model_type,
        "device": args.device
    }
    
    sam_params = {
        "num_points": args.num_points,
        "dropout_percentage": args.dropout_percentage,
        "ignore_border_percentage": args.ignore_border_percentage,
        "algorithm": args.algorithm,
        "use_box_input": args.use_box_input,
        "use_mask_input": args.use_mask_input,
        "box_expansion_rate": args.box_expansion_rate,
        "mask_expansion_rate": args.mask_expansion_rate,
    }
    
    refine_and_save_masks(
        inference_files=args.inference_files,
        image_name=args.image_name,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        sam_model_params=sam_model_params,
        sam_params=sam_params
    )


    '''
    python maskSaving_ColorMatching.py \
    --inference_files /home/sprice/ICCV25/savedInference/particle_yolov8n_inference.pkl /home/sprice/ICCV25/savedInference/particle_yolov8x_inference.pkl /home/sprice/ICCV25/savedInference/particle_maskrcnn_inference.pkl /home/sprice/ICCV25/savedInference/particle_mask2former_inference.pkl\
    --image_name S04_01_SE1_1000X45_png.rf.baa89207016e4da58f6ec0ab4f2b008f.jpg \
    --output_dir colorMatch/ \
    --image_dir /home/sprice/ICCV25/datasets/powder/test \
    --sam_checkpoint /home/sprice/ICCV25/modelWeights/sam_vit_l.pth \
    --sam_model_type vit_l \
    --device cuda \
    --num_points 3 \
    --ignore_border_percentage 10.0 \
    --algorithm Voronoi \
    --use_box_input

    '''
