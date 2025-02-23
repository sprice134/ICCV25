#!/usr/bin/env python3
import os
import pickle
import time
import uuid
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
import sys

sys.path.append('../')  # Adjust if your sam_helper and tools modules are elsewhere

from sam_helper import load_sam_model, select_point_placement, prepare_mask_for_sam
from tools import combine_masks_16bit  # Not used here, but imported in case you need it

def compute_iou(pred_mask, gt_mask):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

def run_sam_inference(
    predictor,
    loop_image,
    listOfPolygons,
    listOfBoxes,
    listOfMasks,
    image_width,
    image_height,
    num_points=4,
    dropout_percentage=0,
    ignore_border_percentage=5,
    algorithm="Voronoi",
    use_box_input=True,
    use_mask_input=False,
    box_expansion_rate=0.0,
    mask_expansion_rate=0.0,
    return_pois=False
):
    """
    Runs SAM segmentation refinement on predicted masks.
    Optionally pass an initial mask to SAM (mask_input) or expand bounding boxes.
    Returns a list of SAM-refined masks and, if requested, the selected points for each mask.
    """
    sam_masks_list = []
    pois_list = []  # To store the selected points (POIs) for each mask
    boxes_list = []  # To store the refined bounding boxes used in prediction

    # Adjust algorithm naming if needed.
    if algorithm == "Distance Max":
        algorithm = "Hill Climbing"

    def expand_bbox_within_border(x1, y1, x2, y2, width, height, expansion_rate=0.0):
        if expansion_rate <= 0:
            return [x1, y1, x2, y2]
        original_w = x2 - x1
        original_h = y2 - y1
        new_w = original_w * expansion_rate
        new_h = original_h * expansion_rate
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        new_x1 = max(0, center_x - new_w / 2)
        new_y1 = max(0, center_y - new_h / 2)
        new_x2 = min(width, center_x + new_w / 2)
        new_y2 = min(height, center_y + new_h / 2)
        return [new_x1, new_y1, new_x2, new_y2]

    def adjust_mask_area(mask, target_percentage, max_iterations=50, kernel_size=(3,3)):
        if target_percentage == 0 or target_percentage == 100:
            return mask
        operation = 'erode' if target_percentage < 100 else 'dilate'
        binary_mask = (mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        original_area = np.sum(binary_mask)
        target_area = original_area * (target_percentage / 100.0)
        new_mask = binary_mask.copy()
        for i in range(max_iterations):
            current_area = np.sum(new_mask)
            if (operation == 'erode' and current_area <= target_area) or \
               (operation == 'dilate' and current_area >= target_area):
                break
            if operation == 'erode':
                new_mask = cv2.erode(new_mask, kernel, iterations=1)
            else:
                new_mask = cv2.dilate(new_mask, kernel, iterations=1)
        return new_mask

    for index in range(len(listOfPolygons)):
        box = listOfBoxes[index]
        box = box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box)

        # Use bounding box if specified.
        if use_box_input:
            x1, y1, x2, y2 = expand_bbox_within_border(
                box[0], box[1], box[2], box[3],
                image_width, image_height,
                expansion_rate=box_expansion_rate
            )
            box = np.array([x1, y1, x2, y2])
        else:
            box = None

        mask = listOfMasks[index]
        if np.sum(mask) > num_points * num_points:
            try:
                if algorithm == "Hill Climbing" and num_points != 1:
                    selected_points, _, _ = select_point_placement(
                        mask=mask,
                        num_points=num_points,
                        dropout_percentage=dropout_percentage,
                        ignore_border_percentage=ignore_border_percentage,
                        algorithm=algorithm,
                        select_perimeter=True
                    )
                else:
                    selected_points, _, _ = select_point_placement(
                        mask=mask,
                        num_points=num_points,
                        dropout_percentage=dropout_percentage,
                        ignore_border_percentage=ignore_border_percentage,
                        algorithm=algorithm,
                        select_perimeter=False
                    )
            except Exception as e:
                print(f"Error selecting points: {e}")
                continue  # Skip this mask if point selection fails

            # Adjust mask area if required.
            mask = adjust_mask_area(mask, mask_expansion_rate)
            
            # Swap the order from (y, x) to (x, y)
            op_y, op_x = zip(*selected_points)
            input_point = np.array(list(zip(op_x, op_y)))
            input_label = np.array([1] * len(input_point))
            
            predictor.set_image(loop_image)
            mask_input = prepare_mask_for_sam(mask) if use_mask_input else None

            predict_kwargs = {
                'point_coords': input_point,
                'point_labels': input_label,
                'multimask_output': True
            }
            if use_box_input and box is not None:
                x1, y1, x2, y2 = box
                if (box_expansion_rate > 0.0) or (x2 - x1 > 0 and y2 - y1 > 0):
                    predict_kwargs['box'] = box[None, :]
            if mask_input is not None:
                predict_kwargs['mask_input'] = mask_input

            try:
                masks_out, scores, logits = predictor.predict(**predict_kwargs)
                sam_masks_list.append(masks_out[0])  # Use the first output mask
                pois_list.append(input_point)
                boxes_list.append(box)
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue
        else:
            print(f"Skipping Mask Because Area ({np.sum(mask)}) Smaller than number of points^2 ({num_points * num_points})")

    if return_pois:
        return sam_masks_list, pois_list, boxes_list
    return sam_masks_list

def main():
    # -------------------------------------------------------------------------
    # Hardcoded parameters and paths
    # -------------------------------------------------------------------------
    image_path = "/home/sprice/ICCV25/datasets/powder/test/Cu-Ni-Powder_250x_10_SE_png.rf.cd93ec4589ad8f4e412cb1ec0e805016.jpg"
    inference_pickle_path = "/home/sprice/ICCV25/savedInference/particle_yolov8n_inference.pkl"
    output_dir = "single_masks"
    os.makedirs(output_dir, exist_ok=True)
    
    sam_model_type = "vit_l"
    sam_checkpoint = "/home/sprice/ICCV25/modelWeights/sam_vit_l.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam_params = {
        "num_points": 3,
        "algorithm": "Voronoi",
        "ignore_border_percentage": 0,
        "use_box_input": True,
        "use_mask_input": False,
    }
    box_expansion_rate = 1.025
    mask_expansion_rate = 1.0

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

    # Convert image to a NumPy array (BGR order expected by SAM)
    loop_image = np.array(pil_image)[:, :, ::-1].copy()

    # Convert masks to binary (uint8) format
    bin_masks = []
    for mask in masks:
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        bin_mask = (mask > 127).astype(np.uint8) if mask.dtype != bool else mask.astype(np.uint8)
        bin_masks.append(bin_mask)

    # -------------------------------------------------------------------------
    # Run SAM inference to refine masks and return POIs and refined boxes
    # -------------------------------------------------------------------------
    print("Running SAM inference to refine masks...")
    start_time = time.time()
    refined_masks, pois_per_mask, refined_boxes = run_sam_inference(
        predictor=sam_predictor,
        loop_image=loop_image,
        listOfPolygons=polygons,
        listOfBoxes=boxes,
        listOfMasks=bin_masks,
        image_width=pil_image.width,
        image_height=pil_image.height,
        num_points=sam_params["num_points"],
        dropout_percentage=0,
        ignore_border_percentage=sam_params["ignore_border_percentage"],
        algorithm=sam_params["algorithm"],
        use_box_input=sam_params["use_box_input"],
        use_mask_input=sam_params["use_mask_input"],
        box_expansion_rate=box_expansion_rate,
        mask_expansion_rate=mask_expansion_rate,
        return_pois=True
    )
    end_time = time.time()
    print(f"SAM inference completed in {end_time - start_time:.2f} seconds.")

    # -------------------------------------------------------------------------
    # Load the GT mask image and extract GT instance masks
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
    gt_ids = [gt_id for gt_id in gt_ids if gt_id != 0]

    # -------------------------------------------------------------------------
    # For each refined mask, compute IoU with GT instances and save visualizations
    # -------------------------------------------------------------------------
    unique_id = f"{int(time.time())}_{uuid.uuid4().hex}"
    for idx, mask in enumerate(refined_masks):
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
        for gt_id in gt_ids:
            gt_instance_mask = (gt_mask_arr == gt_id).astype(np.uint8)
            iou = compute_iou(pred_binary, gt_instance_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
                best_gt_mask = gt_instance_mask * 255

        label_str = ""
        if labels and idx < len(labels):
            label_str = str(labels[idx])
        
        # -------------------------------
        # Create overlay on the refined mask
        # -------------------------------
        base_img = Image.fromarray(mask).convert("RGB").convert("RGBA")
        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Only overlay the initial prediction mask if a mask was provided to SAM.
        if sam_params["use_mask_input"]:
            init_mask = bin_masks[idx]
            init_mask_img = Image.fromarray((init_mask * 255).astype(np.uint8))
            yellow_img = Image.new("RGBA", base_img.size, (255, 165, 0, 100))
            overlay.paste(yellow_img, mask=init_mask_img)

        # Overlay the POIs in red.
        pois = pois_per_mask[idx] if pois_per_mask and idx < len(pois_per_mask) else []
        if len(pois) > 0:
            print(f"Mask {idx} POIs: {pois}")  # Debugging output
            for poi in pois:
                try:
                    x, y = int(poi[0]), int(poi[1])
                    r = 3  # Adjust radius as needed
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0, 255))
                except Exception as e:
                    print(f"Error drawing POI {poi} for mask {idx}: {e}")

        # Overlay the refined bounding box in blue, if available.
        if refined_boxes and idx < len(refined_boxes):
            refined_box = refined_boxes[idx]
            if refined_box is not None:
                refined_box = list(refined_box)
                draw.rectangle(refined_box, outline=(0, 0, 255, 255), width=2)

        # Composite the overlay on top of the base image
        final_img = Image.alpha_composite(base_img, overlay)

        # Save the refined mask with overlay
        pred_mask_filename = f"{os.path.splitext(image_filename)[0]}_mask_{idx}_{unique_id}_{label_str}_iou_{best_iou:.5f}.png"
        pred_mask_path = os.path.join(output_dir, pred_mask_filename)
        final_img.convert("RGB").save(pred_mask_path)
        print(f"Saved predicted mask {idx} to: {pred_mask_path} with IoU: {best_iou:.5f}")
        
        # Save the corresponding GT mask if a match was found
        if best_gt_mask is not None:
            gt_save_filename = f"{os.path.splitext(image_filename)[0]}_mask_{idx}_{unique_id}_{label_str}_gt_{best_gt_id}.png"
            gt_save_path = os.path.join(output_dir, gt_save_filename)
            Image.fromarray(best_gt_mask.astype(np.uint8)).save(gt_save_path)
            print(f"Saved GT mask for predicted mask {idx} to: {gt_save_path}")

if __name__ == "__main__":
    main()
