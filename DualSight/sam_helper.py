import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
from pointSelection import select_furthest_points_from_mask
import torch.nn.functional as F

def load_sam_model(sam_checkpoint, model_type="vit_l", device="cuda"):
    """
    Initialize and return a SAM predictor.
    """
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def prepare_mask_for_sam(mask, target_size=(256, 256)):
    """
    Resize a 2D binary mask to a desired shape for passing to SAM as mask_input.
    Expects mask in [0, 255] or [0,1]. Returns a torch.FloatTensor of shape (1, H, W).
    """
    if len(mask.shape) != 2:
        raise ValueError("Mask must be a 2D array (H, W).")

    # Normalize if mask has values > 1
    if mask.max() > 1:
        mask = mask / 255.0

    mask_tensor = torch.tensor(mask, dtype=torch.float32)[None, None, :, :]  # (1,1,H,W)
    mask_resized = F.interpolate(
        mask_tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)  # -> (1, target_size[0], target_size[1])
    return mask_resized

def run_sam_inference(predictor,
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
                      use_mask_input=False,
                      box_expansion_rate=0.0):
    """
    Runs SAM segmentation refinement on a list of YOLO (or other) polygons/boxes/masks.
    Optionally passes a pre-existing mask to SAM (mask_input).
    Returns a list of SAM-refined masks (numpy arrays of shape [H, W]).
    
    Args:
        use_mask_input (bool): If True, pass a rescaled version of the initial mask to SAM as mask_input.
        box_expansion_rate (float): If nonzero, expand the bounding box before passing to SAM.
    """
    sam_masks_list = []

    def expand_bbox_within_border(x1, y1, x2, y2, width, height, expansion_rate=0.0):
        """
        Stub function to expand a bounding box if needed.
        Currently just returns the original box if expansion_rate=0.0.
        """
        if expansion_rate <= 0:
            return [x1, y1, x2, y2]
        # Example expansion logic: (very simplistic)
        box_w = x2 - x1
        box_h = y2 - y1
        expand_w = box_w * expansion_rate
        expand_h = box_h * expansion_rate
        new_x1 = max(0, x1 - expand_w/2)
        new_y1 = max(0, y1 - expand_h/2)
        new_x2 = min(width, x2 + expand_w/2)
        new_y2 = min(height, y2 + expand_h/2)
        return [new_x1, new_y1, new_x2, new_y2]

    for index in range(len(listOfPolygons)):
        box = listOfBoxes[index]
        # Convert box from torch.Tensor to NumPy if needed
        box = box.cpu().numpy() if isinstance(box, torch.Tensor) else np.array(box)

        # Expand bounding box if needed
        x1, y1, x2, y2 = expand_bbox_within_border(
            box[0], box[1], box[2], box[3],
            image_width, image_height,
            expansion_rate=box_expansion_rate
        )
        box = np.array([x1, y1, x2, y2])

        # Select points within the mask
        mask = listOfMasks[index]
        selected_points, _, _ = select_furthest_points_from_mask(
            mask=mask,
            num_points=num_points,
            dropout_percentage=dropout_percentage,
            ignore_border_percentage=ignore_border_percentage,
            algorithm=algorithm
        )

        op_y, op_x = zip(*selected_points)  # row, col -> (y, x)
        predictor.set_image(loop_image)

        input_point = np.array(list(zip(op_x, op_y)))
        input_label = np.array([1] * len(input_point))

        # Optionally prepare the initial mask for input to SAM
        if use_mask_input:
            mask_input = prepare_mask_for_sam(mask)
        else:
            mask_input = None

        # Construct kwargs for predictor
        predict_kwargs = {
            'point_coords': input_point,
            'point_labels': input_label,
            'multimask_output': True,
        }

        # If box is meaningful, pass it
        if box_expansion_rate > 0.0 or (x2 - x1) > 0:
            predict_kwargs['box'] = box[None, :]

        # If we want to pass a mask_input
        if mask_input is not None:
            predict_kwargs['mask_input'] = mask_input

        masks, scores, logits = predictor.predict(**predict_kwargs)

        # For simplicity, pick the first mask
        mask_out = masks[0]
        sam_masks_list.append(mask_out)

    return sam_masks_list

def combine_masks(masks_list, output_mask_path):
    """
    Combines multiple boolean masks into a single grayscale mask,
    assigning a unique ID (1-based) to each object.
    Saves the result and returns the NumPy array of the combined mask.
    """
    if not masks_list:
        return None

    height, width = masks_list[0].shape
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for idx, mask in enumerate(masks_list, start=1):
        combined_mask[mask > 0] = idx

    cv2.imwrite(output_mask_path, combined_mask)
    print(f"Combined mask saved to {output_mask_path}")

    return combined_mask
