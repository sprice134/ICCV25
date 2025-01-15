import torch
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

def load_yolo_model(yolo_model_path):
    """
    Load a YOLO model from the given path.
    Returns a YOLO model object.
    """
    model = YOLO(yolo_model_path)
    return model

def get_yolo_predictions(model, image_path, min_mask_area=200):
    """
    Run YOLO inference on the given image.
    Returns:
        listOfPolygons, listOfBoxes, listOfMasks, yolo_masks_list
    """
    image = Image.open(image_path)
    results = model(image)

    listOfPolygons = []
    listOfBoxes = []
    listOfMasks = []
    yolo_masks_list = []

    # Assuming only one set of results for a single image
    for result in results[0]:
        mask = result.masks.data
        # If the mask is large enough, proceed
        if mask.sum() > min_mask_area:
            # Retrieve polygon points
            polygon_points = result.masks.xy[0]
            if len(polygon_points) > 2:
                listOfPolygons.append(polygon_points)
                # Store bounding boxes
                listOfBoxes.append(result.boxes.xyxy[0])

                # Create a binary mask with Pillow
                mask_image = Image.new('L', image.size, 0)
                draw = ImageDraw.Draw(mask_image)
                draw.polygon(polygon_points, outline=1, fill=1)
                mask_array = np.array(mask_image) * 255

                listOfMasks.append(mask_array)
                yolo_masks_list.append(mask_array)

    return listOfPolygons, listOfBoxes, listOfMasks, yolo_masks_list
