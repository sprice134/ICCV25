import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as FT
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
from ultralytics import YOLO
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    faster_rcnn,
    mask_rcnn
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models import mobilenet_v3_large
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

############################################################################
#   1) Custom MobileNetV3 segmentation model
############################################################################

class MobileNetV3Segmentation(nn.Module):
    def __init__(self, num_classes):
        """
        MobileNetV3-based segmentation model.

        Args:
            num_classes (int): Number of classes (including background).
        """
        super().__init__()
        # Backbone
        self.backbone = mobilenet_v3_large(pretrained=True).features

        # Decoder and segmentation head
        self.decoder = nn.Sequential(
            nn.Conv2d(960, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        """
        features = self.backbone(x)
        out = self.decoder(features)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


############################################################################
#   2) Load different trained models
############################################################################

def load_yolo_model(yolo_model_path):
    """
    Load a YOLO model from the given path.
    Returns a YOLO model object.
    """
    return YOLO(yolo_model_path)

def load_maskrcnn_model(model_path='/home/sprice/RQ/maskrcnn/final_mask_rcnn_model.pth', 
                        num_classes=2,
                        backbone="resnet50",
                        device="cuda"):  # Added device parameter
    """
    Load and return a trained Mask R-CNN model with specified parameters.
    """
    if backbone == "resnet50":
        model = maskrcnn_resnet50_fpn(pretrained=False)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Modify the classifier head for the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Modify the mask head for the number of classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=hidden_layer,
        num_classes=num_classes,
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move model to the specified device
    model.eval()
    return model

def load_mobilenetv3_model(model_path, num_classes=2, device="cuda"):
    """
    Load a trained MobileNetV3 segmentation model.
    """
    model = MobileNetV3Segmentation(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_mask2former_model(checkpoint, device="cuda"):
    """
    Load a Mask2Former model and its processor.
    
    Args:
        checkpoint (str): Path to the trained Mask2Former checkpoint.
        device (str): Device to load the model onto.
        
    Returns:
        model: The Mask2FormerForUniversalSegmentation model.
        processor: The corresponding AutoImageProcessor.
    """
    # Use the default pretrained weights as a starting point.
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-instance")
    
    # If a checkpoint is provided, load its state dict.
    if os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded Mask2Former checkpoint from {checkpoint}")
    else:
        print(f"[WARNING] Checkpoint {checkpoint} not found. Using default pretrained weights.")

    model.to(device)
    model.eval()
    return model, processor


def load_trained_model(model_type, model_path, device="cuda"):
    model_type = model_type.lower()
    if model_type == 'yolo':
        return load_yolo_model(model_path)
    elif model_type == 'maskrcnn':
        model = load_maskrcnn_model(model_path=model_path, device=device)
        model.eval()
        return model
    elif model_type == 'mobilenetv3':
        return load_mobilenetv3_model(model_path=model_path, device=device)
    elif model_type == 'mask2former':
        # For Mask2Former we return a tuple (model, processor)
        return load_mask2former_model(checkpoint=model_path, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



############################################################################
#   3) Inference functions for each model (return polygons, boxes, masks)
############################################################################

def get_yolo_predictions(model, image_path, min_mask_area=0):
    """
    Run YOLO inference on the given image_path.
    Returns:
        listOfPolygons, listOfBoxes, listOfMasks
    """
    image = Image.open(image_path).convert("RGBA")
    results = model(image)

    listOfPolygons = []
    listOfBoxes = []
    listOfMasks = []

    # YOLO v8 results[0] is a list of detections
    for det in results[0]:
        mask = det.masks.data
        if mask.sum() > min_mask_area:
            polygon_points = det.masks.xy[0]
            if len(polygon_points) > 2:
                listOfPolygons.append(polygon_points)
                listOfBoxes.append(det.boxes.xyxy[0])

                # Create a binary mask with Pillow
                mask_image = Image.new('L', image.size, 0)
                draw = ImageDraw.Draw(mask_image)
                draw.polygon(polygon_points, outline=1, fill=1)
                mask_array = np.array(mask_image) * 255

                listOfMasks.append(mask_array)

    return listOfPolygons, listOfBoxes, listOfMasks

def get_maskrcnn_predictions(model, image_path, device="cuda"):
    """
    Generate Mask R-CNN predictions on an image path.
    Returns (listOfPolygons, listOfBoxes, listOfMasks).
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = FT.to_tensor(image).unsqueeze(0).to(device)

    # Generate predictions
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    listOfPolygons = []
    listOfBoxes = []
    listOfMasks = []
    score_threshold = 0.5

    for i, score in enumerate(predictions['scores']):
        if score >= score_threshold:
            mask = predictions['masks'][i, 0].cpu().numpy()
            box = predictions['boxes'][i].cpu().numpy()

            # Filter out small masks
            if mask.sum() > 200:
                binary_mask = (mask > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    polygon_points = largest_contour.squeeze(axis=1).tolist()
                    if len(polygon_points) > 2:
                        listOfPolygons.append(polygon_points)
                        listOfBoxes.append(box)

                        # Convert polygon to binary mask
                        mask_image = Image.new('L', image.size, 0)
                        draw = ImageDraw.Draw(mask_image)
                        polygon_points_int = [(int(x), int(y)) for x, y in polygon_points]
                        draw.polygon(polygon_points_int, outline=1, fill=1)
                        mask_array = np.array(mask_image) * 255
                        listOfMasks.append(mask_array)

    return listOfPolygons, listOfBoxes, listOfMasks

def get_mobilenetv3_predictions(model, image_path, device="cuda", image_size=(1024, 768)):
    """
    Generate predictions using a MobileNetV3Segmentation model.
    Returns (listOfPolygons, listOfBoxes, listOfMasks).
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Resize for inference
    image = image.resize(image_size)
    image_tensor = FT.to_tensor(image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)
    predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Resize predicted mask back to the original size
    predicted_mask_resized = cv2.resize(
        predicted_mask.astype(np.uint8),
        original_size,
        interpolation=cv2.INTER_NEAREST
    )

    # Label connected components
    labeled_mask, object_count = cv2.connectedComponents(predicted_mask_resized.astype(np.uint8))

    listOfPolygons = []
    listOfBoxes = []
    listOfMasks = []

    # For each unique label (excluding background=0)
    for label_id in range(1, object_count):
        binary_mask = (labeled_mask == label_id).astype(np.uint8)
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            polygon_points = largest_contour.squeeze(axis=1).tolist()
            if len(polygon_points) < 3:
                continue
            listOfPolygons.append(polygon_points)

            # Bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            listOfBoxes.append([x, y, x + w, y + h])

            # Convert polygon to binary mask for consistency
            mask_image = Image.new('L', original_size, 0)
            draw = ImageDraw.Draw(mask_image)
            polygon_points_int = [(int(px), int(py)) for (px, py) in polygon_points]
            draw.polygon(polygon_points_int, outline=1, fill=1)
            mask_array = np.array(mask_image) * 255
            listOfMasks.append(mask_array)

    return listOfPolygons, listOfBoxes, listOfMasks

def get_mask2former_predictions(model, processor, image_path, device="cuda", threshold=0.5):
    """
    Generate predictions using a Mask2Former model.
    Returns (listOfPolygons, listOfBoxes, listOfMasks).

    This function follows a similar approach as the Mask R-CNN version:
      - It processes the image with the provided processor.
      - It runs inference with Mask2Former.
      - It post-processes the outputs to obtain an instance segmentation map.
      - For each detected instance (excluding background) it finds contours,
        extracts a polygon, computes a bounding box, and builds a binary mask.
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Process the image for the model.
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get instance segmentation.
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]
    if "segmentation" not in result:
        raise ValueError("Post-processed results do not contain the 'segmentation' key.")
    
    # The segmentation map (each instance has its own label).
    instance_map = result["segmentation"]
    if isinstance(instance_map, torch.Tensor):
        instance_map = instance_map.cpu().numpy()
    if not isinstance(instance_map, np.ndarray):
        instance_map = np.array(instance_map)
    
    # Determine background label (assumed to be the most frequent label).
    labels, counts = np.unique(instance_map, return_counts=True)
    background_label = labels[np.argmax(counts)]

    listOfPolygons = []
    listOfBoxes = []
    listOfMasks = []

    # Process each instance (skip background)
    for lbl in labels:
        if lbl == background_label:
            continue
        binary_mask = (instance_map == lbl).astype(np.uint8)
        # (Optionally) you might filter out very small instances here.
        if binary_mask.sum() < 50:
            continue
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            polygon_points = largest_contour.squeeze(axis=1).tolist()
            if isinstance(polygon_points[0], int):
                # Ensure polygon_points is a list of (x,y) pairs.
                polygon_points = [polygon_points]
            if len(polygon_points) < 3:
                continue
            listOfPolygons.append(polygon_points)

            # Bounding box from the contour.
            x, y, w, h = cv2.boundingRect(largest_contour)
            listOfBoxes.append([x, y, x+w, y+h])

            # Create a binary mask from the polygon.
            mask_image = Image.new('L', original_size, 0)
            draw = ImageDraw.Draw(mask_image)
            polygon_points_int = [(int(x), int(y)) for (x, y) in polygon_points]
            draw.polygon(polygon_points_int, outline=1, fill=1)
            mask_array = np.array(mask_image) * 255
            listOfMasks.append(mask_array)

    return listOfPolygons, listOfBoxes, listOfMasks




############################################################################
#   4) Unified function to get predictions from any loaded model
############################################################################

def get_inference_predictions(model, model_type, image_path, device="cuda"):
    """
    Wrapper that calls the appropriate inference function based on model_type.
    Returns (listOfPolygons, listOfBoxes, listOfMasks).
    
    For Mask2Former, model is expected to be a tuple (model, processor).
    """
    model_type = model_type.lower()
    if model_type == 'yolo':
        return get_yolo_predictions(model, image_path)
    elif model_type == 'maskrcnn':
        return get_maskrcnn_predictions(model, image_path, device=device)
    elif model_type == 'mobilenetv3':
        return get_mobilenetv3_predictions(model, image_path, device=device)
    elif model_type == 'mask2former':
        # model is a tuple: (mask2former_model, processor)
        mask2former_model, processor = model
        return get_mask2former_predictions(mask2former_model, processor, image_path, device=device)
    else:
        raise ValueError(f"Unknown model type for inference: {model_type}")

