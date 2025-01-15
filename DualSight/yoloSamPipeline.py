import os
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from PIL import Image, ImageDraw  # For loading images
import numpy as np
from ultralytics import YOLO
from samDemo import show_mask, show_points, show_box, mask_to_polygon, generate_random_points_within_polygon, point_to_polygon_distance, find_optimal_points, polygon_to_binary_mask, expand_bbox_within_border, fractal_dimension, apply_mask_to_image
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
from skimage.draw import polygon
from skimage.measure import regionprops
from pointSelection import select_furthest_points_from_mask
import torch.nn.functional as F

# Create the output directory if it does not exist
output_dir = "outputImages/yoloPipeline"
os.makedirs(output_dir, exist_ok=True)

def apply_colored_masks_to_image(image_path, masks_list, output_filename):
    original_image = Image.open(image_path).convert("RGBA")
    composite_mask = np.zeros((*masks_list[0].shape, 4), dtype=np.uint8)

    # Ensure all masks are of boolean type
    masks_list = [mask.astype(bool) for mask in masks_list]

    colors = [
        np.append(np.random.randint(0, 256, size=3), 180)  # Random RGBA color
        for _ in range(len(masks_list))
    ]

    mask_rgba = np.zeros((*masks_list[0].shape, 4), dtype=np.uint8)
    aggregate_mask = np.zeros(masks_list[0].shape, dtype=bool)

    for i, mask in enumerate(masks_list):
        color = colors[i]
        mask_rgba.fill(0)
        mask_rgba[mask] = color
        composite_mask = np.maximum(composite_mask, mask_rgba)
        aggregate_mask = np.logical_or(aggregate_mask, mask)

    composite_mask_image = Image.fromarray(composite_mask, mode='RGBA')
    final_image = Image.alpha_composite(original_image, composite_mask_image)
    final_image.save(f"{output_dir}/{output_filename}.png")

    aggregate_mask_image = (aggregate_mask * 255).astype(np.uint8)
    aggregate_mask_image = Image.fromarray(aggregate_mask_image, mode='L')
    aggregate_mask_image.save(f"{output_dir}/{output_filename}_aggregate_mask.png")

# Initialize YOLO model
model = YOLO('/home/sprice/RQ/models/yolov8n-seg-train2/weights/best.pt')

image_path = '/home/sprice/RQ/demo.v7i.yolov8/test/images/TruForm174-2_00-37_500X16_png.rf.17ddf81ab4dd63c5ce6f66654a48c5b4.jpg'
image = Image.open(image_path)

# Initialize SAM model and predictor
sam_checkpoint = "/home/sprice/RQ/models/sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

listOfPolygons = []
listOfBoxes = []
listOfMasks = []

# Run YOLO on the image
results = model(image)
count = 0
image_rgba = image.convert("RGBA")  # For overlay drawing

for result in results[0]:
    mask = result.masks.data
    if mask.sum() > 200:
        polygon_points = result.masks.xy[0]
        if len(polygon_points) > 2:
            listOfPolygons.append(polygon_points)
            listOfBoxes.append(result.boxes.xyxy[0])
            mask_image = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask_image)
            draw.polygon(polygon_points, outline=1, fill=1)
            mask_array = np.array(mask_image) * 255
            listOfMasks.append(mask_array)

            # Optional: Save intermediate YOLO mask visuals
            tmp = Image.new('RGBA', image.size)
            draw = ImageDraw.Draw(tmp)
            draw.polygon(polygon_points, fill=(30, 144, 255, 180))
            image_with_polygon = Image.alpha_composite(image_rgba, tmp)
            image_with_polygon.convert('RGB').save(f'{output_dir}/{count}_yoloMask.png')
            yoloMaskImage = Image.fromarray(mask_array, mode='L')
            yoloMaskImage.save(f'{output_dir}/{count}_yoloMaskBW.png')
            count += 1

# Sequentially process each detection with custom prompts for SAM
for INDEX in range(len(listOfPolygons)):
    poly = listOfPolygons[INDEX]
    box = listOfBoxes[INDEX]
    # Convert box to numpy if it's a tensor
    box = box.cpu().numpy() if isinstance(box, torch.Tensor) else np.array(box)
    
    mask = listOfMasks[INDEX]
    # Expand the bounding box within image borders
    box = np.array(expand_bbox_within_border(
        box[0], box[1], box[2], box[3], image.width, image.height, expansion_rate=0.0
    ))
    
    # Select furthest points from the mask
    selected_points, aggregate_distance, time_taken = select_furthest_points_from_mask(
        mask=mask,
        num_points=4,
        dropout_percentage=0,
        ignore_border_percentage=5,
        algorithm="Voronoi"
    )
    
    op_y, op_x = zip(*selected_points)
    optimalPoints = list(zip(op_x, op_y))
    
    # Read image using OpenCV for SAM
    loop_image = cv2.imread(image_path)
    
    # Visualize central points (optional)
    plt.figure(figsize=(10, 8))
    plt.imshow(loop_image[..., ::-1])  # Convert BGR to RGB for display
    plt.axis('off')
    plt.plot(op_x, op_y, 'ro', markersize=5)
    plt.savefig(f'{output_dir}/{INDEX}_yoloCentralPointTest.png')
    plt.close()
    
    # Set the image for the predictor
    predictor.set_image(loop_image)
    input_point = np.array(list(zip(op_x, op_y)))
    input_label = np.array([1] * len(input_point))
    
    # Predict mask using custom point prompts
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=box[None, :],
        multimask_output=True,
    )
    
    # Process and save the first predicted mask
    mask_out = masks[0]
    samMaskArray = (mask_out * 255).astype(np.uint8)
    samMaskImage = Image.fromarray(samMaskArray, mode='L')
    samMaskImage.save(f'{output_dir}/{INDEX}_samMaskBW.png')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgba)
    show_mask(mask_out, plt.gca())
    show_box(box, plt.gca())
    plt.title(f"Mask {INDEX+1}, Score: {scores[0]:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f'{output_dir}/{INDEX}_output_mask.png')
    plt.close()

# After sequential SAM predictions, you can optionally combine masks
# For example, saving SAM and YOLO composites:
print("Saving SAM Composite")
apply_colored_masks_to_image(image_path, [mask for mask in listOfMasks], 'samComposite')

print("Saving YOLO Composite")
apply_colored_masks_to_image(image_path, listOfMasks, 'yoloComposite')
