from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model with your specified weights
# yolo_model_path = "/home/sprice/RQ/models/yolov8n-seg-train2/weights/best.pt"
yolo_model_path = '/home/sprice/ICCV25/modelWeights/yolov8n-seg.pt'
model = YOLO(yolo_model_path)

# Input and output paths
# input_image_path = "HP743_5S_500x_png.rf.9ff406796462449f85c2039537f32d6f.jpg"  # Replace with your image path
# input_image_path = 'TruForm174-2_00-37_500X16_png.rf.17ddf81ab4dd63c5ce6f66654a48c5b4.jpg'
input_image_path = 'Cu-Ni-Powder_250x_10_SE_png.rf.cd93ec4589ad8f4e412cb1ec0e805016.jpg'

output_mask_path = "yoloMasks.png"  # Path to save the mask image

# Load the image
image = cv2.imread(input_image_path)

# Get the original image dimensions
original_height, original_width = image.shape[:2]

# Run inference
results = model(image)

# Extract masks from the results
masks = results[0].masks.data  # Extract mask data as tensor
if masks is None or masks.numel() == 0:
    print("No masks detected in the image.")
else:
    # Convert mask tensors to CPU and NumPy format
    masks = masks.cpu().numpy()

    # Combine all masks into one binary mask
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for idx, mask in enumerate(masks, start=1):
        combined_mask[mask > 0.5] = idx  # Assign unique values to each instance

    # Resize the combined mask to match the original image dimensions
    resized_mask = cv2.resize(
        combined_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    # Save the resized mask as an image
    cv2.imwrite(output_mask_path, resized_mask)
    print(f"Mask saved to {output_mask_path}")
