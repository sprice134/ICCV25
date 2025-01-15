import json
import numpy as np
import os
from PIL import Image
from pycocotools import mask as coco_mask

def convert_coco_to_16bit_masks(coco_json_path, output_dir):
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    for image_info in coco_data['images']:
        # Get image details
        image_id = image_info['id']
        image_name = image_info['file_name']
        image_height = image_info['height']
        image_width = image_info['width']

        # Initialize a blank 16-bit mask for the image
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint16)

        # Extract annotations for the current image
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

        # Assign a unique value to each instance
        for instance_id, ann in enumerate(annotations, start=1):
            if instance_id > 65535:
                raise ValueError(f"Image {image_name} contains more than 65,535 instances, which is unsupported.")

            # Decode RLE or polygon annotations
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Handle polygon segmentation
                    poly_mask = coco_mask.frPyObjects(ann['segmentation'], image_height, image_width)
                    rle_mask = coco_mask.merge(poly_mask)
                else:
                    # Handle already encoded RLE
                    rle_mask = ann['segmentation']

                # Decode RLE into a binary mask
                binary_mask = coco_mask.decode(rle_mask)

                # Add the instance ID to the combined mask
                combined_mask[binary_mask > 0] = instance_id

        # Save the combined mask as a 16-bit grayscale PNG
        output_mask_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_mask.png")
        Image.fromarray(combined_mask).save(output_mask_path, format='PNG')

    print(f"Converted 16-bit masks saved to: {output_dir}")

# Example usage
coco_json_path = '_annotations.coco.json'  # Path to the COCO JSON file
output_dir = 'output_masks_16bit'  # Directory to save 16-bit grayscale masks
convert_coco_to_16bit_masks(coco_json_path, output_dir)
