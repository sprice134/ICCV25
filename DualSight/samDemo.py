from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Polygon, Point
from itertools import combinations
import random
from PIL import Image, ImageDraw
from pointSelection import select_furthest_points_from_mask
import time
from scipy.ndimage import label

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255, 0, 0, 0.5])
        # color = np.array([0, 0, 255, 0.5])

    # Ensure mask is within [0, 1]
    mask = np.clip(mask, 0, 1)

    h, w = mask.shape[-2:]

    # Separate RGB and alpha channels
    rgb_color = color[:3]
    alpha_channel = color[3]

    # Apply the mask to the RGB color
    mask_image_rgb = mask.reshape(h, w, 1) * rgb_color.reshape(1, 1, -1)
    mask_image_rgb = np.clip(mask_image_rgb, 0, 1)
    alpha_mask = mask.reshape(h, w) * alpha_channel
    mask_image = np.dstack((mask_image_rgb, alpha_mask))
    mask_image = np.clip(mask_image, 0, 1)

    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def mask_to_polygon(mask):
    '''NOTE: Smaller EPSILON LEADS TO MORE PRECISE POLYGON'''
    EPSILON_SCALAR = 0.001
    # Convert mask to a format suitable for contour detection
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Approximate contour to polygon
        epsilon = EPSILON_SCALAR * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Flatten and convert to list of (x, y) points
        polygon = approx.reshape(-1, 2).tolist()
        polygons.append(polygon)
    
    return polygons

def polygon_to_binary_mask(polygon_points, height, width, fill_value=255):
    # Initialize an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Ensure the polygon_points are in the expected shape (n_points, 1, 2)
    if np.array(polygon_points).ndim == 2:
        polygon_points = np.array([polygon_points])

    # Fill the polygon
    cv2.fillPoly(mask, np.int32([polygon_points]), color=fill_value)

    return mask


    

def generate_random_points_within_polygon(polygon, num_points=500):
    """
    Generate random points within the given polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        random_point = Point([random.uniform(minx, maxx), random.uniform(miny, maxy)])
        if polygon.contains(random_point):
            points.append(random_point)
    return points

def point_to_polygon_distance(point, polygon):
    """
    Calculate the distance from a point to the nearest boundary of the polygon.
    """
    return polygon.boundary.distance(point)

def find_optimal_points(points, polygon, num_result_points=3, border_weight=2):
    """
    Find points that maximize the minimum distance among them and their distance from borders,
    with an adjustable weighting for the border distance to influence the selection more.
    """
    best_score = 0
    best_combo = None
    for combo in combinations(points, num_result_points):
        # Calculate the minimum distance in the current combination
        min_dist_between_points = min([combo[i].distance(combo[j]) for i in range(len(combo)) for j in range(i + 1, len(combo))])
        # Calculate the minimum distance from each point to the polygon's border
        min_dist_to_border = min([point_to_polygon_distance(point, polygon) for point in combo])
        # Adjust the score by applying a weight to the border distance
        score = min_dist_between_points + (min_dist_to_border * border_weight)
        
        if score > best_score:
            best_score = score
            best_combo = combo
    return best_combo


def expand_bbox_within_border(x1, y1, x2, y2, img_width, img_height, expansion_rate=0.1):
    """
    Expand bounding box by a certain rate but ensure it doesn't go beyond image borders.

    Args:
    - x1, y1: Coordinates of the top-left corner of the bounding box.
    - x2, y2: Coordinates of the bottom-right corner of the bounding box.
    - img_width, img_height: Dimensions of the image.
    - expansion_rate: Fraction of the width/height to expand the bounding box.

    Returns:
    - A tuple of expanded bounding box coordinates (x1, y1, x2, y2) adjusted to image borders.
    """
    # Calculate expansion values
    width_expansion = (x2 - x1) * expansion_rate
    height_expansion = (y2 - y1) * expansion_rate

    # Expand each side
    x1_expanded = max(0, x1 - width_expansion)
    y1_expanded = max(0, y1 - height_expansion)
    x2_expanded = min(img_width, x2 + width_expansion)
    y2_expanded = min(img_height, y2 + height_expansion)

    return (x1_expanded, y1_expanded, x2_expanded, y2_expanded)


def fractal_dimension(Z, threshold=0.9):
    # Only for 2d image, Z is binary image
    assert(len(Z.shape) == 2)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])
    # Transform Z into a binary array
    Z = (Z < threshold)
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def apply_mask_to_image(image_path, mask, color_with_alpha):
    # Load the image
    image = Image.open(image_path).convert("RGBA")
    original = np.array(image)
    
    # Create a colored mask
    colored_mask = np.zeros((*mask.shape, 4))  # Prepare a 4-channel (RGBA) image
    colored_mask[mask == 1] = color_with_alpha  # Apply color and alpha to the mask area
    
    # Blend the original image and the colored mask
    # First, normalize alpha values from [0, 1] to [0, 255] for blending
    alpha = colored_mask[..., 3] * 255
    foreground = colored_mask[..., :3] * alpha[..., None]
    background = original[..., :3] * (255 - alpha[..., None])
    
    # Combine foreground and background, then divide by alpha to respect transparency
    combined_rgb = (foreground + background) / 255
    combined_alpha = alpha + (255 - alpha) * (original[..., 3] / 255)
    combined = np.dstack((combined_rgb, combined_alpha))  # Stack them together into a single image
    
    # Convert back to an Image object
    result_image = Image.fromarray(np.uint8(combined))
    
    return result_image


if __name__ == '__main__':
    sam_checkpoint = "/Users/sprice/Documents/GitHub/RQ/models/sam_vit_l_0b3195.pth"
    # sam_checkpoint = "model/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    # device = "cuda"
    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_path = 'demo.v7i.yolov8/test/images/S02_03_SE1_1000X24_png.rf.61ceee7fe0a4f4ccabd61c1e71524baf.jpg'
    image = cv2.imread(image_path)
    cv2.imwrite('outputImages/prePrediction.png', image)

    predictor.set_image(image)
    input_point = np.array([[850, 550]])
    input_label = np.array([1])
    
    startTime = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    postPrediction = time.time()

    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.savefig(f'outputImages/output_mask_{i+1}.png')


    for i, (mask, score) in enumerate(zip(masks, scores)):
        if i == 0:
            generatedSamplePoints = time.time()
            selected_points, aggregate_distance, time_taken = select_furthest_points_from_mask(
                            mask=mask,
                            num_points=3,
                            dropout_percentage=0,
                            ignore_border_percentage=0,
                            algorithm="Hill Climbing"
                        )
            postPointFinding = time.time()
            op_y, op_x = zip(*selected_points)
            print('Aggregated Distance: ', aggregate_distance)
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            plt.plot(op_x, op_y, 'ro', markersize=5)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.savefig(f'outputImages/output_mask_{i+1}_centralPoints.png')
            plt.close()
            print(postPrediction - startTime)
            print(postPointFinding - generatedSamplePoints)


            generatedSamplePoints = time.time()
            selected_points, aggregate_distance, time_taken = select_furthest_points_from_mask(
                            mask=mask,
                            num_points=3,
                            dropout_percentage=98,
                            ignore_border_percentage=0,
                            algorithm="Naive"
                        )
            print('Aggregated Distance: ', aggregate_distance)
            postPointFinding = time.time()
            op_y, op_x = zip(*selected_points)
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            plt.plot(op_x, op_y, 'ro', markersize=5)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.savefig(f'outputImages/output_mask_{i+1}_naive.png')
            plt.close()
            print(postPrediction - startTime)
            print(postPointFinding - generatedSamplePoints)

def apply_colored_masks_to_image(image_path, masks_list, output_filename):
    original_image = Image.open(image_path).convert("RGBA")
    composite_mask = np.zeros((*masks_list[0].shape, 4), dtype=np.uint8)
    image = Image.open(image_path)

    # Ensure all masks are of boolean type
    masks_list = [mask.astype(bool) for mask in masks_list]  # Updated for compatibility

    colors = [
        np.append(np.random.randint(0, 256, size=3), 180)  # Random RGB and alpha of 127
        for _ in range(len(masks_list))
    ]

    mask_rgba = np.zeros((*masks_list[0].shape, 4), dtype=np.uint8)
    for i, mask in enumerate(masks_list):
        color = colors[i]
        mask_rgba.fill(0)  # Clear the array instead of recreating
        mask_rgba[mask] = color
        composite_mask = np.maximum(composite_mask, mask_rgba)

    composite_mask_image = Image.fromarray(composite_mask, mode='RGBA')
    final_image = Image.alpha_composite(original_image, composite_mask_image)
    final_image.save(output_filename)
    

def largest_object_only(binary_mask):

    labeled_mask, num_features = label(binary_mask)
    
    if num_features == 0:
        # If no features are found, return the original mask (all zeros)
        return binary_mask

    # Calculate the size of each connected component
    component_sizes = np.bincount(labeled_mask.ravel())

    # Ignore the background label (0)
    component_sizes[0] = 0

    # Find the largest component (by size)
    largest_component = component_sizes.argmax()

    # Create a mask for the largest component only
    largest_object_mask = (labeled_mask == largest_component)

    return largest_object_mask.astype(np.uint8)

def remove_contained_masks(masks, containment_threshold=0.95):
    """
    Remove masks that are more than a specified percentage contained by another mask.

    Parameters:
    masks (list of numpy arrays): List of binary masks.
    containment_threshold (float): Percentage threshold for containment (default 0.95).

    Returns:
    list of numpy arrays: Filtered list of masks.
    """
    # Keep track of masks to keep
    keep = [True] * len(masks)

    # Iterate through all pairs of masks
    for i in range(len(masks)):
        for j in range(len(masks)):
            if i != j and keep[i] and keep[j]:
                # Calculate intersection
                intersection = np.logical_and(masks[i], masks[j])

                # Calculate areas
                area_i = np.sum(masks[i])
                area_j = np.sum(masks[j])
                intersection_area = np.sum(intersection)

                # Determine containment percentages
                if area_i > 0 and area_j > 0:
                    containment_i = intersection_area / area_i
                    containment_j = intersection_area / area_j

                    # Check if one is contained within the other
                    if containment_i > containment_threshold:
                        keep[i] = False
                    elif containment_j > containment_threshold:
                        keep[j] = False

    # Return the filtered list of masks
    return [mask for mask, to_keep in zip(masks, keep) if to_keep]

