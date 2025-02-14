import os
import pickle
import csv
import json
import numpy as np
import pandas as pd
from PIL import Image
import time
import platform
import psutil
import torch
import math
import uuid  # <--- Added for unique file identifiers

# SAM / Inference helpers
import sys
sys.path.append('../')  # Adjust path as needed
from sam_helper import load_sam_model, run_sam_inference

# Tools or helper functions
from tools import (
    combine_masks_16bit,
    generate_coco_annotations_from_multi_instance_masks_16bit,
    evaluate_coco_metrics,
    compute_pixel_precision_recall
)

def safe_get(metrics_dict, key):
    if metrics_dict is not None and key in metrics_dict:
        return metrics_dict[key]
    return None

def average_or_none(values):
    return float(np.mean(values)) if values else None

def get_system_info():
    cpu_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False)
    }
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info["count"] = torch.cuda.device_count()
        gpu_info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        gpu_info["count"] = 0
        gpu_info["devices"] = []
    return {"cpu_info": cpu_info, "gpu_info": gpu_info}

def pct_to_float(pct_str):
    if isinstance(pct_str, str):
        pct_str = pct_str.strip()
        if pct_str.endswith('%'):
            pct_str = pct_str[:-1]
        if pct_str.lower() in ["n/a", "na", "none", "null"]:
            return None
    try:
        val = float(pct_str)
        # If it's NaN, return None
        if math.isnan(val):
            return None
        return val
    except:
        return None

def run_experiments(doiDF, inference_dir, images_dir, output_dir, filters):
    """
    Runs experiments based on filtered DOI configurations.
    - doiDF: DataFrame containing DOI data.
    - inference_dir: Directory where model-specific pickle files are stored.
    - images_dir: Directory containing test images.
    - output_dir: Directory to store output JSONs.
    - filters: Dictionary specifying filter criteria.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter the DOI DataFrame based on provided filters
    filtered_df = doiDF.copy()
    for key, values in filters.items():
        if key in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[key].isin(values)]

    if filtered_df.empty:
        print("[ERROR] No configurations match the provided filters.")
        return

    # Load SAM model once
    sam_model_type = "vit_l"
    sam_checkpoint = f"/home/sprice/ICCV25/modelWeights/sam_{sam_model_type}.pth"
    device = "cuda"
    print(f"[INFO] Loading SAM model from {sam_checkpoint}")
    sam_predictor = load_sam_model(
        sam_checkpoint=sam_checkpoint,
        model_type=sam_model_type,
        device=device
    )

    system_info = get_system_info()

    # Mapping from display model names to keys used in pickle filenames
    model_name_mapping = {
        "YOLOv8 Nano": "yolov8n",
        "YOLOv8 XL": "yolov8x",
        "Mask R-CNN": "maskrcnn",
        "Mask2Former": "mask2former",
        "YOLOv8 Nano + Sam": "yolov8n_dualsight",
        "YOLOv8 X-Large + Sam": "yolov8x_dualsight",
        "Mask R-CNN + Sam": "yolov8n_dualsight",
        "Mask2Former + Sam": "yolov8x_dualsight"
    }

    for _, row in filtered_df.iterrows():
        config_id = row["ID"]
        model_display_name = row["Model"]

        if model_display_name not in model_name_mapping:
            print(f"[WARNING] Model '{model_display_name}' not recognized. Skipping configuration ID {config_id}.")
            continue

        model_key = model_name_mapping[model_display_name]
        pickle_path = os.path.join(inference_dir, f"particle_{model_key}_inference.pkl")

        if not os.path.exists(pickle_path):
            print(f"[ERROR] Pickle file '{pickle_path}' not found for configuration ID {config_id}. Skipping...")
            continue

        with open(pickle_path, "rb") as pf:
            model_inference_data = pickle.load(pf)

        # Extract and process configuration parameters
        box_inclusion = str(row["BoxInclusion"]).strip().lower() == "true"
        mask_inclusion = str(row["MaskInclusion"]).strip().lower() == "true"
        number_of_pois = int(row["NumberOfPOIs"])
        poi_placement_alg = row["POIPlacementAlgorithm"]
        perimeter_buffer = pct_to_float(row["PerimeterBuffer"])
        bounding_box_distortion = pct_to_float(row["BoundingBoxDistortion"]) if box_inclusion else None
        mask_distortion = pct_to_float(row["MaskDistortion"]) if mask_inclusion else None

        if bounding_box_distortion is None or np.isnan(bounding_box_distortion):
            bounding_box_distortion = 0.0
        if mask_distortion is None or np.isnan(mask_distortion):
            mask_distortion = 0.0

        box_expansion_rate = bounding_box_distortion / 100.0
        mask_expansion_rate = mask_distortion / 100.0

        sam_params = {
            "num_points": number_of_pois,
            "algorithm": poi_placement_alg,
            "ignore_border_percentage": perimeter_buffer if perimeter_buffer is not None else 0.0,
            "use_mask_input": mask_inclusion,
            "use_box_input": box_inclusion
        }

        run_data = {
            "config_id": config_id,
            "config": row.to_dict(),
            "system_info": system_info,
            "per_image": [],
            "average_metrics": {}
        }

        # ---------------------------------------------------------------------
        # Modified: Now we also track AR@50, AR@75, AR@95, AR@50:95
        # ---------------------------------------------------------------------
        metrics_accumulators = {k: [] for k in [
            # Base BBox AP
            "base_box_AP@50", "base_box_AP@75", "base_box_AP@95", "base_box_AP@50:95",
            # Base Mask AP
            "base_mask_AP@50", "base_mask_AP@75", "base_mask_AP@95", "base_mask_AP@50:95",
            # SAM BBox AP
            "sam_box_AP@50", "sam_box_AP@75", "sam_box_AP@95", "sam_box_AP@50:95",
            # SAM Mask AP
            "sam_mask_AP@50", "sam_mask_AP@75", "sam_mask_AP@95", "sam_mask_AP@50:95",
            # Base BBox AR
            "base_box_AR@50", "base_box_AR@75", "base_box_AR@95", "base_box_AR@50:95",
            # Base Mask AR
            "base_mask_AR@50", "base_mask_AR@75", "base_mask_AR@95", "base_mask_AR@50:95",
            # SAM BBox AR
            "sam_box_AR@50", "sam_box_AR@75", "sam_box_AR@95", "sam_box_AR@50:95",
            # SAM Mask AR
            "sam_mask_AR@50", "sam_mask_AR@75", "sam_mask_AR@95", "sam_mask_AR@50:95",
        ]}

        for image_name, image_data in model_inference_data.items():
            image_path = os.path.join(images_dir, image_name)
            image_dir = os.path.dirname(image_path)
            gt_mask_dir = os.path.join(image_dir, "annotations")
            gt_mask_path = os.path.join(gt_mask_dir, image_name.replace(".jpg", "_mask.png"))

            if not os.path.exists(gt_mask_path):
                print(f"[WARNING] GT mask '{gt_mask_path}' not found. Skipping image '{image_name}'.")
                continue

            try:
                pil_image = Image.open(image_path)
                loop_image = np.array(pil_image)[:, :, ::-1].copy()  # BGR
            except Exception as e:
                print(f"[ERROR] Failed to load image '{image_path}': {e}. Skipping...")
                continue

            boxes = image_data["boxes"]
            polygons = image_data["polygons"]
            masks = image_data["masks"]

            bin_masks = []
            for mask in masks:
                if torch.is_tensor(mask):
                    mask = mask.cpu().numpy()
                bin_mask = (mask > 127).astype(np.uint8) if mask.dtype != bool else mask.astype(np.uint8)
                bin_masks.append(bin_mask)

            # Combine binary masks into 16-bit arrays
            base_16bit = combine_masks_16bit(bin_masks, return_array=True)

            start_time = time.time()
            # print(f'Running Analysis on {len(boxes)} Samples')
            # print(sam_params["algorithm"])
            sam_masks_list = run_sam_inference(
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
                use_box_input=sam_params['use_box_input'],
                use_mask_input=sam_params["use_mask_input"],
                box_expansion_rate=box_expansion_rate,
                mask_expansion_rate=mask_expansion_rate
            )
            end_time = time.time()
            processing_time = end_time - start_time
            
            sam_16bit = combine_masks_16bit(sam_masks_list, return_array=True)

            # Create unique random IDs for these temp files
            unique_id = f"{int(time.time())}_{uuid.uuid4().hex}"
            base_16bit_path = os.path.join(output_dir, f"temp_{image_name}_base_16bit_{unique_id}.png")
            sam_16bit_path = os.path.join(output_dir, f"temp_{image_name}_sam_16bit_{unique_id}.png")

            # Save the combined 16-bit masks to disk
            Image.fromarray(base_16bit).save(base_16bit_path)
            Image.fromarray(sam_16bit).save(sam_16bit_path)

            try:
                # ------------------ Base metrics (bbox + segm) ------------------
                gt_data, pred_data_base = generate_coco_annotations_from_multi_instance_masks_16bit(
                    gt_mask_path, base_16bit_path, image_path
                )
                base_segm_metrics = evaluate_coco_metrics(gt_data, pred_data_base, iou_type="segm", max_dets=450)
                base_bbox_metrics = evaluate_coco_metrics(gt_data, pred_data_base, iou_type="bbox", max_dets=450)

                # ------------------ SAM metrics (bbox + segm) ------------------
                gt_data, pred_data_sam = generate_coco_annotations_from_multi_instance_masks_16bit(
                    gt_mask_path, sam_16bit_path, image_path
                )
                sam_segm_metrics = evaluate_coco_metrics(gt_data, pred_data_sam, iou_type="segm", max_dets=450)
                sam_bbox_metrics = evaluate_coco_metrics(gt_data, pred_data_sam, iou_type="bbox", max_dets=450)

            except Exception as e:
                print(f"[ERROR] Failed to evaluate metrics for image '{image_name}': {e}")
                base_segm_metrics = base_bbox_metrics = {}
                sam_segm_metrics = sam_bbox_metrics = {}

            # Build image-level metrics
            image_result = {
                "image_name": image_name,
                "original_image_path": image_path,
                "gt_mask_path": gt_mask_path,
                "processing_time_sec": processing_time,

                # ------------------ Base BBox AP metrics ------------------
                "base_box_AP@50": safe_get(base_bbox_metrics, "AP@50"),
                "base_box_AP@75": safe_get(base_bbox_metrics, "AP@75"),
                "base_box_AP@95": safe_get(base_bbox_metrics, "AP@95"),
                "base_box_AP@50:95": safe_get(base_bbox_metrics, "AP@50:95"),

                # ------------------ Base Mask AP metrics ------------------
                "base_mask_AP@50": safe_get(base_segm_metrics, "AP@50"),
                "base_mask_AP@75": safe_get(base_segm_metrics, "AP@75"),
                "base_mask_AP@95": safe_get(base_segm_metrics, "AP@95"),
                "base_mask_AP@50:95": safe_get(base_segm_metrics, "AP@50:95"),

                # ------------------ SAM BBox AP metrics ------------------
                "sam_box_AP@50": safe_get(sam_bbox_metrics, "AP@50"),
                "sam_box_AP@75": safe_get(sam_bbox_metrics, "AP@75"),
                "sam_box_AP@95": safe_get(sam_bbox_metrics, "AP@95"),
                "sam_box_AP@50:95": safe_get(sam_bbox_metrics, "AP@50:95"),

                # ------------------ SAM Mask AP metrics ------------------
                "sam_mask_AP@50": safe_get(sam_segm_metrics, "AP@50"),
                "sam_mask_AP@75": safe_get(sam_segm_metrics, "AP@75"),
                "sam_mask_AP@95": safe_get(sam_segm_metrics, "AP@95"),
                "sam_mask_AP@50:95": safe_get(sam_segm_metrics, "AP@50:95"),

                # ------------------ Base BBox AR metrics ------------------
                "base_box_AR@50": safe_get(base_bbox_metrics, "AR@50"),
                "base_box_AR@75": safe_get(base_bbox_metrics, "AR@75"),
                "base_box_AR@95": safe_get(base_bbox_metrics, "AR@95"),
                "base_box_AR@50:95": safe_get(base_bbox_metrics, "AR@50:95"),

                # ------------------ Base Mask AR metrics ------------------
                "base_mask_AR@50": safe_get(base_segm_metrics, "AR@50"),
                "base_mask_AR@75": safe_get(base_segm_metrics, "AR@75"),
                "base_mask_AR@95": safe_get(base_segm_metrics, "AR@95"),
                "base_mask_AR@50:95": safe_get(base_segm_metrics, "AR@50:95"),

                # ------------------ SAM BBox AR metrics ------------------
                "sam_box_AR@50": safe_get(sam_bbox_metrics, "AR@50"),
                "sam_box_AR@75": safe_get(sam_bbox_metrics, "AR@75"),
                "sam_box_AR@95": safe_get(sam_bbox_metrics, "AR@95"),
                "sam_box_AR@50:95": safe_get(sam_bbox_metrics, "AR@50:95"),

                # ------------------ SAM Mask AR metrics ------------------
                "sam_mask_AR@50": safe_get(sam_segm_metrics, "AR@50"),
                "sam_mask_AR@75": safe_get(sam_segm_metrics, "AR@75"),
                "sam_mask_AR@95": safe_get(sam_segm_metrics, "AR@95"),
                "sam_mask_AR@50:95": safe_get(sam_segm_metrics, "AR@50:95"),
            }
            # print(image_result)
            # Save image_result to run_data
            run_data["per_image"].append(image_result)

            # Accumulate for averaging
            for k in metrics_accumulators.keys():
                val = image_result.get(k, None)
                if val is not None:
                    metrics_accumulators[k].append(val)

            # Remove temporary mask files
            if os.path.exists(base_16bit_path):
                os.remove(base_16bit_path)
            if os.path.exists(sam_16bit_path):
                os.remove(sam_16bit_path)

        # After processing all images for this config
        run_data["average_metrics"] = {k: average_or_none(v) for k, v in metrics_accumulators.items()}

        json_filename = f"{config_id}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w") as jf:
            json.dump(run_data, jf, indent=4)

        print(f"[INFO] Saved results for configuration ID {config_id} to {json_path}")



if __name__ == "__main__":
    # Load DOI.csv into a DataFrame
    doiDF = pd.read_csv("DOI.csv")

    # Define experiment filters as desired
    experiment_filters = {
        'ID': list(range(9865, 10065))
        # 'ID': [4660, 4661, 4662, 4687, 4688, 4689, 4696, 4700, 4701, 4705, 4706, 4707, 4712, 4713, 4714, 4715, 4716,
            #    4720, 4721, 4722, 4723, 4724, 4725]
        # 'ID': [3684]# + list(range(3611, 3684)) #+ list(range(3577,3599))
        # 'BoxInclusion': [True],
        # 'MaskInclusion': [True],
        # 'BoundingBoxDistortion': ['100%'],
        # 'Model': ['YOLOv8 Nano + Sam'],
        # 'NumberOfPOIs': [1]
        # "POIPlacementAlgorithm": ['Random']

    }

    run_experiments(
        doiDF=doiDF,
        inference_dir="../../savedInference",
        images_dir="/home/sprice/ICCV25/datasets/powder/test",
        output_dir="ablation_outputs",
        filters=experiment_filters
    )
