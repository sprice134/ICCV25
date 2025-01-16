import os
import torch
import mmcv
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.utils import replace_cfg_vals, rfnext_init_model
from mmdet.core.mask import BitmapMasks

def refine_single_image_multi_object():
    # ------------------------------------------------------------------------------
    # 1) Paths
    # ------------------------------------------------------------------------------
    config_path = "/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py"
    checkpoint_path = "/home/sprice/ICCV25/modelWeights/segrefiner_lr_latest.pth"
    # image_path = "/home/sprice/ICCV25/SegRefinerV2/TruForm174-2_00-37_500X16_png.rf.17ddf81ab4dd63c5ce6f66654a48c5b4.jpg"
    image_path = "/home/sprice/ICCV25/SegRefinerV2/Cu-Ni-Powder_250x_10_SE_png.rf.cd93ec4589ad8f4e412cb1ec0e805016.jpg"
    mask_path = "/home/sprice/ICCV25/SegRefinerV2/yoloMasks.png"  # Coarse multi-object mask (YOLO)
    output_path = "/home/sprice/ICCV25/SegRefinerV2/refined_grayscale.png"

    # ------------------------------------------------------------------------------
    # 2) Load SegRefiner configuration & model
    # ------------------------------------------------------------------------------
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)

    cfg.device = 'cuda'
    cfg.gpu_ids = [0]

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None

    cfg.model.train_cfg = None  # Ensure testing mode
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    rfnext_init_model(model, cfg=cfg)

    load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
    model = model.to(cfg.device)
    model.eval()

    # ------------------------------------------------------------------------------
    # 3) Load and preprocess the image
    # ------------------------------------------------------------------------------
    img = mmcv.imread(image_path)
    img_name = os.path.basename(image_path)

    if 'img_norm_cfg' in cfg:
        mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
        to_rgb = cfg.img_norm_cfg.get('to_rgb', True)
        img = mmcv.imnormalize(img, mean=mean, std=std, to_rgb=to_rgb)

    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(cfg.device)

    # ------------------------------------------------------------------------------
    # 4) Load the coarse multi-object mask (YOLO)
    # ------------------------------------------------------------------------------
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Coarse mask file not found: {mask_path}")

    coarse_masks = mmcv.imread(mask_path, flag='grayscale')  # single-channel mask
    unique_labels = np.unique(coarse_masks)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (0)

    mask_height, mask_width = coarse_masks.shape

    # ------------------------------------------------------------------------------
    # 5) Prepare final combined refined mask + output directories for single masks
    # ------------------------------------------------------------------------------
    refined_grayscale_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    base_output_dir = os.path.dirname(output_path)
    os.makedirs(base_output_dir, exist_ok=True)

    # Sub-directories for per-object coarse/refined masks
    coarse_single_masks_dir = os.path.join(base_output_dir, "coarse_single_masks")
    refined_single_masks_dir = os.path.join(base_output_dir, "refined_single_masks")
    os.makedirs(coarse_single_masks_dir, exist_ok=True)
    os.makedirs(refined_single_masks_dir, exist_ok=True)

    # ------------------------------------------------------------------------------
    # 6) For each instance ID, refine the single mask
    # ------------------------------------------------------------------------------
    for idx, obj_id in enumerate(unique_labels, start=1):
        # -----------------------------
        # 6A) Extract single-object coarse mask
        # -----------------------------
        single_mask = (coarse_masks == obj_id).astype(np.uint8)  # shape: (H, W)
        single_mask_3d = single_mask[np.newaxis, :, :]  # shape: (1, H, W) for BitmapMasks

        # Save coarse single-object mask as black/white
        single_coarse_filename = os.path.join(
            coarse_single_masks_dir, f"coarse_mask_obj_{obj_id}.png"
        )
        mmcv.imwrite(single_mask * 255, single_coarse_filename)

        # Prepare the BitmapMasks object
        coarse_bitmap_mask = BitmapMasks(
            single_mask_3d, height=mask_height, width=mask_width
        )

        # -----------------------------
        # 6B) Run SegRefiner
        # -----------------------------
        img_metas = {
            'ori_filename': img_name,
            'img_shape': img.shape[:2] + (3,),
            'ori_shape': img.shape[:2] + (3,),
            'pad_shape': img.shape[:2] + (3,),
            'scale_factor': 1.0,
            'flip': False,
        }

        data = {
            'img': img_tensor,
            'img_metas': [img_metas],
            'coarse_masks': [coarse_bitmap_mask],
        }

        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)

        # results[0][0] is presumably the refined mask for this single object
        refined_image = results[0][0]
        if isinstance(refined_image, torch.Tensor):
            refined_image = refined_image.cpu().numpy()

        # If shape is (3,H,W), convert to (H,W,3) then reduce if needed
        if refined_image.ndim == 3 and refined_image.shape[0] == 3:
            refined_image = refined_image.transpose(1, 2, 0)

        # Convert to 0~255
        refined_image = (refined_image * 255).astype(np.uint8)

        # Save the single refined mask as black/white
        refined_bw = (refined_image > 0).astype(np.uint8) * 255
        single_refined_filename = os.path.join(
            refined_single_masks_dir, f"refined_mask_obj_{obj_id}.png"
        )
        mmcv.imwrite(refined_bw, single_refined_filename)

        # -----------------------------
        # 6C) Update the final combined grayscale
        # -----------------------------
        # Assign a unique (idx) label for all refined pixels > 0
        refined_grayscale_mask[refined_image > 0] = idx

    # ------------------------------------------------------------------------------
    # 7) Save final combined mask
    # ------------------------------------------------------------------------------
    mmcv.imwrite(refined_grayscale_mask, output_path)
    print(f"Refined grayscale mask saved at: {output_path}")

if __name__ == '__main__':
    refine_single_image_multi_object()
