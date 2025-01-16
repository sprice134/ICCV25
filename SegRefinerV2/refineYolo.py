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
    # Paths
    config_path = "/home/sprice/ICCV25/SegRefinerV2/configs/segrefiner/segrefiner_config.py"
    checkpoint_path = "/home/sprice/ICCV25/modelWeights/segrefiner_lr_latest.pth"
    # image_path = "/home/sprice/ICCV25/SegRefiner/jan15Test/HP743_5S_500x_png.rf.9ff406796462449f85c2039537f32d6f.jpg"
    image_path = '/home/sprice/ICCV25/SegRefinerV2/TruForm174-2_00-37_500X16_png.rf.17ddf81ab4dd63c5ce6f66654a48c5b4.jpg'
    mask_path = "/home/sprice/ICCV25/SegRefinerV2/yoloMasks.png"  # Coarse masks
    output_path = "/home/sprice/ICCV25/SegRefinerV2/refined_grayscale.png"

    # Load configuration
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)

    # Set device
    cfg.device = 'cuda'
    cfg.gpu_ids = [0]

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None

    # Build model
    cfg.model.train_cfg = None  # Ensure testing mode
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    rfnext_init_model(model, cfg=cfg)

    # Load checkpoint
    load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
    model = model.to(cfg.device)
    model.eval()

    # Load and preprocess the image
    img = mmcv.imread(image_path)
    img_name = os.path.basename(image_path)

    if 'img_norm_cfg' in cfg:
        mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
        img = mmcv.imnormalize(img, mean=mean, std=std, to_rgb=cfg.img_norm_cfg.get('to_rgb', True))

    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(cfg.device)

    # Load and preprocess the coarse masks
    if os.path.exists(mask_path):
        coarse_masks = mmcv.imread(mask_path, flag='grayscale')  # Load multi-object mask as grayscale
        unique_labels = np.unique(coarse_masks)  # Unique object IDs (0 for background)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background (0)
    else:
        raise FileNotFoundError(f"Coarse mask file not found: {mask_path}")

    # Create a single-channel output for refined masks
    mask_height, mask_width = coarse_masks.shape
    refined_grayscale_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # Iterate over unique_labels in reverse order
    for idx, obj_id in enumerate(unique_labels, start=1):  # Start from the last label
        # Extract a single object's mask
        single_mask = (coarse_masks == obj_id).astype(np.uint8)  # Binary mask for the current object
        single_mask = single_mask[np.newaxis, :, :]  # Add channel dimension
        coarse_bitmap_mask = BitmapMasks(single_mask, height=mask_height, width=mask_width)

        # Create a properly structured img_metas dictionary
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

        refined_image = results[0][0]  # Adjust index based on model's output structure
        if isinstance(refined_image, torch.Tensor):
            refined_image = refined_image.cpu().numpy()
        if refined_image.ndim == 3 and refined_image.shape[0] == 3:  # CHW to HWC
            refined_image = refined_image.transpose(1, 2, 0)
        refined_image = (refined_image * 255).astype(np.uint8)

        # Add the refined mask to the grayscale output
        refined_grayscale_mask[refined_image > 0] = idx  # Assign a unique value for the object

    # Save the refined mask as a grayscale image
    mmcv.imwrite(refined_grayscale_mask, output_path)
    print(f"Refined grayscale mask saved at: {output_path}")


if __name__ == '__main__':
    refine_single_image_multi_object()
