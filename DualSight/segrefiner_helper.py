import numpy as np
import sys
sys.path.append('../SegRefinerV2/')
import mmcv
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.utils import replace_cfg_vals, rfnext_init_model
from mmdet.core.mask import BitmapMasks


def load_segrefiner_model(config_path, checkpoint_path, device='cuda'):
    """
    Load SegRefiner model based on your provided code.
    """
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)

    cfg.device = device
    cfg.gpu_ids = [0]

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None

    # Build the SegRefiner model
    cfg.model.train_cfg = None  # test mode
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    rfnext_init_model(model, cfg=cfg)

    # Load checkpoint
    load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
    model = model.to(device)
    model.eval()
    return model, cfg


def run_segrefiner_inference(segrefiner_model, segrefiner_cfg,
                             list_of_binary_masks, # same shape as your YOLO masks
                             loop_image_bgr,       # the original BGR image
                             device='cuda'):
    """
    Refine each mask in `list_of_binary_masks` using SegRefiner.
    Returns a new list of refined binary masks (same count).
    """
    # Make sure all masks share the same size as loop_image_bgr
    H, W = loop_image_bgr.shape[:2]

    # Preprocess the image for SegRefiner
    img = mmcv.bgr2rgb(loop_image_bgr)  # or keep it BGR if that’s correct
    if 'img_norm_cfg' in segrefiner_cfg:
        mean = np.array(segrefiner_cfg.img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(segrefiner_cfg.img_norm_cfg['std'], dtype=np.float32)
        to_rgb = segrefiner_cfg.img_norm_cfg.get('to_rgb', True)
        img = mmcv.imnormalize(img, mean=mean, std=std, to_rgb=to_rgb)

    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    refined_masks_list = []

    for single_binary_mask in list_of_binary_masks:
        # single_binary_mask is shape [H, W] with 0/1
        single_mask_np = single_binary_mask.astype(np.uint8)
        single_mask_np = single_mask_np[np.newaxis, :, :]  # shape [1,H,W]

        # Convert to an mmdet BitmapMasks
        coarse_bitmap_mask = BitmapMasks(single_mask_np, height=H, width=W)

        # Prepare the data
        img_metas = {
            'ori_filename': 'tmp.jpg',
            'img_shape': (H, W, 3),
            'ori_shape': (H, W, 3),
            'pad_shape': (H, W, 3),
            'scale_factor': 1.0,
            'flip': False,
        }
        data = {
            'img': img_tensor,
            'img_metas': [img_metas],
            'coarse_masks': [coarse_bitmap_mask],
        }

        with torch.no_grad():
            results = segrefiner_model(return_loss=False, rescale=True, **data)

        refined_image = results[0][0]
        if isinstance(refined_image, torch.Tensor):
            refined_image = refined_image.cpu().numpy()

        # If shape is (3,H,W), reduce to single channel if needed
        if refined_image.ndim == 3 and refined_image.shape[0] == 3:
            refined_image = refined_image.transpose(1, 2, 0)

        # Convert from [0..1] float to 0/1 mask
        refined_mask = (refined_image > 0.5).astype(np.uint8)

        refined_masks_list.append(refined_mask)

    return refined_masks_list