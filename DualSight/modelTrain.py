#!/usr/bin/env python

import argparse
import os
import time
import psutil
import shutil
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.models import mobilenet_v3_large

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from scipy.ndimage import label

##############################################################################
#                               1) YOLO TRAINING
##############################################################################
# You must have 'ultralytics' installed (pip install ultralytics)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def train_yolo(
    data_yaml="../demo.v7i.yolov8/data.yaml",
    epochs=500,
    batch=16,
    device="0",
    model_size="n"
):
    """
    Trains a YOLOv8 segmentation model. Uses an ultralytics pretrained checkpoint
    (n, s, m, l, xl).
    """

    # Map from short 'model_size' to the ultralytics pretrained segmentation checkpoint
    size_map = {
        "n": "yolov8n-seg.pt",
        "s": "yolov8s-seg.pt",
        "m": "yolov8m-seg.pt",
        "l": "yolov8l-seg.pt",
        "x": "yolov8x-seg.pt",
        "xl": "yolov8x-seg.pt",  # 'x' and 'xl' point to the same file at time of writing
    }

    if YOLO is None:
        raise ImportError("The ultralytics package is not installed. Cannot train YOLO.")
    if model_size not in size_map:
        raise ValueError(f"Unknown model_size '{model_size}'. Choose from n, s, m, l, x, xl.")

    base_checkpoint = size_map[model_size]
    print(f"[YOLO] Using checkpoint: {base_checkpoint}")

    # Load the YOLO model
    model = YOLO(base_checkpoint)

    # Train
    model.train(
        data=data_yaml,        # path to the data.yaml
        epochs=epochs,
        batch=batch,
        name=f"yolov8{model_size}-seg-train",
        device=device
    )


##############################################################################
#                           2) MASK R-CNN TRAINING
##############################################################################
import torch
import torchvision
import matplotlib.colors as mcolors
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_maskrcnn_model(num_classes=2, pretrained=True):
    """
    Return a Mask R-CNN model (ResNet50 FPN backbone), with the head adjusted
    for `num_classes`.
    """
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    # Replace the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=hidden_layer,
        num_classes=num_classes
    )
    return model


def visualize_sample_with_labels(image, target, save_path):
    """
    Simple visualization of image, bounding boxes, and instance masks.
    """
    # Convert image tensor to [0,1] range for matplotlib
    image = image.permute(1, 2, 0).cpu().numpy()
    image = image - image.min()
    image = image / image.max()

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Draw predicted or GT masks
    masks = target['masks'].cpu().numpy()  # (N, H, W)
    for i, mask in enumerate(masks):
        mask = mask.squeeze()
        mask_color = mcolors.hsv_to_rgb([i / (len(masks)+1), 1, 1]) 
        ax.contour(mask, levels=[0.5], colors=[mask_color], linewidths=2)
        ax.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.4, cmap='hsv')

    # Draw bounding boxes
    boxes = target['boxes'].cpu().numpy()
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


class MaskRCNNDataset(Dataset):
    """
    A simple dataset that loads images from an 'images/' folder,
    and corresponding instance mask PNGs from a 'masks/' folder.
    """
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Expect a .png mask with the same name (just changed extension)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load and convert
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask_np = np.array(mask)  # shape HxW

        # Make sure we have integer labels
        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[obj_ids != 0]  # remove background

        # Generate masks for each object
        masks = (mask_np[None, ...] == obj_ids[:, None, None])  # shape => (#objs, H, W)

        # For bounding boxes
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # single class = 1
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Convert image to tensor
        image_tensor = F.to_tensor(img)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_maskrcnn(
    train_image_dir,
    train_mask_dir,
    valid_image_dir,
    valid_mask_dir,
    num_classes=2,
    num_epochs=1000,
    patience=50,
    batch_size=4,
    learning_rate=0.0005
):
    """
    Example function to train a Mask R-CNN model with patience-based early stopping.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & Dataloader
    train_dataset = MaskRCNNDataset(train_image_dir, train_mask_dir)
    valid_dataset = MaskRCNNDataset(valid_image_dir, valid_mask_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Model, optimizer, scheduler
    model = get_maskrcnn_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    cpu_mem_usage = []
    gpu_mem_usage = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Check for empty targets
            if any(len(t["boxes"]) == 0 for t in targets):
                continue

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        if lr_scheduler:
            lr_scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for images, targets in valid_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                if any(len(t["boxes"]) == 0 for t in targets):
                    continue

                # Switch temporarily to train mode to compute losses
                model.train()
                loss_dict = model(images, targets)
                model.eval()

                val_step_loss = sum(loss for loss in loss_dict.values())
                val_loss += val_step_loss.item()
                val_steps += 1

        val_loss = val_loss / val_steps if val_steps > 0 else float('inf')

        print(f"[Epoch {epoch+1:03d}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'best_mask_rcnn_model.pth')
            print("Validation loss decreased -> best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

        # Memory usage
        cpu_mem = psutil.virtual_memory().used / (1024 ** 3)
        if torch.cuda.is_available():
            dev_idx = device.index if device.type == 'cuda' else 0
            torch.cuda.synchronize(dev_idx)
            gpu_mem = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
        else:
            gpu_mem = 0.0

        cpu_mem_usage.append(cpu_mem)
        gpu_mem_usage.append(gpu_mem)

    # Restore best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state from training.")

    torch.save(model.state_dict(), 'final_mask_rcnn_model.pth')
    print("Training complete. Final model: 'final_mask_rcnn_model.pth'.")

    end_time = time.time()
    total_time = end_time - start_time
    avg_cpu_mem = sum(cpu_mem_usage)/len(cpu_mem_usage) if cpu_mem_usage else 0
    avg_gpu_mem = sum(gpu_mem_usage)/len(gpu_mem_usage) if gpu_mem_usage else 0

    print("\n[Training Summary for Mask R-CNN]")
    print(f"Average CPU Memory Usage: {avg_cpu_mem:.2f} GB")
    print(f"Average GPU Memory Usage: {avg_gpu_mem:.2f} GB")
    print(f"Total Training Time: {total_time/3600:.2f} hours ({total_time:.2f} seconds)")


##############################################################################
#                        3) MobileNetV3 SEG TRAINING
##############################################################################

class MobileNetV3Segmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = mobilenet_v3_large(pretrained=True).features
        self.decoder = nn.Sequential(
            nn.Conv2d(960, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.decoder(features)
        out = nn.functional.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


class YoloV8SegmentationDataset(Dataset):
    """
    Example dataset for YOLOv8 polygon-based annotation files (.txt).
    Converts polygons into masks on the fly.
    """
    def __init__(self, image_dir, label_dir, transforms=None, image_size=(1024, 768)):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.image_size = image_size

        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        self.label_files = sorted(list(self.label_dir.glob("*.txt")))

    def __len__(self):
        return len(self.image_files)

    def _yolo_to_mask(self, label_path, image_size):
        """
        Convert YOLO polygon lines into a single binary mask for segmentation.
        """
        h, w = image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        with open(label_path, "r") as f:
            for line in f:
                elements = line.strip().split()
                cls, *polygon_points = elements
                polygon_points = list(map(float, polygon_points))
                polygon_points = np.array(polygon_points, dtype=np.float32).reshape(-1, 2)

                # Scale up to image size
                polygon_points[:, 0] *= w
                polygon_points[:, 1] *= h
                polygon_points = polygon_points.astype(np.int32)

                # Fill the polygon to create instance regions
                cv2.fillPoly(mask, [polygon_points], color=1)
        return mask

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.image_size)
        w, h = img.size

        label_path = self.label_files[idx]
        mask = self._yolo_to_mask(label_path, (h, w))

        if self.transforms:
            img = self.transforms(img)

        mask = torch.as_tensor(mask, dtype=torch.long)
        return img, mask


def train_mobilenetv3(
    base_dir="/home/sprice/RQ/demo.v7i.yolov8",
    num_classes=2,
    batch_size=4,
    num_epochs=200,
    learning_rate=0.001,
    patience=15
):
    """
    Example function to train a MobileNetV3-based segmentation model
    on a YOLOv8-style dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = YoloV8SegmentationDataset(
        image_dir=os.path.join(base_dir, "train/images"),
        label_dir=os.path.join(base_dir, "train/labels"),
        transforms=F.to_tensor
    )
    valid_dataset = YoloV8SegmentationDataset(
        image_dir=os.path.join(base_dir, "valid/images"),
        label_dir=os.path.join(base_dir, "valid/labels"),
        transforms=F.to_tensor
    )
    test_dataset = YoloV8SegmentationDataset(
        image_dir=os.path.join(base_dir, "test/images"),
        label_dir=os.path.join(base_dir, "test/labels"),
        transforms=F.to_tensor
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    model = MobileNetV3Segmentation(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    cpu_mem_usage = []
    gpu_mem_usage = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, masks in train_loader:
            images = images.to
