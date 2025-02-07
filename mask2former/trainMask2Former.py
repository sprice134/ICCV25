import os
from collections import Counter
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
import torch.optim as optim
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from pycocotools import mask as maskUtils

# ------------------------------
# 1️⃣ Define Paths
# ------------------------------
train_img_path = "/home/sprice/ICCV25/datasets/powder/train"
train_json_path = "/home/sprice/ICCV25/datasets/powder/train/_annotations.coco.json"
valid_img_path = "/home/sprice/ICCV25/datasets/powder/valid"
valid_json_path = "/home/sprice/ICCV25/datasets/powder/valid/_annotations.coco.json"

# ------------------------------
# 2️⃣ Determine the Most Common Image Size (width, height) from Train Images
# ------------------------------
def get_image_sizes(image_folder):
    sizes = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, filename)
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)  # (width, height)
            except Exception as e:
                print(f"Could not open {img_path}: {e}")
    return sizes

train_sizes = get_image_sizes(train_img_path)
most_common_size = Counter(train_sizes).most_common(1)[0][0]  # (width, height)
print(f"Most common image size (W,H): {most_common_size}")

# ------------------------------
# 3️⃣ Custom Dataset Filtering by Image Size
# ------------------------------
class FilteredCocoDetection(CocoDetection):
    """
    Only includes images that match `desired_size`.
    Each sample: (PIL_image, annotation_list, img_info).
    """
    def __init__(self, root, annFile, transform, desired_size):
        super().__init__(root, annFile, transform=None)  # Transforms are applied manually below.
        self.desired_size = desired_size  # (width, height)
        self.filtered_ids = []
        for img_id in self.ids:
            info = self.coco.imgs[img_id]
            if (info["width"], info["height"]) == desired_size:
                self.filtered_ids.append(img_id)
        print(f"Filtered dataset: {len(self.filtered_ids)} images "
              f"(out of {len(self.ids)}) with size {self.desired_size}")
        self.transform = transform

    def __len__(self):
        return len(self.filtered_ids)

    def __getitem__(self, idx):
        img_id = self.filtered_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        image_pil = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image_pil = self.transform(image_pil)
        return image_pil, anns, img_info

# ------------------------------
# 4️⃣ Dataset & Collate Function
# ------------------------------
# Note: transforms.Resize expects (height, width), so swap (width, height) -> (height, width)
target_size = (most_common_size[1], most_common_size[0])
resize_transform = transforms.Resize(target_size)

train_dataset = FilteredCocoDetection(train_img_path, train_json_path, resize_transform, most_common_size)
valid_dataset = FilteredCocoDetection(valid_img_path, valid_json_path, resize_transform, most_common_size)

def simple_collate_fn(batch):
    images, annotations, infos = [], [], []
    for b in batch:
        images.append(b[0])
        annotations.append(b[1])
        infos.append(b[2])
    return images, annotations, infos

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, 
                          collate_fn=simple_collate_fn, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, 
                          collate_fn=simple_collate_fn, num_workers=4)

print(f"Train samples: {len(train_dataset)}")
print(f"Valid samples: {len(valid_dataset)}")

# ------------------------------
# 5️⃣ Create Instance Masks and Class Labels from COCO Annotations
# ------------------------------
def ann_to_mask_and_label(ann, height, width):
    """
    Converts a single annotation dict into a binary mask and a category id.
    Handles both polygon and RLE segmentations.
    """
    seg = ann.get("segmentation", [])
    if not seg:
        return None, None
    if isinstance(seg, list):
        # Polygon format
        rle = maskUtils.frPyObjects(seg, height, width)
        rle = maskUtils.merge(rle)
        mask = maskUtils.decode(rle)
    elif isinstance(seg, dict):
        # RLE format
        mask = maskUtils.decode(seg)
    else:
        return None, None
    cat_id = ann.get("category_id", 0)
    return mask, cat_id

def create_instance_labels(anns, height, width):
    """
    For each annotation in anns, creates:
      - A binary mask of shape [H, W]
      - A class label
    Returns:
      - masks_tensor: [num_instances, H, W] (float tensor)
      - classes_tensor: [num_instances] (long tensor)
    If no valid annotations exist, a dummy mask is created.
    """
    mask_list = []
    class_list = []
    for ann in anns:
        mask, cat_id = ann_to_mask_and_label(ann, height, width)
        if mask is None:
            continue
        mask_list.append(mask)
        class_list.append(cat_id)
    if not mask_list:
        # Create a dummy background mask if no annotations are valid.
        mask_list = [np.zeros((height, width), dtype=np.uint8)]
        class_list = [0]
    # Convert masks to float instead of bool so that grid_sample can operate on them.
    masks_tensor = torch.from_numpy(np.stack(mask_list, axis=0)).float()  # [N, H, W]
    classes_tensor = torch.tensor(class_list, dtype=torch.long)          # [N]
    return masks_tensor, classes_tensor

# ------------------------------
# 6️⃣ Load the Model & Image Processor
# ------------------------------
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-instance")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------
# 7️⃣ Training & Evaluation Functions (Instance Segmentation)
# ------------------------------
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for pil_images, ann_lists, infos in dataloader:
        batch_pixel_values = []
        batch_mask_labels = []
        batch_class_labels = []
        for image_pil, anns in zip(pil_images, ann_lists):
            # Convert the PIL image to a tensor using the processor.
            processed = processor(images=image_pil, return_tensors="pt")
            pixel_values_single = processed["pixel_values"].squeeze(0)  # [3, H, W]
            # Note: image_pil.size returns (width, height)
            w, h = image_pil.size  
            masks_tensor, classes_tensor = create_instance_labels(anns, h, w)
            batch_pixel_values.append(pixel_values_single)
            batch_mask_labels.append(masks_tensor)
            batch_class_labels.append(classes_tensor)
        # Stack all images to create a batch tensor.
        pixel_values = torch.stack(batch_pixel_values, dim=0).to(device)  # [B, 3, H, W]
        # Move the mask and class label tensors to the device.
        batch_mask_labels = [mask.to(device) for mask in batch_mask_labels]
        batch_class_labels = [cls.to(device) for cls in batch_class_labels]
        
        optimizer.zero_grad()
        outputs = model(
            pixel_values=pixel_values,
            mask_labels=batch_mask_labels,     # List of [num_instances, H, W]
            class_labels=batch_class_labels     # List of [num_instances]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for pil_images, ann_lists, infos in dataloader:
            batch_pixel_values = []
            batch_mask_labels = []
            batch_class_labels = []
            for image_pil, anns in zip(pil_images, ann_lists):
                processed = processor(images=image_pil, return_tensors="pt")
                pixel_values_single = processed["pixel_values"].squeeze(0)
                w, h = image_pil.size
                masks_tensor, classes_tensor = create_instance_labels(anns, h, w)
                batch_pixel_values.append(pixel_values_single)
                batch_mask_labels.append(masks_tensor)
                batch_class_labels.append(classes_tensor)
            pixel_values = torch.stack(batch_pixel_values, dim=0).to(device)
            batch_mask_labels = [mask.to(device) for mask in batch_mask_labels]
            batch_class_labels = [cls.to(device) for cls in batch_class_labels]
            
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=batch_mask_labels,
                class_labels=batch_class_labels
            )
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ------------------------------
# 8️⃣ Training Loop with Early Stopping
# ------------------------------
epochs = 100
patience = 10  # Number of epochs with no improvement after which training will be stopped.
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = evaluate(model, valid_loader, device)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Check if validation loss improved.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model.
        torch.save(model.state_dict(), "mask2former_small_best_instance_seg.pth")
        print("Validation loss improved. Saving model.")
    else:
        patience_counter += 1
        print(f"No improvement in validation loss for {patience_counter} epoch(s).")
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# ------------------------------
# 9️⃣ Save Final Model (if not already saved by early stopping)
# ------------------------------
torch.save(model.state_dict(), "mask2former_small_instance_seg.pth")
print("Model saved successfully!")
