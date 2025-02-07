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
import optuna
import json
import pandas as pd

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
# transforms.Resize expects (height, width), so swap (width, height) -> (height, width)
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
        rle = maskUtils.frPyObjects(seg, height, width)
        rle = maskUtils.merge(rle)
        mask = maskUtils.decode(rle)
    elif isinstance(seg, dict):
        mask = maskUtils.decode(seg)
    else:
        return None, None
    cat_id = ann.get("category_id", 0)
    return mask, cat_id

def create_instance_labels(anns, height, width):
    """
    For each annotation in anns, creates:
      - A binary mask of shape [H, W]
      - A class label.
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
        mask_list = [np.zeros((height, width), dtype=np.uint8)]
        class_list = [0]
    masks_tensor = torch.from_numpy(np.stack(mask_list, axis=0)).float()
    classes_tensor = torch.tensor(class_list, dtype=torch.long)
    return masks_tensor, classes_tensor

# ------------------------------
# 6️⃣ Global Settings for Tuning
# ------------------------------

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 7️⃣ Define the Objective Function for Hyperparameter Tuning with Optuna
# ------------------------------
def objective(trial):
    # --- Hyperparameters to tune ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    max_epochs = trial.suggest_int("max_epochs", 25, 150, step=25)
    # Remove tuning of patience; set it to a fixed value.
    patience = 15
    
    # --- Create DataLoaders with the tuned batch size ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=simple_collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                              collate_fn=simple_collate_fn, num_workers=4)
    
    # --- Initialize Model ---
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-instance")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        # --- Training Loop ---
        for pil_images, ann_lists, infos in train_loader:
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
            
            optimizer.zero_grad()
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=batch_mask_labels,
                class_labels=batch_class_labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for pil_images, ann_lists, infos in valid_loader:
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
                total_val_loss += loss.item()
        val_loss = total_val_loss / len(valid_loader)
        trial.report(val_loss, epoch)
        
        # Early stopping check using fixed patience.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    return best_val_loss

# ------------------------------
# 8️⃣ Run the Hyperparameter Study with Optuna
# ------------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# ------------------------------
# 9️⃣ Save the Study Results for Later Use
# ------------------------------
# Save the best trial as JSON
best_trial_dict = {
    "trial_number": trial.number,
    "value": trial.value,
    "params": trial.params,
    "datetime_start": str(trial.datetime_start),
    "datetime_complete": str(trial.datetime_complete)
}
with open("best_trial_v2.json", "w") as f:
    json.dump(best_trial_dict, f, indent=2)
print("Saved best trial to best_trial.json")

# Save all trial results to CSV using a DataFrame.
df = study.trials_dataframe()
df.to_csv("study_results_v2.csv", index=False)
print("Saved full study results to study_results.csv")

# Optionally, retrain a final model on the full training set using these best parameters.
