"""
Subway Surfers CNN training pipeline using supervised learning on collected gameplay data.

This script defines the full end-to-end training workflow for an imitation learning model:

1. Dataset Construction
   - Loads labeled gameplay screenshots from disk (jump, roll, left, right, noop).
   - Applies on-the-fly data augmentation (happens during training):
     • Each image is duplicated with a horizontally flipped version.
     • Directional labels are remapped using a flip map (left ↔ right).
   - Crops a fixed game region using a predefined `game_region`.
   - Converts images to grayscale and resizes them to 96×96 for model input.
   - Outputs tensors suitable for PyTorch training.

2. Model Architecture (SubwayCNN)
   - Convolutional feature extractor:
     • Conv2d(1 → 32, kernel=8, stride=4)
     • Conv2d(32 → 64, kernel=4, stride=2)
     • Conv2d(64 → 64, kernel=3, stride=1)
   - Fully connected head:
     • Flatten → Linear(4096 → 512) → ReLU
     • Linear(512 → 5 actions)
   - Designed for low-resolution spatial feature extraction from gameplay frames.

3. Training Setup (This was chosen by Claude 😅)
   - Loss function: CrossEntropyLoss (multi-class classification over 5 actions)
   - Optimizer: Adam (lr = 1e-3)
   - Batch size: 32
   - Epochs: 20
   - Train/validation split: 85% / 15%
   - Device: CPU (can be switched to CUDA if available)

4. Training Loop
   - Forward pass → loss computation → backpropagation → optimizer step
   - Tracks training accuracy and loss per batch using tqdm progress bars
   - Evaluates validation accuracy after each epoch (no gradient computation)

5. Model Checkpointing
   - Saves best-performing model based on validation accuracy
   - Stores:
     • model_state_dict
     • optimizer_state_dict
     • epoch number
     • validation accuracy
     • architecture metadata (in_channels, num_actions)

This pipeline implements a supervised imitation learning approach where
the model learns to map visual game states directly to discrete action labels.
"""

# %%
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms

screen_width = 1920
screen_height = 1080
width=400
height=500
height_add=150
game_region = {
    'width': width,
    'height': height,
    'left': (screen_width - width) // 2,
    'top': (screen_height + height_add - height) // 2
}
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import time
import random
from tqdm import tqdm

# %%





BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATHS = {
    "jump": os.path.join(DATASET_DIR, "jump"),
    "roll": os.path.join(DATASET_DIR, "roll"),
    "left": os.path.join(DATASET_DIR, "left"),
    "right": os.path.join(DATASET_DIR, "right"),
    "noop": os.path.join(DATASET_DIR, "noop")
}
ACTIONS={
    "roll": 0,
    "left": 1,
    "noop": 2,
    "right": 3,
    "jump": 4
}
SAVE_PATH = os.path.join("..", "models", "subway_cnn.pth")  # ← needs filename
os.makedirs(os.path.join("..", "models"), exist_ok=True)
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 1e-3
VAL_SPLIT    = 0.15       # 15% of data for test
DEVICE       = "cpu"

# %%

class SubwayCNN(nn.Module):
    def __init__(self, in_channels=1, num_actions=5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # → 23×23
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # → 10×10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # → 8×8
            nn.ReLU(),
            nn.Flatten(),                                          # → 4096
            nn.Linear(4096, 512),
            nn.ReLU(),
        )
        self.head = nn.Linear(512, num_actions)

    def forward(self, x):
        return self.head(self.backbone(x))

# %%
class SubwayDataset:
    def __init__(self,dataset_dir=DATASET_DIR,transform=None):
        self.transform=transform                        
        self.dataset_dir=dataset_dir
        self.class_to_idx = ACTIONS
        self.flip_map = {
    1: 3,  # left → right
    3: 1   # right → left
}  
        self.samples = []

        for class_name in self.class_to_idx:
            folder = os.path.join(dataset_dir, class_name)

            for filename in os.listdir(folder):
                path = os.path.join(folder, filename)

                label = self.class_to_idx[class_name]

                # original
                self.samples.append((path, label, False))

                # flipped copy
                self.samples.append((path, label, True))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        
        path, label, to_flip = self.samples[idx]

        image = Image.open(path)

        if to_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = self.flip_map.get(label, label)  
        if self.transform:
            image = self.transform(image)
        

    
       
        return image, label

# %%

class CropRegionTransformGameRegion:
    def __call__(self, img):
        left = game_region["left"]
        top = game_region["top"]
        right = left + game_region["width"]
        bottom = top + game_region["height"]

        return img.crop((left, top, right, bottom))
    

# %%

# ── dataset ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    CropRegionTransformGameRegion(),
    transforms.Grayscale(),
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

dataset    = SubwayDataset(dataset_dir=DATASET_DIR, transform=transform)
val_size   = int(VAL_SPLIT * len(dataset))
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

# %%
print(len(dataset))

# %%
model     = SubwayCNN(in_channels=1, num_actions=5).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# %%
if __name__ == "__main__":
    # ── training loop ──────────────────────────────────────────────────────────
    best_val_acc = 0.0
    for epoch in range(EPOCHS):

        # train
        model.train()
        total_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS}", leave=False)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

        train_acc = correct / total

        # validate
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="  validating", leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":     val_acc,
                "in_channels": 1,
                "num_actions": 5,
            }, SAVE_PATH)
            print(f"  saved (best val acc: {val_acc:.3f})")

    print(f"\nDone. Best val acc: {best_val_acc:.3f} → {SAVE_PATH}")



