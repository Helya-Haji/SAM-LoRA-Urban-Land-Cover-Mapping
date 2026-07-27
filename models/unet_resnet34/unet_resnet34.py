import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pandas as pd
import time
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import segmentation_models_pytorch as smp

# ======================
# CONFIG
# ======================
IMAGE_SIZE = 512
NUM_CLASSES = 9
IGNORE_INDEX = 0
BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "others",       # 0 — ignored
    "bareland",     # 1
    "rangeland",    # 2
    "developed",    # 3
    "Road",        # 4
    "Tree",       # 5
    "Water",  # 6
    "Agriculture",      # 7
    "Building",      # 8
]

TRAIN_IMG = r"...\OpenEarthMap\images\train"
TRAIN_MASK = r"...\OpenEarthMap\labels\train"
VAL_IMG = r"...\OpenEarthMap\images\val"
VAL_MASK = r"...\OpenEarthMap\labels\val"

OUTPUT_DIR = r"...\unet-resnet34"
os.makedirs(OUTPUT_DIR, exist_ok=True)

history = []

# DATASET
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]

        if train:
            self.tf = A.Compose([
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(0.1, 0.2, 30, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.2),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.tf = A.Compose([
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)

        aug = self.tf(image=img, mask=mask)
        return aug["image"], aug["mask"].long()


# MODEL: U-Net ResNet34
class UNetResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

# LOSS (same)
class TverskyLoss(nn.Module):
    def __init__(self, classes, alpha=0.7, beta=0.3, ignore_index=0):
        super().__init__()
        self.classes = classes
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        valid = (target != self.ignore_index).float()

        loss = 0
        n = 0

        for c in range(self.classes):
            if c == self.ignore_index:
                continue

            p = pred[:, c] * valid
            t = (target == c).float()

            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t * valid).sum()

            tversky = (tp + 1e-5) / (tp + 0.7 * fp + 0.3 * fn + 1e-5)

            loss += (1 - tversky)
            n += 1

        return loss / n

# IoU
def compute_iou(pred, mask):
    pred = torch.argmax(pred, dim=1)
    ious = []

    for c in range(NUM_CLASSES):
        if c == IGNORE_INDEX:
            continue

        p = (pred == c)
        m = (mask == c)

        inter = (p & m).sum().float()
        union = (p | m).sum().float()

        ious.append((inter / union).item() if union > 0 else np.nan)

    return ious


# TRAIN
def train():
    train_ds = SegDataset(TRAIN_IMG, TRAIN_MASK, train=True)
    val_ds   = SegDataset(VAL_IMG, VAL_MASK, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = UNetResNet34(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = TverskyLoss(NUM_CLASSES)

    best = 0
    best_row = None

    for epoch in range(EPOCHS):

        start = time.time()

        # TRAIN
        model.train()
        train_loss = 0

        for img, mask in tqdm(train_loader):
            img, mask = img.to(DEVICE), mask.to(DEVICE)

            optimizer.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, mask)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # VAL
        model.eval()
        ious = []

        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                pred = model(img)
                ious.append(compute_iou(pred, mask))

        ious = np.nanmean(np.array(ious), axis=0)
        miou = np.nanmean(ious)

        epoch_time = time.time() - start

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "miou": miou,
            "epoch_time_sec": round(epoch_time, 2),
            "epoch_time_min": round(epoch_time, 2)
        }

        idx = 0
        for c in range(NUM_CLASSES):
            if c == IGNORE_INDEX:
                continue
            row[f"iou_{CLASS_NAMES[c]}"] = ious[idx]
            idx += 1

        history.append(row)

        print(f"Epoch {epoch+1} | Loss: {row['train_loss']:.4f} | mIoU: {miou:.4f}")

        if miou > best:
            best = miou
            best_row = row.copy()

            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, "deeplabv3_best.pth"))

    pd.DataFrame(history).to_excel(
        os.path.join(OUTPUT_DIR, "training_log.xlsx"),
        index=False
    )

    pd.DataFrame([best_row]).to_excel(
        os.path.join(OUTPUT_DIR, "best_results.xlsx"),
        index=False
    )

    print("\nBEST RESULT:", best_row)

if __name__ == "__main__":
    train()