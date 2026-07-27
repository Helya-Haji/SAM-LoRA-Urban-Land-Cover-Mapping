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

# CONFIG
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


# OUTPUT
OUTPUT_DIR = r"...\unet"
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

# MODEL
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)

# LOSS
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

# IoU FIXED
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

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4, 
        pin_memory=True,
        persistent_workers=(4 > 0) 
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2, 
        pin_memory=True,
        persistent_workers=(2 > 0) 
    )

    model = UNet(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = TverskyLoss(NUM_CLASSES)

    best = 0
    best_row = None

    for epoch in range(EPOCHS):

        epoch_start = time.time()

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

        # VALIDATION
        model.eval()
        ious = []

        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                pred = model(img)
                ious.append(compute_iou(pred, mask))

        ious = np.nanmean(np.array(ious), axis=0)
        miou = np.nanmean(ious)

        epoch_time = time.time() - epoch_start
        epoch_time_min = epoch_time 

        # LOG
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "miou": miou,
            "epoch_time_sec": round(epoch_time, 2),
            "epoch_time_min": round(epoch_time_min, 2)
        }

        iou_idx = 0
        for c in range(NUM_CLASSES):
            if c == IGNORE_INDEX:
                continue
            row[f"iou_{CLASS_NAMES[c]}"] = ious[iou_idx]
            iou_idx += 1



        history.append(row)

        print(f"Epoch {epoch+1}")
        print("Train Loss:", train_loss / len(train_loader))
        print("mIoU:", miou)

        # SAVE BEST MODEL
        if miou > best:
            best = miou
            best_row = row.copy()

            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_DIR, "unet_best.pth")
            )

            print("Saved best UNet")

        # SAVE LOG EVERY EPOCH
        pd.DataFrame(history).to_excel(
            os.path.join(OUTPUT_DIR, "training_log.xlsx"),
            index=False
        )

    # FINAL REPORT
    best_df = pd.DataFrame([best_row])
    best_df.to_excel(
        os.path.join(OUTPUT_DIR, "best_results.xlsx"),
        index=False
    )

    print("\n===== FINAL BEST RESULT =====")
    print(best_row)

if __name__ == "__main__":
    train()
