import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

# CONFIG
IMAGE_SIZE = 512
NUM_CLASSES = 9
IGNORE_INDEX = 0
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"...\unet\unet_best.pth"
OUTPUT_DIR = r"...\unet"

VAL_IMG = r"...\OpenEarthMap\images\val"
VAL_MASK = r"E:...\OpenEarthMap\labels\val"

CLASS_NAMES = [
    "others","bareland","rangeland","developed",
    "Road","Tree","Water","Agriculture","Building"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# DATASET
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
        self.img_dir = img_dir
        self.mask_dir = mask_dir

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

# MODEL (UNet from scratch)
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

# METRICS
def compute_iou_per_class(pred, mask):
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


def mean_iou(all_ious):
    arr = np.array(all_ious, dtype=np.float32)
    return np.nanmean(arr, axis=0), np.nanmean(arr)

# PARAMS + FLOPs
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model):
    try:
        from thop import profile
        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except:
        return None

# TTA
def tta_predict(model, img):
    preds = []

    preds.append(model(img))
    preds.append(model(torch.flip(img, dims=[3])))
    preds.append(model(torch.flip(img, dims=[2])))

    hv = torch.flip(img, dims=[2, 3])
    preds.append(model(hv))

    preds[1] = torch.flip(preds[1], dims=[3])
    preds[2] = torch.flip(preds[2], dims=[2])
    preds[3] = torch.flip(preds[3], dims=[2, 3])

    return torch.mean(torch.stack(preds), dim=0)

# EVAL
def evaluate(model, loader, use_tta=False):
    model.eval()
    all_ious = []

    with torch.no_grad():
        for img, mask in loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            if use_tta:
                pred = tta_predict(model, img)
            else:
                pred = model(img)

            all_ious.append(compute_iou_per_class(pred, mask))

    return mean_iou(all_ious)

# MAIN
def main():

    val_ds = SegDataset(VAL_IMG, VAL_MASK)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=True)

    # Params + FLOPs
    params = count_params(model)
    flops = compute_flops(model)

    tta_flops = flops * 4 if flops is not None else None

    print(f"\nParams (M): {params/1e6:.3f}")
    print(f"FLOPs (G): {flops:.3f}" if flops else "FLOPs not computed")

    # WITHOUT TTA
    iou_no_tta, miou_no_tta = evaluate(model, val_loader, use_tta=False)

    # WITH TTA
    iou_tta, miou_tta = evaluate(model, val_loader, use_tta=True)

    # SAVE
    rows = []

    def make_row(name, ious, miou):
        row = {"Method": name, "mIoU": miou}
        idx = 0
        for c in range(NUM_CLASSES):
            if c == IGNORE_INDEX:
                continue
            row[CLASS_NAMES[c]] = ious[idx]
            idx += 1
        return row

    rows.append(make_row("No TTA", iou_no_tta, miou_no_tta))
    rows.append(make_row("TTA", iou_tta, miou_tta))

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(OUTPUT_DIR, "unet_inference_results.xlsx"), index=False)

    print("\n===== RESULTS =====")
    print("No TTA mIoU:", miou_no_tta)
    print("TTA mIoU:", miou_tta)

    print(f"Params (M): {params/1e6:.3f}")
    print(f"FLOPs (G): {flops:.3f}" if flops else "")
    print(f"FLOPs with TTA (G): {tta_flops:.3f}" if tta_flops else "")

if __name__ == "__main__":
    main()