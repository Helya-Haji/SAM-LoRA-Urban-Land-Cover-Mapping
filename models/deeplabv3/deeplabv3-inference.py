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
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


# CONFIG
IMAGE_SIZE = 512
NUM_CLASSES = 9
IGNORE_INDEX = 0
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"...\deeplabv3\deeplabv3_best.pth"
OUTPUT_DIR = r"...\deeplabv3"

VAL_IMG = r"...\OpenEarthMap\images\val"
VAL_MASK = r"...\OpenEarthMap\labels\val"

CLASS_NAMES = [
    "others","bareland","rangeland","developed",
    "Road","Tree","Water","Agriculture","Building"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# DATASET
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]

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
class DeepLabV3Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=None)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        return self.model(x)["out"]

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


def mean_iou(iou_list):
    arr = np.array(iou_list, dtype=np.float32)
    return np.nanmean(arr, axis=0), np.nanmean(arr)

# FLOPs + Params
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

# TTA INFERENCE
def tta_predict(model, img):
    # img: (B,C,H,W)

    preds = []

    with torch.no_grad():
        preds.append(model(img))

        preds.append(model(torch.flip(img, dims=[3])))  # H flip
        preds.append(model(torch.flip(img, dims=[2])))  # V flip

        hv = torch.flip(img, dims=[2,3])
        preds.append(model(hv))

    # restore flips
    preds[1] = torch.flip(preds[1], dims=[3])
    preds[2] = torch.flip(preds[2], dims=[2])
    preds[3] = torch.flip(preds[3], dims=[2,3])

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

    class_iou, miou = mean_iou(all_ious)
    return class_iou, miou


# MAIN
def main():

    # dataset
    val_ds = SegDataset(VAL_IMG, VAL_MASK)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = DeepLabV3Model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)

    # params & flops
    params = count_params(model)
    flops = compute_flops(model)

    tta_flops = flops * 4 if flops is not None else None

    print(f"\nParams (M): {params/1e6:.3f}")
    print(f"FLOPs (G): {flops:.3f}" if flops else "FLOPs: thop not installed")

    # WITHOUT TTA
    iou_no_tta, miou_no_tta = evaluate(model, val_loader, use_tta=False)


    # WITH TTA
    iou_tta, miou_tta = evaluate(model, val_loader, use_tta=True)

    # SAVE RESULTS
    rows = []

    def build_row(name, ious, miou):
        row = {"Method": name, "mIoU": miou}
        idx = 0
        for c in range(NUM_CLASSES):
            if c == IGNORE_INDEX:
                continue
            row[CLASS_NAMES[c]] = ious[idx]
            idx += 1
        return row

    rows.append(build_row("No TTA", iou_no_tta, miou_no_tta))
    rows.append(build_row("TTA", iou_tta, miou_tta))

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(OUTPUT_DIR, "final_eval_results.xlsx"), index=False)

    print("\n===== RESULTS =====")
    print("No TTA mIoU:", miou_no_tta)
    print("TTA mIoU:", miou_tta)

    print(f"\nParams (M): {params/1e6:.3f}")
    print(f"FLOPs (G): {flops:.3f}" if flops else "FLOPs: thop not installed")
    print(f"FLOPs with TTA (G): {tta_flops:.3f}" if tta_flops else "")
    print(f"FLOPs without TTA (G): {flops:.3f}")

if __name__ == "__main__":
    main()
