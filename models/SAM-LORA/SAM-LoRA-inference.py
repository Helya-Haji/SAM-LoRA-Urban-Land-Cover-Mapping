import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from ptflops import get_model_complexity_info


from train import (
    IMAGE_SIZE,
    NUM_CLASSES,
    IGNORE_INDEX,
    CLASS_NAMES,
    SAM_LoRA,
    SAM_CHECKPOINT
)

VAL_IMG    = r"...\OpenEarthMap\images\val"
VAL_MASK   = r"...\OpenEarthMap\labels\val"
MODEL_PATH = r"...\models\sam_vit_b_01ec64.pth"
OUTPUT_DIR = r"...\sam_results"

# DATASET
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.files    = [f for f in os.listdir(img_dir) if f.endswith(".tif")]

        self.tf = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img  = cv2.imread(os.path.join(self.img_dir, name))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)
        aug  = self.tf(image=img, mask=mask)
        return aug["image"], aug["mask"].long()


# TTA PREDICTION
def predict_tta(model, img_tensor):
    """Average softmax over 4 orientations."""
    preds = []
    preds.append(torch.softmax(model(img_tensor), dim=1))
    preds.append(torch.softmax(model(torch.flip(img_tensor, [3])), dim=1).flip(3))
    preds.append(torch.softmax(model(torch.flip(img_tensor, [2])), dim=1).flip(2))
    preds.append(torch.softmax(model(torch.flip(img_tensor, [2, 3])), dim=1).flip([2, 3]))
    return torch.stack(preds).mean(0)


# METRICS
def compute_iou(pred_prob, mask, num_classes, ignore_index=0):
    """pred_prob: [B, C, H, W] softmax probabilities."""
    pred = torch.argmax(pred_prob, dim=1)
    ious = []
    for c in range(num_classes):
        if c == ignore_index:
            ious.append(np.nan)
            continue
        valid = mask != ignore_index
        p     = (pred == c) & valid
        m     = (mask == c) & valid
        inter = (p & m).sum().item()
        union = (p | m).sum().item()
        ious.append(inter / union if union > 0 else np.nan)
    return ious


def update_confusion_matrix(conf_mat, pred_prob, mask, num_classes, ignore_index=0):
    """Accumulate confusion matrix (ignores ignore_index pixels)."""
    pred  = torch.argmax(pred_prob, dim=1).cpu().numpy().flatten()
    gt    = mask.cpu().numpy().flatten()
    valid = gt != ignore_index
    pred  = pred[valid]
    gt    = gt[valid]
    np.add.at(conf_mat, (gt, pred), 1)

# EVALUATION LOOP
def evaluate(model, loader, device, use_tta=False):
    model.eval()
    all_ious = []
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for img, mask in tqdm(loader, desc="TTA" if use_tta else "Standard"):
            img  = img.to(device)
            mask = mask.to(device)

            if use_tta:
                pred_prob = predict_tta(model, img)
            else:
                pred_prob = torch.softmax(model(img), dim=1)

            all_ious.append(compute_iou(pred_prob, mask, NUM_CLASSES, IGNORE_INDEX))
            update_confusion_matrix(conf_mat, pred_prob, mask, NUM_CLASSES, IGNORE_INDEX)

    all_ious  = np.array(all_ious, dtype=float)
    class_iou = np.nanmean(all_ious, axis=0)
    miou      = float(np.nanmean(class_iou))

    return miou, class_iou, conf_mat


# SAVE EXCEL RESULTS
def save_excel(miou_std, class_iou_std, miou_tta, class_iou_tta, path):
    rows = []

    # Standard row
    row_std = {"mode": "Standard (no TTA)", "mIoU": round(miou_std, 4)}
    for c in range(NUM_CLASSES):
        v = class_iou_std[c]
        row_std[CLASS_NAMES[c]] = round(float(v), 4) if not np.isnan(v) else None
    rows.append(row_std)

    # TTA row
    row_tta = {"mode": "TTA", "mIoU": round(miou_tta, 4)}
    for c in range(NUM_CLASSES):
        v = class_iou_tta[c]
        row_tta[CLASS_NAMES[c]] = round(float(v), 4) if not np.isnan(v) else None
    rows.append(row_tta)

    pd.DataFrame(rows).to_excel(path, index=False)
    print(f"Saved Excel results → {path}")


# SAVE CONFUSION MATRIX
def save_confusion_matrix(conf_mat, title, save_path):


    # Remove background class
    valid_idx = [c for c in range(NUM_CLASSES) if c != IGNORE_INDEX]

    valid_names = [CLASS_NAMES[c] for c in valid_idx]

    cm = conf_mat[np.ix_(valid_idx, valid_idx)]

    # Row-wise normalization (%)
    row_sums = cm.sum(axis=1, keepdims=True)

    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0
    )

    cm_norm *= 100.0

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))


    # Raw counts
    sns.heatmap(
        cm,
        ax=axes[0],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=valid_names,
        yticklabels=valid_names
    )

    axes[0].set_title(
        f"{title} - Raw Counts",
        fontsize=14
    )

    axes[0].set_xlabel(
        "Predicted Class",
        fontsize=12
    )

    axes[0].set_ylabel(
        "Ground Truth Class",
        fontsize=12
    )

    axes[0].tick_params(
        axis="x",
        rotation=45
    )

    axes[0].tick_params(
        axis="y",
        rotation=0
    )

    # Normalized (%)
    sns.heatmap(
        cm_norm,
        ax=axes[1],
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=valid_names,
        yticklabels=valid_names,
        vmin=0,
        vmax=100
    )

    axes[1].set_title(
        f"{title} - Normalized (%)",
        fontsize=14
    )

    axes[1].set_xlabel(
        "Predicted Class",
        fontsize=12
    )

    axes[1].set_ylabel(
        "Ground Truth Class",
        fontsize=12
    )

    axes[1].tick_params(
        axis="x",
        rotation=45
    )

    axes[1].tick_params(
        axis="y",
        rotation=0
    )

    plt.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved confusion matrix → {save_path}")


# PARAM COUNT
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# FLOPS
def compute_flops(model):
    model.eval()
    macs, params = get_model_complexity_info(
        model,
        (3, IMAGE_SIZE, IMAGE_SIZE),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )
    return macs, params


# MAIN
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    #  Load model 
    model = SAM_LoRA(checkpoint=SAM_CHECKPOINT, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    #  Data 
    dataset = SegDataset(VAL_IMG, VAL_MASK)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    # Standard evaluation
    print("\n[1/2] Standard evaluation (no TTA)...")
    miou_std, class_iou_std, conf_std = evaluate(
        model, loader, device, use_tta=False)

    print(f"\n  mIoU (standard) : {miou_std:.4f}")
    for c in range(NUM_CLASSES):
        if c == IGNORE_INDEX:
            continue
        v = class_iou_std[c]
        print(f"  {CLASS_NAMES[c]:15s}: {v:.4f}" if not np.isnan(v) else
              f"  {CLASS_NAMES[c]:15s}: NaN")

    # TTA evaluation 
    print("\n[2/2] TTA evaluation...")
    miou_tta, class_iou_tta, conf_tta = evaluate(
        model, loader, device, use_tta=True)

    print(f"\n  mIoU (TTA)      : {miou_tta:.4f}")
    for c in range(NUM_CLASSES):
        if c == IGNORE_INDEX:
            continue
        v = class_iou_tta[c]
        print(f"  {CLASS_NAMES[c]:15s}: {v:.4f}" if not np.isnan(v) else
              f"  {CLASS_NAMES[c]:15s}: NaN")

    # Save Excel 
    excel_path = os.path.join(OUTPUT_DIR, "eval_results.xlsx")
    save_excel(miou_std, class_iou_std, miou_tta, class_iou_tta, excel_path)

    # Save confusion matrices 
    save_confusion_matrix(
        conf_std,
        title="Standard",
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_standard.png")
    )
    save_confusion_matrix(
        conf_tta,
        title="TTA",
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_tta.png")
    )

    # Parameters 
    total, trainable = count_params(model)
    print("\n================ PARAMETERS ================")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")

    # FLOPs
    flops, params_flops = compute_flops(model)
    print("\n================ FLOPs ================")
    print(f"  FLOPs (MACs)   : {flops}")
    print(f"  Params (ptflops): {params_flops}")

    print("\nDone.")


if __name__ == "__main__":
    main()