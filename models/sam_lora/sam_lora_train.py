import os
import math
import random
import gc

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import warnings
warnings.filterwarnings("ignore")

import cv2
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from segment_anything import sam_model_registry
import time

torch.backends.cudnn.benchmark = True


IMAGE_SIZE    = 512
NUM_CLASSES   = 9
IGNORE_INDEX  = 0
BATCH_SIZE    = 4       
EPOCHS        = 60
LR            = 1e-4
WEIGHT_DECAY  = 1e-3     
WARMUP_EPOCHS = 5
PATIENCE      = 12

BARELAND_CLS         = 1
BARELAND_WEIGHT_MULT = 3.0   
BARELAND_PIX_THRESH  = 0.05 
BARELAND_SAMPLE_MULT = 2      

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

TRAIN_IMG      = r"...\OpenEarthMap\images\train"
TRAIN_MASK     = r"...\OpenEarthMap\labels\train"
VAL_IMG        = r"...\OpenEarthMap\images\val"
VAL_MASK       = r"...\OpenEarthMap\labels\val"
SAM_CHECKPOINT = r"...\models\sam_vit_b_01ec64.pth"
OUTPUT_DIR     = r"...\SAM_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)



# CLASS WEIGHTS

def compute_class_weights(mask_dir, num_classes, ignore_index=0):
    print("Computing class weights (ignoring label 0)...")
    pixel_count = np.zeros(num_classes, dtype=np.float64)
    files = [f for f in os.listdir(mask_dir) if f.endswith(".tif")]
    for f in tqdm(files):
        mask = cv2.imread(os.path.join(mask_dir, f), 0)
        if mask is None:
            continue
        for c in range(num_classes):
            pixel_count[c] += np.sum(mask == c)
    pixel_count[ignore_index] = 0.0
    valid_mask = pixel_count > 0
    total      = pixel_count[valid_mask].sum()
    weights    = np.zeros(num_classes, dtype=np.float64)
    weights[valid_mask] = total / (valid_mask.sum() * pixel_count[valid_mask] + 1e-6)
    weights = (weights / (weights.sum() + 1e-6)) * (num_classes - 1)
    weights[ignore_index] = 0.0
    print("Base class weights:", np.round(weights, 4))
    return torch.tensor(weights, dtype=torch.float32)


# WEIGHTED SAMPLER 

def make_weighted_sampler(mask_dir, files, class_weights_np):
    print("Building weighted sampler (bareland-boosted)...")
    sample_weights = []
    for f in tqdm(files):
        mask = cv2.imread(os.path.join(mask_dir, f), 0)
        if mask is None:
            sample_weights.append(1.0)
            continue
        bareland_frac = np.sum(mask == BARELAND_CLS) / (mask.size + 1e-6)
        classes_present = np.unique(mask)
        valid = [c for c in classes_present
                 if c < len(class_weights_np) and c != IGNORE_INDEX]
        w = float(max(class_weights_np[c] for c in valid)) if valid else 1.0
        if bareland_frac >= BARELAND_PIX_THRESH:
            w *= BARELAND_SAMPLE_MULT
        sample_weights.append(w)
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# DATASET

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512, train=True, copy_paste=None):       
        self.img_dir    = img_dir
        self.mask_dir   = mask_dir
        self.files      = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
        self.copy_paste = copy_paste

        if train:
            self.transform = A.Compose([
                A.Resize(size, size,
                         interpolation=cv2.INTER_LINEAR,
                         mask_interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                                   rotate_limit=30,
                                   border_mode=cv2.BORDER_REFLECT, p=0.5),

                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=40,
                        val_shift_limit=20
                    ),
                    A.CLAHE(clip_limit=4.0),
                    A.RandomGamma(gamma_limit=(60,140)),
                ], p=0.6),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(size, size,
                         interpolation=cv2.INTER_LINEAR,
                         mask_interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img  = cv2.imread(os.path.join(self.img_dir, name))
        if img is None:
            raise FileNotFoundError(f"Image not found: {name}")
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {name}")
        if self.copy_paste is not None:
            img, mask = self.copy_paste(img, mask)
        aug = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].long()


# FOCAL TVERSKY LOSS
class TverskyLoss(nn.Module):

    def __init__(
        self,
        classes,
        alpha=0.7,
        beta=0.3,
        smooth=1e-5,
        ignore_index=0
    ):
        super().__init__()

        self.classes = classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):

        pred = torch.softmax(pred, dim=1)

        valid = (target != self.ignore_index).float()

        loss = 0.0
        n = 0

        for c in range(self.classes):

            if c == self.ignore_index:
                continue

            p = pred[:, c] * valid

            t = (target == c).float()

            tp = (p * t).sum()

            fp = (p * (1 - t)).sum()

            fn = ((1 - p) * t * valid).sum()

            tversky = (
                tp + self.smooth
            ) / (
                tp
                + self.alpha * fp
                + self.beta * fn
                + self.smooth
            )

            loss += (1 - tversky)

            n += 1

        return loss / n

# LORA

class LoRA_qkv(nn.Module):
    def __init__(self, qkv, r=16):              # rank
        super().__init__()
        self.qkv = qkv
        dim = qkv.in_features
        self.qA = nn.Linear(dim, r, bias=False)
        self.qB = nn.Linear(r, dim, bias=False)
        self.vA = nn.Linear(dim, r, bias=False)
        self.vB = nn.Linear(r, dim, bias=False)
        nn.init.zeros_(self.qB.weight)
        nn.init.zeros_(self.vB.weight)

    def forward(self, x):
        qkv        = self.qkv(x)
        B, H, W, C = qkv.shape
        dim        = C // 3
        qkv        = qkv.view(B, H, W, 3, dim)
        q, k, v    = qkv[:,:,:,0,:], qkv[:,:,:,1,:], qkv[:,:,:,2,:]
        q = q + self.qB(self.qA(x))
        v = v + self.vB(self.vA(x))
        return torch.stack([q, k, v], dim=3).view(B, H, W, 3 * dim)


class LoRA_MLP(nn.Module):
    def __init__(self, linear, r=16):                   # rank
        super().__init__()
        self.linear = linear
        dim = linear.in_features
        self.A = nn.Linear(dim, r, bias=False)
        self.B = nn.Linear(r, linear.out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.linear(x) + self.B(self.A(x))


# ASPP + DECODER

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        gp = F.interpolate(self.global_pool(x), size=x.shape[2:],
                           mode="bilinear", align_corners=False)
        return self.project(torch.cat(
            [self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x), gp], dim=1))


class DeepLabDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch, num_classes):
        super().__init__()
        self.aspp = ASPP(in_ch, 256)
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(304, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(0.3))
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Dropout2d(0.2))
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, neck_out, x_spatial):
        high = F.interpolate(self.aspp(neck_out), scale_factor=2,
                             mode="bilinear", align_corners=False)
        low  = F.interpolate(self.skip_proj(x_spatial), size=high.shape[2:],
                             mode="bilinear", align_corners=False)
        x = self.conv1(torch.cat([high, low], dim=1))
        x = self.conv2(F.interpolate(x, scale_factor=2,
                                     mode="bilinear", align_corners=False))
        return self.out(F.interpolate(x, scale_factor=2,
                                      mode="bilinear", align_corners=False))


# MODEL

class SAM_LoRA(nn.Module):
    def __init__(self, checkpoint, num_classes, lora_r=16):              # rank
        super().__init__()
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        self.encoder = sam.image_encoder

        for p in self.encoder.parameters():
            p.requires_grad = False

        # Inject LoRA into every block
        for blk in self.encoder.blocks:
            blk.attn.qkv = LoRA_qkv(blk.attn.qkv, r=lora_r)
            blk.mlp.lin1 = LoRA_MLP(blk.mlp.lin1, r=lora_r)
            blk.mlp.lin2 = LoRA_MLP(blk.mlp.lin2, r=lora_r)

        # Unfreeze last 4 blocks fully  (same as v14) + add DropPath
        n = len(self.encoder.blocks)
        for i, blk in enumerate(self.encoder.blocks):
            if i >= n - 4:                                   # blocks
                for p in blk.parameters():
                    p.requires_grad = True

        neck_ch = self.encoder.neck[0].out_channels
        vit_ch  = self.encoder.blocks[0].attn.qkv.qkv.in_features
        self.decoder = DeepLabDecoder(neck_ch, vit_ch, num_classes)

    def forward(self, x):
        x = self.encoder.patch_embed(x)
        pos = self.encoder.pos_embed
        if pos is not None:
            pos = F.interpolate(
                pos.permute(0, 3, 1, 2),
                size=(x.shape[1], x.shape[2]),
                mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)
            x = x + pos
        for blk in self.encoder.blocks:
            x = blk(x)
        x_spatial = x.permute(0, 3, 1, 2)
        neck_out   = self.encoder.neck(x_spatial)
        out = self.decoder(neck_out, x_spatial)
        return F.interpolate(out, size=(IMAGE_SIZE, IMAGE_SIZE),
                             mode="bilinear", align_corners=False)


# LR SCHEDULER

def build_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# METRICS

def compute_iou(pred, mask, num_classes, ignore_index=0):
    pred = torch.argmax(pred, dim=1)
    ious = []
    for c in range(num_classes):
        if c == ignore_index:
            ious.append(float("nan"))
            continue
        valid = mask != ignore_index
        p     = (pred == c) & valid
        m     = (mask == c) & valid
        inter = (p & m).sum().float()
        union = (p | m).sum().float()
        ious.append((inter / union).item() if union > 0 else float("nan"))
    return ious


# TEST-TIME AUGMENTATION 

def predict_tta(model, img_tensor):
    """
    img_tensor: [B, C, H, W]
    Returns averaged softmax logits [B, NUM_CLASSES, H, W]
    """
    preds = []
    # Original
    preds.append(torch.softmax(model(img_tensor), dim=1))
    # Horizontal flip
    preds.append(torch.softmax(model(torch.flip(img_tensor, [3])), dim=1).flip(3))
    # Vertical flip
    preds.append(torch.softmax(model(torch.flip(img_tensor, [2])), dim=1).flip(2))
    # Both flips
    preds.append(torch.softmax(model(torch.flip(img_tensor, [2, 3])), dim=1).flip([2, 3]))
    return torch.stack(preds).mean(0)


def evaluate_with_tta(model, val_loader, device, num_classes, ignore_index):
    """Run full validation with TTA. Use for final paper numbers."""
    model.eval()
    iou_list = []
    with torch.no_grad():
        for img, mask in tqdm(val_loader, desc="TTA evaluation"):
            img  = img.to(device)
            mask = mask.to(device)
            pred = predict_tta(model, img)
            iou_list.append(compute_iou(pred, mask, num_classes, ignore_index))
    class_iou = np.nanmean(np.array(iou_list, dtype=float), axis=0)
    miou      = float(np.nanmean(class_iou))
    return miou, class_iou

# TRAINING

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    class_weights_tensor = compute_class_weights(
        TRAIN_MASK, NUM_CLASSES, ignore_index=IGNORE_INDEX)
    class_weights_np = class_weights_tensor.numpy()

    manual_weights    = class_weights_tensor.clone()
    manual_weights[BARELAND_CLS] *= BARELAND_WEIGHT_MULT
    manual_weights    = manual_weights.to(device)
    print(f"Loss weights (bareland ×{BARELAND_WEIGHT_MULT}): "
          f"{np.round(manual_weights.cpu().numpy(), 4)}")

    train_dataset = SegDataset(
        TRAIN_IMG,
        TRAIN_MASK,
        IMAGE_SIZE,
        train=True,
        copy_paste=None
    )
    val_dataset   = SegDataset(VAL_IMG,   VAL_MASK,   IMAGE_SIZE,
                               train=False, copy_paste=None)


    sampler = make_weighted_sampler(TRAIN_MASK, train_dataset.files, class_weights_np)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True,
                              drop_last=True)     
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=2, pin_memory=True)

    model = SAM_LoRA(SAM_CHECKPOINT, NUM_CLASSES).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    loss_fn = TverskyLoss(
        classes=NUM_CLASSES,
        alpha=0.7,
        beta=0.3,
        ignore_index=IGNORE_INDEX
    )

    scaler        = GradScaler()
    best_miou     = 0.0
    early_counter = 0
    history       = []

    for epoch in range(EPOCHS):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        epoch_start      = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # TRAIN
        model.train()
        train_loss = 0.0
        for img, mask in tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} train"):
            img  = img.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(img)
                loss = loss_fn(pred, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        iou_list = []
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Ep {epoch+1}/{EPOCHS} val"):
                img  = img.to(device)
                mask = mask.to(device)
                pred = model(img)
                loss = loss_fn(pred, mask)
                val_loss += loss.item()
                iou_list.append(compute_iou(pred, mask, NUM_CLASSES, IGNORE_INDEX))
        val_loss  /= len(val_loader)
        class_iou  = np.nanmean(np.array(iou_list, dtype=float), axis=0)
        miou       = float(np.nanmean(class_iou))

        scheduler.step()


        epoch_time = time.time() - epoch_start
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        gap     = val_loss - train_loss
        print(f"\nEpoch {epoch+1}/{EPOCHS}  lr={current_lr:.2e}")
        print(f"  Train Loss   : {train_loss:.4f}")
        print(f"  Val Loss     : {val_loss:.4f}  (gap={gap:.4f})")
        print(f"  mIoU (8 cls) : {miou:.4f}")
        print(f"  Bareland  (cls 1) : {class_iou[1]:.4f}")
        print(f"  Rangeland (cls 2) : {class_iou[2]:.4f}")
        print(f"  All classes  : {np.round(class_iou[1:], 4)}")
        print(f"  Time         : {epoch_time:.1f}s")
        print(f"  Peak GPU Mem : {peak_memory:.2f} GB")

        row = {
            "epoch": epoch+1,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gap": gap,
            "miou": miou,
            "epoch_time_sec": epoch_time,
            "peak_gpu_memory_gb": peak_memory
        }
        for c in range(NUM_CLASSES):
            row[f"iou_{CLASS_NAMES[c]}"] = (
                float(class_iou[c]) if not np.isnan(class_iou[c]) else None)
        history.append(row)
        pd.DataFrame(history).to_excel(
            os.path.join(OUTPUT_DIR, "training_log.xlsx"), index=False)

        if miou > best_miou:
            best_miou     = miou
            early_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  ✓ Best model saved  (mIoU={best_miou:.4f})")
        else:
            early_counter += 1

        if early_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

    # FINAL EVALUATION WITH TTA  

    print("\n" + "="*60)
    print("Running FINAL evaluation with TTA on best checkpoint...")
    print("="*60)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    tta_miou, tta_class_iou = evaluate_with_tta(
        model, val_loader, device, NUM_CLASSES, IGNORE_INDEX)

    print(f"\n{'='*60}")
    print(f"  PAPER NUMBERS  (best model + TTA)")
    print(f"{'='*60}")
    print(f"  mIoU (8 classes) : {tta_miou:.4f}")
    for c in range(1, NUM_CLASSES):
        v = tta_class_iou[c]
        marker = " ← bareland" if c == 1 else ""
        print(f"  {CLASS_NAMES[c]:15s} (cls {c}): "
              f"{v:.4f}{marker}" if not np.isnan(v) else f"  {CLASS_NAMES[c]:15s}: NaN")
    print(f"{'='*60}")

    # Save TTA results to Excel
    df_tta = pd.DataFrame([{
        "metric": "TTA_mIoU", "value": tta_miou,
        **{f"iou_{CLASS_NAMES[c]}": (
            float(tta_class_iou[c]) if not np.isnan(tta_class_iou[c]) else None)
           for c in range(NUM_CLASSES)}
    }])
    df_tta.to_excel(os.path.join(OUTPUT_DIR, "final_tta_results.xlsx"), index=False)
    print("TTA results saved to final_tta_results.xlsx")

    # PLOTS
    df = pd.DataFrame(history)

    fig, ax = plt.subplots()
    ax.plot(df.epoch, df.train_loss, label="train")
    ax.plot(df.epoch, df.val_loss,   label="val")
    ax.axvspan(0.5, WARMUP_EPOCHS+0.5, alpha=0.08, color="gray", label="warmup")
    ax.legend(); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curve (v16 final)")     
    fig.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(df.epoch, df.miou)
    ax.axhline(tta_miou, color="red", linestyle="--",
               label=f"TTA best: {tta_miou:.4f}")
    ax.axvspan(0.5, WARMUP_EPOCHS+0.5, alpha=0.08, color="gray", label="warmup")
    ax.legend(); ax.set_xlabel("Epoch"); ax.set_ylabel("mIoU")
    ax.set_title("mIoU Curve (red dashed = TTA final)")
    fig.savefig(os.path.join(OUTPUT_DIR, "miou_curve.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for c in range(NUM_CLASSES):
        if c == IGNORE_INDEX: continue
        col = f"iou_{CLASS_NAMES[c]}"
        if col not in df.columns: continue
        lw    = 2.5 if c in (1, 2) else 1.0
        style = "-" if c == 1 else ("--" if c == 2 else "-")
        ax.plot(df.epoch, df[col], label=CLASS_NAMES[c], lw=lw, ls=style)
    ax.axvspan(0.5, WARMUP_EPOCHS+0.5, alpha=0.08, color="gray", label="warmup")
    ax.legend(fontsize=8); ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
    ax.set_title("Per-Class IoU (cls0 excluded)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "perclass_iou_curve.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(df.epoch, df.lr)
    ax.axvspan(0.5, WARMUP_EPOCHS+0.5, alpha=0.08, color="gray", label="warmup")
    ax.legend(); ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_title("LR Schedule")
    fig.savefig(os.path.join(OUTPUT_DIR, "lr_curve.png"), dpi=150); plt.close(fig)

    print(f"\nTraining finished.  Best mIoU (no TTA): {best_miou:.4f}")
    print(f"                   Best mIoU (TTA):     {tta_miou:.4f}")


if __name__ == "__main__":
    train()
