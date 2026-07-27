import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


from models.sam_lora.sam_lora_train import (
    IMAGE_SIZE,
    NUM_CLASSES,
    SAM_LoRA,
    SAM_CHECKPOINT,
)

# PATHS
MODEL_PATH = r"...\sam_results\best_model.pth"

CLASS_NAMES = [
    "Background",
    "Bareland",
    "Rangeland",
    "Developed",
    "Road",
    "Tree",
    "Water",
    "Agriculture",
    "Building",
]

COLORS = np.array([
    [0,   0,   0  ],   # 0 Background  – black
    [128, 0,   0  ],   # 1 Bareland    – #800000
    [0,   255, 36 ],   # 2 Rangeland   – #00FF24
    [148, 148, 148],   # 3 Developed   – #949494
    [255, 255, 255],   # 4 Road        – #FFFFFF
    [34,  97,  38 ],   # 5 Tree        – #226126
    [0,   69,  255],   # 6 Water       – #0045FF
    [75,  181, 73 ],   # 7 Agriculture – #4BB549
    [222, 31,  7  ],   # 8 Building    – #DE1F07
], dtype=np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Convert a class-index mask to an RGB color image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):    
        color[mask == c] = COLORS[c]
    return color

def visualize_sample(model, img_path: str, mask_path: str, device: str = "cpu"):
    """
    Run inference on one image and save three files next to the original:
        <stem>_image.png   – resized input
        <stem>_gt.png      – ground-truth mask (colorized)
        <stem>_pred.png    – model prediction  (colorized)

    Also shows a combined figure with plt.show().
    """
    model.eval()

    # Load
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Pre-process
    tf = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    aug = tf(image=img)
    x   = aug["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred = model(x)

        # Remove background class from prediction
        pred[:, 0, :, :] = float("-inf")

        pred = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)

    # Resize GT & image to match model output size 
    gt  = cv2.resize(gt,  (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Colorize
    gt_color   = colorize(gt)
    pred_color = colorize(pred)

    # Derive output paths
    base_dir  = os.path.dirname(img_path)
    stem      = os.path.splitext(os.path.basename(img_path))[0]

    out_img  = os.path.join(base_dir, f"{stem}_image.png")
    out_gt   = os.path.join(base_dir, f"{stem}_gt.png")
    out_pred = os.path.join(base_dir, f"{stem}_pred.png")

    # Save individual files (RGB → BGR for cv2)
    cv2.imwrite(out_img,  cv2.cvtColor(img,        cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_gt,   cv2.cvtColor(gt_color,   cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_pred, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

    print(f"Saved: {out_img}")
    print(f"Saved: {out_gt}")
    print(f"Saved: {out_pred}")

    # Combined plot 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title in zip(
        axes,
        [img, gt_color, pred_color],
        ["Input Image", "Ground Truth", "SAM-LoRA Prediction"],
    ):
        ax.imshow(data)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# MAIN
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = SAM_LoRA(checkpoint=SAM_CHECKPOINT, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    IMG_PATH  = r"...\images\austin_46.tif"
    MASK_PATH = r"...\images\masks\austin_46.tif"

    visualize_sample(model, IMG_PATH, MASK_PATH, device)
