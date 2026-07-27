import os
import cv2
import numpy as np
import torch
from collections import defaultdict

# CONFIG
IMAGE_SIZE = 512
NUM_CLASSES = 9 

CLASS_NAMES = [
    "Background",
    "Bareland",
    "Rangeland",
    "Developed",
    "Road",
    "Tree",
    "Water",
    "Agriculture",
    "Building"
]

VALID_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8]

# LOAD MODEL
from models.sam_lora.sam_lora_train import SAM_LoRA, SAM_CHECKPOINT

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SAM_LoRA(
    checkpoint=SAM_CHECKPOINT,
    num_classes=NUM_CLASSES
).to(device)

model.load_state_dict(
    torch.load(
        r"...\sam_results\best_model.pth",
        map_location=device
    )
)

model.eval()

# PATHS
BASE_IMG_DIR = r"...\OpenEarthMap\images\val"
BASE_MASK_DIR = r"...\OpenEarthMap\labels\val"

OUTPUT_FILE = r"...\city_analysis_results.txt"

# COLLECT FILES BY CITY
def get_city_files():

    city_files = defaultdict(list)

    for f in os.listdir(BASE_IMG_DIR):

        if not (f.endswith(".tif") or f.endswith(".tiff")):
            continue

        city_name = f.split("_")[0]

        img_path = os.path.join(BASE_IMG_DIR, f)
        mask_path = os.path.join(BASE_MASK_DIR, f)

        if os.path.exists(mask_path):
            city_files[city_name].append(
                (img_path, mask_path)
            )

    return city_files


# ANALYZE ONE CITY
def analyze_city(file_list):

    gt_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
    pred_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)

    total_pixels = 0

    with torch.no_grad():

        for img_path, mask_path in file_list:

            # Read image and mask
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, 0)

            img = cv2.resize(
                img,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            mask = cv2.resize(
                mask,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_NEAREST
            )

            total_pixels += mask.size

            # Model inference
            img_tensor = (
                torch.tensor(img)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            img_tensor = img_tensor.unsqueeze(0).to(device)

            pred = model(img_tensor)

            pred = torch.argmax(
                pred,
                dim=1
            ).cpu().numpy()[0]

            # Count pixels per class
            for c in VALID_CLASSES:

                gt_pixels[c] += np.sum(mask == c)
                pred_pixels[c] += np.sum(pred == c)

    # Semantic totals
    gt_total_semantic = gt_pixels[VALID_CLASSES].sum()
    pred_total_semantic = pred_pixels[VALID_CLASSES].sum()

    gt_percent = (
        gt_pixels /
        (gt_total_semantic + 1e-8)
        * 100
    )

    pred_percent = (
        pred_pixels /
        (pred_total_semantic + 1e-8)
        * 100
    )

    return (
        gt_pixels,
        pred_pixels,
        gt_percent,
        pred_percent,
        total_pixels,
        gt_total_semantic,
        pred_total_semantic
    )


# RUN ANALYSIS
city_files = get_city_files()

print("=" * 60)
print("TOTAL FILES :", sum(len(v) for v in city_files.values()))
print("TOTAL CITIES:", len(city_files))
print("=" * 60)

results = {}

for city, files in city_files.items():

    (
        gt_pixels,
        pred_pixels,
        gt_percent,
        pred_percent,
        total_pixels,
        gt_total_semantic,
        pred_total_semantic
    ) = analyze_city(files)

    results[city] = (
        gt_pixels,
        pred_pixels,
        gt_percent,
        pred_percent,
        total_pixels,
        gt_total_semantic,
        pred_total_semantic
    )


# SAVE RESULTS
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    title = (
        "============================================================\n"
        "CITY-LEVEL LAND COVER ANALYSIS\n"
        "(8 SEMANTIC CLASSES - BACKGROUND EXCLUDED)\n"
        "============================================================\n\n"
    )

    print(title)
    f.write(title)

    for city in sorted(results.keys()):

        (
            gt_pixels,
            pred_pixels,
            gt_percent,
            pred_percent,
            total_pixels,
            gt_total_semantic,
            pred_total_semantic
        ) = results[city]

        header = (
            "\n"
            "============================================================\n"
            f"CITY: {city}\n"
            "============================================================\n"
            f"TOTAL PIXELS          : {total_pixels:,}\n"
            f"GT SEMANTIC PIXELS    : {gt_total_semantic:,}\n"
            f"PRED SEMANTIC PIXELS  : {pred_total_semantic:,}\n"
            "============================================================\n"
        )

        print(header)
        f.write(header)

        table_header = (
            f"{'CLASS':12s} | "
            f"{'GT_PIXELS':>12s} | "
            f"{'PRED_PIXELS':>12s} | "
            f"{'GT_%':>8s} | "
            f"{'PRED_%':>8s} | "
            f"{'ERROR_%':>8s}"
        )

        print(table_header)
        print("-" * len(table_header))

        f.write(table_header + "\n")
        f.write("-" * len(table_header) + "\n")

        for i in VALID_CLASSES:

            if pred_pixels[i] > 0:
                error = (
                    abs(gt_pixels[i] - pred_pixels[i])
                    / gt_pixels[i]
                ) * 100
            else:
                error = 0.0

            line = (
                f"{CLASS_NAMES[i]:12s} | "
                f"{gt_pixels[i]:12,d} | "
                f"{pred_pixels[i]:12,d} | "
                f"{gt_percent[i]:8.2f} | "
                f"{pred_percent[i]:8.2f} | "
                f"{error:8.2f}"
            )

            print(line)
            f.write(line + "\n")

        f.write("\n")

print("\nResults saved to:")
print(OUTPUT_FILE)
