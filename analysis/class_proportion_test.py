import os
from PIL import Image
import numpy as np
import pandas as pd

TRAIN_MASK = r"...\OpenEarthMap\labels\train"

CLASS_NAMES = {
    1: "Bareland",
    2: "Rangeland",
    3: "Developed",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Agriculture",
    8: "Building"
}

pixel_counts = {k: 0 for k in CLASS_NAMES}

for file in os.listdir(TRAIN_MASK):

    mask = np.array(
        Image.open(
            os.path.join(TRAIN_MASK, file)
        )
    )

    unique, counts = np.unique(mask, return_counts=True)

    for cls, cnt in zip(unique, counts):

        if cls in pixel_counts:
            pixel_counts[cls] += int(cnt)

total_pixels = sum(pixel_counts.values())

rows = []

for cls_id in CLASS_NAMES:

    pixels = pixel_counts[cls_id]

    rows.append({
        "Class": CLASS_NAMES[cls_id],
        "Pixels": pixels,
        "Percentage": round(
            pixels / total_pixels * 100,
            3
        )
    })

df = pd.DataFrame(rows)

df = df.sort_values(
    "Percentage",
    ascending=False
)

print(df)

df.to_csv(
    "OpenEarthMap_Train_ClassDistribution.csv",
    index=False
)
