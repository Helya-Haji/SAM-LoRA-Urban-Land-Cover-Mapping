import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet50


IMAGE_SIZE = 512
NUM_CLASSES = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"...\deeplabv3\deeplabv3_best.pth"
IMAGE_PATH = r"...\images\tyrolw_69.tif"
SAVE_PATH = r"...\images\tyrolw_69_deeplab.png"

USE_TTA = True

# COLORS 
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

# MODEL
class DeepLabV3Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = deeplabv3_resnet50(weights=None)

        self.model.classifier[-1] = nn.Conv2d(
            256,
            NUM_CLASSES,
            kernel_size=1
        )

    def forward(self,x):

        return self.model(x)["out"]

# PREPROCESS
transform = A.Compose([
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# TTA
def tta_predict(model,img):

    preds=[]

    with torch.no_grad():

        preds.append(model(img))

        preds.append(model(torch.flip(img,[3])))
        preds.append(model(torch.flip(img,[2])))
        preds.append(model(torch.flip(img,[2,3])))

    preds[1]=torch.flip(preds[1],[3])
    preds[2]=torch.flip(preds[2],[2])
    preds[3]=torch.flip(preds[3],[2,3])

    return torch.mean(torch.stack(preds),0)


# LOAD MODEL
model = DeepLabV3Model().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH,map_location=DEVICE),
    strict=False
)

model.eval()


# LOAD IMAGE
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
orig_h, orig_w = img.shape[:2]
original = img.copy()
aug = transform(image=img)
tensor = aug["image"].unsqueeze(0).to(DEVICE)

# PREDICTION

# with torch.no_grad():

#     if USE_TTA:
#         output = tta_predict(model,tensor)
#     else:
#         output = model(tensor)
# pred = torch.argmax(output,1).squeeze().cpu().numpy()

with torch.no_grad():

    if USE_TTA:
        output = tta_predict(model, tensor)
    else:
        output = model(tensor)
    output[:, 0, :, :] = float("-inf")

    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()


pred = cv2.resize(
    pred.astype(np.uint8),
    (orig_w, orig_h),
    interpolation=cv2.INTER_NEAREST
)


# COLORIZE

pred_color = COLORS[pred]
pred_color = COLORS[pred]

cv2.imwrite(
    SAVE_PATH,
    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
)

print(f"Saved: {SAVE_PATH}")