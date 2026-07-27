
# SAM-LoRA-Urban-LandCover

Official implementation of the paper:

**Parameter-Efficient Semantic Segmentation for Urban Land-Cover Mapping Using Segment Anything Model with LoRA Fine-Tuning**

<p align="center">
  <img src="E:\work\baradaran\Picture3.png" width="900">
</p>

## Overview

This repository provides the official implementation of a parameter-efficient semantic segmentation framework for urban land-cover mapping based on the Segment Anything Model (SAM) and Low-Rank Adaptation (LoRA).

The proposed approach combines the powerful visual representations of the pretrained SAM ViT-B encoder with lightweight LoRA fine-tuning and an ASPP-based decoder, enabling efficient adaptation for high-resolution urban semantic segmentation while training only a small fraction of the original model parameters.

The framework is evaluated on the **OpenEarthMap** benchmark and compared against several commonly used semantic segmentation baselines.

---

## Features

- SAM ViT-B encoder
- LoRA-based parameter-efficient fine-tuning
- ASPP decoder
- Multi-class semantic segmentation
- OpenEarthMap benchmark
- Baseline implementations:
  - U-Net
  - U-Net (ResNet34)
  - DeepLabV3
- Urban land-cover analysis scripts
- Visualization utilities

---

## Repository Structure

```
SAM-LoRA-Urban-Land-Cover-Mapping/
│
├── analysis/
│   ├── urban_land_cover_analysis.py
│   ├── visualization.py
│   └── class_proportion_test.py
│
├── models/
│   ├── sam_lora/
│   │   ├── sam_lora_train.py
│   │   ├── sam_lora_inference.py
│   │
│   ├── deeplabv3/
│   ├── unet/
│   └── unet_resnet34/
│
├── data/
├── checkpoints/
├── outputs/
│
├── requirements.txt
└── README.md

```

---

## Installation

Clone the repository

```bash
git clone https://github.com/USERNAME/SAM-LoRA-Urban-LandCover.git

cd SAM-LoRA-Urban-Land-Cover-Mapping
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the publicly available **OpenEarthMap** dataset.

Please download the dataset from: https://www.kaggle.com/datasets/aletbm/global-land-cover-mapping-openearthmap

---

## Pretrained SAM Checkpoint

The pretrained SAM weights are **not included** due to GitHub file size limitations.

Download the official **SAM ViT-B** checkpoint from: https://github.com/facebookresearch/segment-anything

Then update the checkpoint path in the code accordingly.

---

## Training

Run

```bash
python models/sam_lora/sam_lora_train.py
```

---

## Inference

Run

```bash
python models/sam_lora/sam_lora_inference.py
```

---

## Baseline Models

This repository also contains implementations of several baseline methods used for comparison:

- U-Net
- U-Net with ResNet34 encoder
- DeepLabV3

Each baseline can be trained independently.

---

## Urban Land-Cover Analysis

The repository includes scripts for

- Urban indicator extraction
- Land-cover statistics
- Visualization
- Cross-city comparison

located in

```
analysis/
```

---

