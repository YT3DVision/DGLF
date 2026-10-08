# DGCL: Decoupled Global-Local Consensus Learning

[简体中文说明](README.zh-CN.md)

This repository provides the paper's **two-stage reproduction pipeline**. Stage 1 fine-tunes a vision backbone with LoRA and extracts frozen PSM features. Stage 2 trains a classifier with Dual-Stream FBP, GSC, and NASC on those features. Dataset images and pretrained weights are not included.

The code covers DINOv2 experiments on Web-Bird, Web-Car, and Web-Aircraft, along with a DINOv3 option and synthetic-noise CIFAR-100N / CIFAR-80N experiments.

## Repository structure

```text
dgcl_release/
├── README.md                 # English README
├── README.zh-CN.md           # Chinese README
├── requirements.txt
├── LICENSE                   # MIT license for this repository
├── third_party/              # Original AutoAugment license
└── dgcl/
    ├── stage1.py             # WebFG Stage 1
    ├── stage1_cifar.py       # CIFAR Stage 1
    ├── stage2.py             # Shared Stage 2
    ├── evaluate.py           # Stage 2 classification accuracy
    └── data/
        ├── autoaugment.py    # ImageNet AutoAugment policy
        └── cifar.py          # CIFAR loader and synthetic label noise
```

Run all commands from the **repository root**. In the examples, root-level `data/` holds local datasets, and training outputs go to `outputs/` by default.

## Environment

Python 3.10–3.12 is recommended; full training requires a CUDA GPU. The paper reports using an RTX 4090. Install PyTorch and a matching torchvision build for your CUDA setup using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements.txt
```

The default DINOv2 model is [`facebook/dinov2-large`](https://huggingface.co/facebook/dinov2-large). The default DINOv3 model is [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).

## WebFG: two-stage reproduction

Get Web-Bird, Web-Car, or Web-Aircraft from the [dataset authors](https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset). Store images by class; `train/` and `val/` must have the same class directory names:

```text
data/web-bird/
├── train/<class_name>/*.jpg
└── val/<class_name>/*.jpg
```

For Web-Bird with DINOv2, run:

```bash
python -m dgcl.stage1 --data-root data/web-bird --output-dir outputs/web-bird/dinov2/stage1
python -m dgcl.stage2 --features-dir outputs/web-bird/dinov2/stage1 --output-dir outputs/web-bird/dinov2/stage2
python -m dgcl.evaluate --checkpoint outputs/web-bird/dinov2/stage2/stage2_best.pt --features outputs/web-bird/dinov2/stage1/val_features.pt
```

Stage 1 writes `train_features.pt` and `val_features.pt` for Stage 2. For another WebFG dataset, change the paths. For DINOv3, add `--backbone dinov3` to Stage 1 and use a separate output directory.

## CIFAR with synthetic noise: two-stage reproduction

Use the official CIFAR-100 Python files with this layout:

```text
data/cifar100/cifar-100-python/
├── train
└── test
```

`cifar100n` is the 100-class closed-set setting; `cifar80n` treats true classes 80–99 as OOD. The code generates the noisy labels.

Example: DINOv2 on CIFAR-80N with 80% symmetric noise:

```bash
python -m dgcl.stage1_cifar --data-root data/cifar100 --dataset cifar80n --noise-type sym --noise-rate 0.8 --backbone dinov2 --output-dir outputs/cifar80n/sym08/dinov2/stage1
python -m dgcl.stage2 --features-dir outputs/cifar80n/sym08/dinov2/stage1 --output-dir outputs/cifar80n/sym08/dinov2/stage2 --epochs 20 --warmup-epochs 5 --knn-clean-rate 0.2 --sct-min-threshold 0.2
python -m dgcl.evaluate --checkpoint outputs/cifar80n/sym08/dinov2/stage2/stage2_best.pt --features outputs/cifar80n/sym08/dinov2/stage1/val_features.pt
```

For DINOv3 with 40% asymmetric noise, use `--noise-type asym --noise-rate 0.4 --backbone dinov3` in Stage 1. Local weights can be specified with `--model-id`.

The evaluation commands above report Top-1 classification accuracy. This minimal release does not include dedicated scripts for the paper's noise-filtering and OOD analysis tables.
