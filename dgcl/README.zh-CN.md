# DGCL：解耦的全局与局部共识学习

[English README](README.md)

本仓库整理了论文的**两阶段复现流程**：第一阶段对视觉骨干进行 LoRA 微调，并通过 PSM 提取冻结特征；第二阶段使用这些特征训练包含 Dual-Stream FBP、GSC 和 NASC 的分类器。仓库不含数据集图片或预训练权重。

代码覆盖 Web-Bird、Web-Car、Web-Aircraft 的 DINOv2 实验，另包含DINOv3 选项和合成噪声 CIFAR-100N / CIFAR-80N 实验。

## 目录结构

```text
dgcl_release/
├── README.md                 # 英文说明
├── README.zh-CN.md           # 中文说明
├── requirements.txt
├── LICENSE                   # 本仓库 MIT 许可证
├── third_party/              # AutoAugment 原作者许可证
└── dgcl/
    ├── stage1.py             # WebFG 第一阶段
    ├── stage1_cifar.py       # CIFAR 第一阶段
    ├── stage2.py             # 共用第二阶段
    ├── evaluate.py           # 第二阶段分类准确率评估
    └── data/
        ├── autoaugment.py    # ImageNet AutoAugment 策略
        └── cifar.py          # CIFAR 读取及合成标签噪声
```

以下命令均在**仓库根目录**运行。示例中的根目录 `data/` 用于放本地数据集，训练输出默认放在 `outputs/`。

## 环境准备

推荐 Python 3.10–3.12；完整训练需要 CUDA GPU。论文报告使用 RTX 4090。先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装适合机器 CUDA 版本的 PyTorch 与匹配的 torchvision，再安装依赖：

```bash
python -m venv .venv
# 按当前终端的方式激活 .venv
python -m pip install -r requirements.txt
```

默认 DINOv2 模型为 [`facebook/dinov2-large`](https://huggingface.co/facebook/dinov2-large)。默认 DINOv3 模型为 [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)。

## WebFG：两阶段复现

从[数据集作者仓库](https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset)获取 Web-Bird、Web-Car 或 Web-Aircraft。数据按类别存放，`train/` 与 `val/` 的类别目录名须一致：

```text
data/web-bird/
├── train/<类别名>/*.jpg
└── val/<类别名>/*.jpg
```

以 Web-Bird + DINOv2 为例，在仓库根目录运行：

```bash
python -m dgcl.stage1 --data-root data/web-bird --output-dir outputs/web-bird/dinov2/stage1
python -m dgcl.stage2 --features-dir outputs/web-bird/dinov2/stage1 --output-dir outputs/web-bird/dinov2/stage2
python -m dgcl.evaluate --checkpoint outputs/web-bird/dinov2/stage2/stage2_best.pt --features outputs/web-bird/dinov2/stage1/val_features.pt
```

第一阶段会生成 `train_features.pt` 和 `val_features.pt`，供第二阶段读取。其他 WebFG 数据集替换路径即可；DINOv3 在第一阶段加 `--backbone dinov3`，并另设输出目录。

## CIFAR 合成噪声：两阶段复现

使用 CIFAR-100 官方 Python 版，目录如下：

```text
data/cifar100/cifar-100-python/
├── train
└── test
```

`cifar100n` 为 100 类闭集；`cifar80n` 将真实类别 80–99 作为 OOD。噪声标签由代码合成。

DINOv2 + CIFAR-80N、对称 80% 噪声示例：

```bash
python -m dgcl.stage1_cifar --data-root data/cifar100 --dataset cifar80n --noise-type sym --noise-rate 0.8 --backbone dinov2 --output-dir outputs/cifar80n/sym08/dinov2/stage1
python -m dgcl.stage2 --features-dir outputs/cifar80n/sym08/dinov2/stage1 --output-dir outputs/cifar80n/sym08/dinov2/stage2 --epochs 20 --warmup-epochs 5 --knn-clean-rate 0.2 --sct-min-threshold 0.2
python -m dgcl.evaluate --checkpoint outputs/cifar80n/sym08/dinov2/stage2/stage2_best.pt --features outputs/cifar80n/sym08/dinov2/stage1/val_features.pt
```

DINOv3 + 非对称 40% 噪声：第一阶段改用 `--noise-type asym --noise-rate 0.4 --backbone dinov3`（本地权重可用 `--model-id` 指定）。

以上评估命令报告 Top-1 分类准确率；本精简版未包含论文中噪声过滤和 OOD 分析表的专用评估脚本。


