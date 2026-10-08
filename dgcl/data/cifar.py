"""Deterministic synthetic noisy CIFAR-100 / CIFAR-80 data from the official Python files."""

import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def cifar_file(data_root, split):
    root = Path(data_root)
    candidate = root / "cifar-100-python" / split
    if not candidate.is_file():
        candidate = root / split
    if not candidate.is_file():
        raise FileNotFoundError(f"Expected CIFAR-100 Python file: {candidate}")
    return candidate


def read_cifar(data_root, split):
    # The official CIFAR-100 Python distribution is a pickle. Use trusted files only.
    with cifar_file(data_root, split).open("rb") as handle:
        record = pickle.load(handle, encoding="latin1")
    images = np.asarray(record["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    fine = np.asarray(record["fine_labels"], dtype=np.int64)
    coarse = np.asarray(record["coarse_labels"], dtype=np.int64)
    if len(images) != len(fine) or len(fine) != len(coarse):
        raise ValueError("Inconsistent CIFAR-100 data and label lengths")
    return images, fine, coarse


def corrupt_labels(fine, coarse, dataset, noise_type, noise_rate, seed):
    """Match the reference scripts' uniform and within-coarse-class transitions."""
    classes = 80 if dataset == "cifar80n" else 100
    if noise_type not in {"sym", "asym"} or not 0 <= noise_rate <= 1:
        raise ValueError("noise-type must be sym/asym and noise-rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    noisy = fine.copy()
    mask_gt = np.ones(len(fine), dtype=np.int64)
    successors = {}
    if noise_type == "asym":
        for coarse_class in np.unique(coarse):
            members = sorted(set(fine[(coarse == coarse_class) & (fine < classes)].tolist()))
            for position, label in enumerate(members):
                successors[label] = members[(position + 1) % len(members)]
    for index, true_label in enumerate(fine):
        if true_label >= classes:
            # CIFAR-80N: classes 80..99 remain in the training images as OOD.
            noisy[index] = rng.integers(classes)
            mask_gt[index] = 0
        elif noise_type == "sym":
            if rng.random() < noise_rate:
                noisy[index] = rng.integers(classes)
            mask_gt[index] = int(noisy[index] == true_label)
        else:
            if rng.random() < noise_rate:
                noisy[index] = successors[int(true_label)]
            mask_gt[index] = int(noisy[index] == true_label)
    return noisy, mask_gt


class NoisyCIFAR100(Dataset):
    def __init__(self, data_root, split, dataset="cifar100n", noise_type="sym",
                 noise_rate=0.2, seed=42, transform=None):
        if dataset not in {"cifar100n", "cifar80n"} or split not in {"train", "val"}:
            raise ValueError("dataset must be cifar100n/cifar80n and split train/val")
        self.transform = transform
        self.class_count = 80 if dataset == "cifar80n" else 100
        self.class_to_idx = {str(index): index for index in range(self.class_count)}
        images, fine, coarse = read_cifar(data_root, "train" if split == "train" else "test")
        if split == "train":
            noisy, mask_gt = corrupt_labels(fine, coarse, dataset, noise_type, noise_rate, seed)
        else:
            selected = fine < self.class_count
            images, fine = images[selected], fine[selected]
            noisy, mask_gt = fine.copy(), np.ones(len(fine), dtype=np.int64)
        self.images = images
        self.labels = noisy
        self.clean_labels = fine
        self.mask_gt = mask_gt

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        return (image, int(self.labels[index]), int(self.clean_labels[index]),
                int(self.mask_gt[index]))
