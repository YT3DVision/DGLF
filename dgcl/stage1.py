"""Stage 1: LoRA fine-tuning followed by frozen DINOv2/DINOv3 PSM extraction."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm
from transformers import AutoModel

from dgcl.data.autoaugment import AutoAugImageNetPolicy


MODEL_IDS = {
    "dinov2": "facebook/dinov2-large",
    "dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SquarePad:
    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        return TF.pad(image, (left, top, side - width - left, side - height - top))


def transforms(image_size, training, augmentation="autoaugment"):
    operations = [SquarePad(), T.Resize((image_size, image_size), InterpolationMode.BICUBIC)]
    if training and augmentation == "bird_jitter":
        operations += [T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                       T.RandomHorizontalFlip(), T.ToTensor(),
                       T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
                       T.Normalize(MEAN, STD)]
        return T.Compose(operations)
    if training:
        operations += [T.RandomHorizontalFlip(), AutoAugImageNetPolicy()]
    operations += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(operations)


def load_base(model_id, trust_remote_code=False, backbone_kind="dinov2",
              eager_attention=False):
    kwargs = {"trust_remote_code": trust_remote_code}
    if backbone_kind == "dinov3" and eager_attention:
        # HF DINOv3 v4.57.1 does not put attentions in the model output.
        # Its last attention module returns weights when using the eager backend.
        kwargs["attn_implementation"] = "eager"
    return AutoModel.from_pretrained(model_id, **kwargs)


class Stage1Model(nn.Module):
    def __init__(self, num_classes, rank, backbone_kind="dinov2", model_id=None,
                 trust_remote_code=False):
        super().__init__()
        model_id = model_id or MODEL_IDS[backbone_kind]
        backbone = load_base(model_id, trust_remote_code, backbone_kind)
        targets = (["query", "key", "value", "attention.output.dense"]
                   if backbone_kind == "dinov2" else "all-linear")
        config = LoraConfig(
            r=rank, lora_alpha=16, lora_dropout=0.1,
            target_modules=targets,
        )
        self.backbone = get_peft_model(backbone, config)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, images):
        return self.head(self.backbone(images).last_hidden_state[:, 0])


class LastAttention:
    """Compute only CLS attention, avoiding the full quadratic attention matrix."""

    def __init__(self, backbone):
        try:
            self.module = backbone.base_model.model.encoder.layer[-1].attention.attention
        except AttributeError as exc:
            raise RuntimeError("Unsupported DINOv2 attention layout; check transformers version") from exc
        self.input = None
        self.handle = self.module.register_forward_hook(self._capture)

    def _capture(self, _module, inputs, _output):
        self.input = inputs[0]

    def patch_scores(self, patch_count):
        if self.input is None:
            raise RuntimeError("Last-layer attention hook did not receive input")
        x, module = self.input, self.module
        heads, head_dim = module.num_attention_heads, module.attention_head_size
        query = module.query(x[:, :1]).reshape(x.shape[0], 1, heads, head_dim).transpose(1, 2)
        key = module.key(x).reshape(x.shape[0], x.shape[1], heads, head_dim).transpose(1, 2)
        scores = (query @ key.transpose(-1, -2)).squeeze(2) / math.sqrt(head_dim)
        start = scores.shape[-1] - patch_count
        if start < 1:
            raise RuntimeError("Unexpected CLS/patch token layout")
        return scores.softmax(dim=-1)[:, :, start:].mean(dim=1)

    def close(self):
        self.handle.remove()


class DINOv3LastAttention:
    """Capture native HF DINOv3's last attention output with eager attention."""

    def __init__(self, backbone):
        try:
            self.module = backbone.base_model.model.layer[-1].attention
        except AttributeError as exc:
            raise RuntimeError("Unsupported DINOv3 attention layout") from exc
        self.weights = None
        self.handle = self.module.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self.weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None

    def patch_scores(self, patch_count):
        if self.weights is None:
            raise RuntimeError("DINOv3 attention weights unavailable; use native HF model with eager attention")
        return self.weights[:, :, 0, -patch_count:].mean(dim=1)

    def close(self):
        self.handle.remove()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels, *_ in loader:
            predictions = model(images.to(device)).argmax(1)
            correct += (predictions == labels.to(device)).sum().item()
            total += labels.numel()
    return correct / total


def extract(backbone, dataset, path, batch_size, workers, top_k, device,
            backbone_kind="dinov2"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    backbone.eval()
    attention = (LastAttention(backbone) if backbone_kind == "dinov2"
                 else DINOv3LastAttention(backbone))
    features, labels = [], []
    clean_labels, noise_masks = [], []
    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extract {path.stem}"):
                images, batch_labels, *metadata = batch
                hidden = backbone(images.to(device)).last_hidden_state
                if backbone_kind == "dinov3":
                    registers = getattr(backbone.config, "num_register_tokens", 0)
                    cls, patches = hidden[:, 0], hidden[:, 1 + registers:]
                else:
                    cls, patches = hidden[:, 0], hidden[:, 1:]
                scores = attention.patch_scores(patches.shape[1])
                if not 0 < patches.shape[1] or scores.shape[1] != patches.shape[1]:
                    raise RuntimeError("Attention and patch token counts do not match")
                indices = scores.topk(min(top_k, patches.shape[1]), dim=1).indices
                chosen = patches.gather(1, indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
                features.append(torch.cat((cls, chosen.mean(1), chosen.max(1).values), dim=1).cpu())
                labels.append(batch_labels)
                if metadata:
                    clean_labels.append(metadata[0].cpu())
                    noise_masks.append(metadata[1].cpu())
    finally:
        attention.close()
    payload = {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "class_to_idx": dataset.class_to_idx,
    }
    if clean_labels:
        payload["clean_labels"] = torch.cat(clean_labels)
        payload["mask_gt"] = torch.cat(noise_masks)
    torch.save(payload, path)
    print(f"Saved {path}: {sum(len(batch) for batch in labels)} samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="ImageFolder train/ and val/ root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=MODEL_IDS, default="dinov2")
    parser.add_argument("--model-id", help="Local model directory or Hugging Face model ID")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Only for a trusted local custom DINOv3 model implementation")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--extract-batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--augmentation", choices=("autoaugment", "bird_jitter"),
                        default="autoaugment")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.extract_batch_size, args.rank, args.top_k) < 1:
        parser.error("epochs, batch sizes, rank, and top-k must be positive")
    if args.image_size != 518:
        parser.error("WebFG release uses the paper's 518-pixel input size")
    for split in ("train", "val"):
        if not (args.data_root / split).is_dir():
            parser.error(f"Missing {args.data_root / split}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = ImageFolder(args.data_root / "train",
                        transform=transforms(args.image_size, True, args.augmentation))
    val = ImageFolder(args.data_root / "val", transform=transforms(args.image_size, False))
    test = (ImageFolder(args.data_root / "test", transform=transforms(args.image_size, False))
            if (args.data_root / "test").is_dir() else None)
    if train.class_to_idx != val.class_to_idx:
        parser.error("train/ and val/ must contain the same class directory names")
    if test is not None and train.class_to_idx != test.class_to_idx:
        parser.error("test/ must contain the same class directory names as train/")
    if not train.samples or not val.samples:
        parser.error("train/ and val/ must contain images")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stage1_config.json").write_text(
        json.dumps({**vars(args), "data_root": str(args.data_root), "output_dir": str(args.output_dir),
                    "model_id": args.model_id or MODEL_IDS[args.backbone],
                    "class_to_idx": train.class_to_idx}, indent=2), encoding="utf-8")

    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val, args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
    model_id = args.model_id or MODEL_IDS[args.backbone]
    model = Stage1Model(len(train.classes), args.rank, args.backbone, model_id,
                        args.trust_remote_code).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_accuracy = -1.0
    checkpoint = args.output_dir / "stage1_best"
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels, *_ in tqdm(train_loader, desc=f"Stage 1 {epoch + 1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}: loss={running_loss / len(train_loader):.4f}, val_acc={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.backbone.save_pretrained(str(checkpoint))
            torch.save(model.head.state_dict(), checkpoint / "head.pt")
        scheduler.step()

    # Reload the selected adapter; the last epoch is not necessarily the best epoch.
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    base = load_base(model_id, args.trust_remote_code, args.backbone,
                     eager_attention=args.backbone == "dinov3")
    backbone = PeftModel.from_pretrained(base, str(checkpoint)).to(device).eval()
    clean_transform = transforms(args.image_size, False)
    train.transform = clean_transform
    extract(backbone, train, args.output_dir / "train_features.pt", args.extract_batch_size,
            args.workers, args.top_k, device, args.backbone)
    extract(backbone, val, args.output_dir / "val_features.pt", args.extract_batch_size,
            args.workers, args.top_k, device, args.backbone)
    if test is not None:
        extract(backbone, test, args.output_dir / "test_features.pt", args.extract_batch_size,
                args.workers, args.top_k, device, args.backbone)
    print(f"Stage 1 complete; best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
