"""Stage 1 for synthetic-noise CIFAR-100N / CIFAR-80N with DINOv2 or DINOv3."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from peft import PeftModel

from dgcl.data.cifar import NoisyCIFAR100, cifar_file
from dgcl.stage1 import MODEL_IDS, MEAN, STD, Stage1Model, evaluate, extract, load_base, set_seed


def make_transform(image_size, training):
    operations = [T.Resize((image_size, image_size), InterpolationMode.BICUBIC)]
    if training:
        operations.append(T.RandomHorizontalFlip())
    operations += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(operations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing cifar-100-python/train and test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cifar100n", "cifar80n"), required=True)
    parser.add_argument("--noise-type", choices=("sym", "asym"), required=True)
    parser.add_argument("--noise-rate", type=float, required=True)
    parser.add_argument("--backbone", choices=MODEL_IDS, default="dinov2")
    parser.add_argument("--model-id", help="Local model directory or Hugging Face model ID")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Only for a trusted local custom DINOv3 model implementation")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.extract_batch_size, args.rank, args.top_k) < 1:
        parser.error("epochs, batch sizes, rank, and top-k must be positive")
    if not 0 <= args.noise_rate <= 1 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error("noise-rate must be in [0,1] and warmup-epochs < epochs")
    try:
        cifar_file(args.data_root, "train")
        cifar_file(args.data_root, "test")
    except FileNotFoundError as exc:
        parser.error(str(exc))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = make_transform(args.image_size, True)
    clean_transform = make_transform(args.image_size, False)
    train = NoisyCIFAR100(args.data_root, "train", args.dataset, args.noise_type,
                          args.noise_rate, args.seed, train_transform)
    val = NoisyCIFAR100(args.data_root, "val", args.dataset, args.noise_type,
                        args.noise_rate, args.seed, clean_transform)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val, args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
    model_id = args.model_id or MODEL_IDS[args.backbone]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stage1_config.json").write_text(
        json.dumps({**vars(args), "data_root": str(args.data_root), "output_dir": str(args.output_dir),
                    "model_id": model_id, "class_to_idx": train.class_to_idx}, indent=2),
        encoding="utf-8")
    model = Stage1Model(train.class_count, args.rank, args.backbone, model_id,
                        args.trust_remote_code).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)

    def schedule(epoch):
        if epoch < args.warmup_epochs:
            return 0.1 + 0.9 * epoch / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    checkpoint = args.output_dir / "stage1_best"
    best_accuracy = -1.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _, _ in tqdm(train_loader, desc=f"CIFAR Stage 1 {epoch + 1}/{args.epochs}"):
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

    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    backbone = PeftModel.from_pretrained(load_base(model_id, args.trust_remote_code, args.backbone,
                                                  eager_attention=args.backbone == "dinov3"),
                                         str(checkpoint)).to(device).eval()
    train.transform = clean_transform
    extract(backbone, train, args.output_dir / "train_features.pt", args.extract_batch_size,
            args.workers, args.top_k, device, args.backbone)
    extract(backbone, val, args.output_dir / "val_features.pt", args.extract_batch_size,
            args.workers, args.top_k, device, args.backbone)
    print(f"CIFAR Stage 1 complete; best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
