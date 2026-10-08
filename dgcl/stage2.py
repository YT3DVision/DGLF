"""Stage 2: frozen-feature Dual-Stream FBP, GSC, and NASC training."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def knn_consistency(features, labels, neighbors, clean_rate, device, chunk_size):
    """Global cosine KNN, explicitly excluding each sample itself."""
    if neighbors >= len(features):
        raise ValueError("knn-k must be smaller than the number of training samples")
    normalized = F.normalize(features.to(device), dim=1)
    labels = labels.to(device)
    result = torch.empty(len(features), dtype=torch.bool)
    for start in tqdm(range(0, len(features), chunk_size), desc="GSC geometry"):
        end = min(start + chunk_size, len(features))
        similarity = normalized[start:end] @ normalized.T
        rows = torch.arange(end - start, device=device)
        similarity[rows, start + rows] = -float("inf")
        indices = similarity.topk(neighbors, dim=1).indices
        votes = (labels[indices] == labels[start:end, None]).sum(dim=1)
        result[start:end] = (votes.float() / neighbors >= clean_rate).cpu()
    return result


class SemanticThreshold:
    """Median EMA with class-aware modulation from predicted-class confidence."""

    def __init__(self, classes, momentum):
        self.class_confidence = torch.full((classes,), 1 / classes)
        self.global_threshold = torch.tensor(1 / classes)
        self.momentum = momentum
        self.true_confidences = []
        self.predicted_confidences = []
        self.predicted_labels = []

    @torch.no_grad()
    def select(self, probabilities, labels):
        labels = labels.long()
        rows = torch.arange(len(labels), device=labels.device)
        true_confidence = probabilities[rows, labels]
        predicted_confidence, predicted_label = probabilities.max(dim=1)
        self.true_confidences.append(true_confidence.detach().cpu())
        self.predicted_confidences.append(predicted_confidence.detach().cpu())
        self.predicted_labels.append(predicted_label.detach().cpu())
        ratios = self.class_confidence.to(labels.device)
        ratios = ratios / ratios.max().clamp_min(1e-8)
        return true_confidence >= self.global_threshold.to(labels.device) * ratios[labels]

    @torch.no_grad()
    def update(self, epoch, warmup, initial_momentum, min_threshold=0.0):
        if not self.true_confidences:
            return
        progress = min(1.0, (epoch - 1) / max(1, warmup - 1))
        alpha = initial_momentum + progress * (self.momentum - initial_momentum)
        self.global_threshold = (alpha * self.global_threshold
                                 + (1 - alpha) * torch.cat(self.true_confidences).median()).clamp(min_threshold, 0.95)
        confidence = torch.cat(self.predicted_confidences)
        predicted = torch.cat(self.predicted_labels)
        sums = torch.zeros_like(self.class_confidence).scatter_add_(0, predicted, confidence)
        counts = torch.zeros_like(self.class_confidence).scatter_add_(0, predicted, torch.ones_like(confidence))
        present = counts > 0
        self.class_confidence[present] = (self.momentum * self.class_confidence[present]
                                          + (1 - self.momentum) * sums[present] / counts[present])
        self.true_confidences.clear()
        self.predicted_confidences.clear()
        self.predicted_labels.clear()


class BilinearHead(nn.Module):
    def __init__(self, global_dim, local_dim, projection_dim, classes, dropout):
        super().__init__()
        self.global_projection = nn.Linear(global_dim, projection_dim, bias=False)
        self.local_projection = nn.Linear(local_dim, projection_dim, bias=False)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(projection_dim, classes))

    def forward(self, global_feature, local_feature):
        product = self.global_projection(global_feature) * self.local_projection(local_feature)
        pooled = F.normalize(torch.sign(product) * torch.sqrt(product.abs() + 1e-5), dim=1)
        return self.classifier(pooled)


class NASCQueue(nn.Module):
    def __init__(self, input_dim, projection_dim, size, momentum, temperature):
        super().__init__()
        self.query_encoder = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(),
                                           nn.Linear(input_dim, projection_dim))
        self.key_encoder = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(),
                                         nn.Linear(input_dim, projection_dim))
        self.key_encoder.load_state_dict(self.query_encoder.state_dict())
        for parameter in self.key_encoder.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("keys", torch.zeros(size, projection_dim))
        self.register_buffer("labels", torch.full((size,), -1, dtype=torch.long))
        self.register_buffer("noise", torch.zeros(size, dtype=torch.bool))
        self.register_buffer("pointer", torch.zeros((), dtype=torch.long))
        self.momentum = momentum
        self.temperature = temperature

    def loss(self, clean_features, clean_labels):
        queries = F.normalize(self.query_encoder(clean_features), dim=1)
        valid = self.labels >= 0
        if not valid.any():
            return queries.sum() * 0
        logits = queries @ self.keys.T / self.temperature
        logits[:, ~valid] = -float("inf")
        positives = (clean_labels[:, None] == self.labels[None, :]) & (~self.noise[None, :]) & valid[None, :]
        has_positive = positives.any(dim=1)
        if not has_positive.any():
            return queries.sum() * 0
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_count = positives.sum(dim=1).clamp_min(1)
        per_sample = -(log_prob.masked_fill(~positives, 0).sum(dim=1) / positive_count)
        return per_sample[has_positive].mean()

    @torch.no_grad()
    def update(self, features, labels, noise):
        for query, key in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            key.mul_(self.momentum).add_(query, alpha=1 - self.momentum)
        keys = F.normalize(self.key_encoder(features), dim=1)
        # Ring-buffer write works even when queue size is not divisible by batch size.
        indices = (torch.arange(len(features), device=features.device) + self.pointer) % len(self.labels)
        self.keys[indices] = keys
        self.labels[indices] = labels
        self.noise[indices] = noise
        self.pointer.copy_((self.pointer + len(features)) % len(self.labels))


class DGCLClassifier(nn.Module):
    def __init__(self, feature_dim, classes, projection_dim, dropout, queue_size,
                 contrastive_dim, momentum, temperature):
        super().__init__()
        if feature_dim % 3:
            raise ValueError("Expected concatenated CLS, average, and max features")
        self.token_dim = feature_dim // 3
        self.appearance = BilinearHead(self.token_dim, self.token_dim, projection_dim, classes, dropout)
        self.saliency = BilinearHead(self.token_dim, self.token_dim, projection_dim, classes, dropout)
        self.nasc = NASCQueue(self.token_dim, contrastive_dim, queue_size, momentum, temperature)

    def forward(self, features):
        cls, average, maximum = features.split(self.token_dim, dim=1)
        return self.appearance(cls, average) + self.saliency(cls, maximum)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for features, labels in loader:
        predictions = model(features.to(device)).argmax(dim=1)
        correct += (predictions == labels.to(device)).sum().item()
        total += len(labels)
    return correct / total


def read_features(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not {"features", "labels", "class_to_idx"} <= data.keys():
        raise ValueError(f"Invalid feature file: {path}")
    features, labels = data["features"].float(), data["labels"].long()
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels) or len(labels) == 0:
        raise ValueError(f"Invalid feature/label shapes: {path}")
    if not torch.isfinite(features).all():
        raise ValueError(f"Non-finite features: {path}")
    return features, labels, data["class_to_idx"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--knn-clean-rate", type=float, default=0.5)
    parser.add_argument("--knn-chunk-size", type=int, default=512)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--sct-min-threshold", type=float, default=0.0)
    parser.add_argument("--uncertain-weight", type=float, default=0.5)
    parser.add_argument("--nasc-weight", type=float, default=0.5)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--contrastive-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    positive = (args.epochs, args.batch_size, args.projection_dim, args.knn_k,
                args.knn_chunk_size, args.queue_size, args.contrastive_dim, args.warmup_epochs)
    if min(positive) < 1 or not 0 <= args.dropout < 1 or not 0 <= args.knn_clean_rate <= 1:
        parser.error("Invalid positive integer, dropout, or KNN threshold")
    if args.queue_size < args.batch_size:
        parser.error("queue-size must be at least batch-size")
    if not 0 <= args.uncertain_weight <= 1 or args.nasc_weight < 0 or args.temperature <= 0:
        parser.error("Invalid loss weight or temperature")
    if not 0 <= args.sct_min_threshold <= 0.95:
        parser.error("sct-min-threshold must be in [0, 0.95]")
    train_x, train_y, class_to_idx = read_features(args.features_dir / "train_features.pt")
    val_x, val_y, val_classes = read_features(args.features_dir / "val_features.pt")
    if class_to_idx != val_classes or train_x.shape[1] != val_x.shape[1]:
        parser.error("Train/validation feature dimensions or class mappings differ")
    classes = len(class_to_idx)
    if sorted(class_to_idx.values()) != list(range(classes)):
        parser.error("Class IDs must be contiguous from zero")
    if min(train_y.min(), val_y.min()) < 0 or max(train_y.max(), val_y.max()) >= classes:
        parser.error("Feature labels are outside the class mapping")
    if train_x.shape[1] != 3072:
        parser.error("This DINOv2-L release expects 3072-dimensional PSM features")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geometry = knn_consistency(train_x, train_y, args.knn_k, args.knn_clean_rate,
                               device, args.knn_chunk_size)
    train_loader = DataLoader(TensorDataset(train_x, train_y, geometry), args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(TensorDataset(val_x, val_y), args.batch_size,
                            shuffle=False, num_workers=args.workers)
    model = DGCLClassifier(train_x.shape[1], classes, args.projection_dim, args.dropout,
                           args.queue_size, args.contrastive_dim, 0.999, args.temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    semantic = SemanticThreshold(classes, momentum=0.999)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stage2_config.json").write_text(
        json.dumps({**vars(args), "features_dir": str(args.features_dir),
                    "output_dir": str(args.output_dir), "class_to_idx": class_to_idx}, indent=2),
        encoding="utf-8")
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_ce = total_nasc = 0.0
        counts = torch.zeros(3, dtype=torch.long)
        for features, labels, geometric_clean in tqdm(train_loader, desc=f"Stage 2 {epoch}/{args.epochs}"):
            features, labels = features.to(device), labels.to(device)
            geometric_clean = geometric_clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            semantic_clean = semantic.select(F.softmax(logits.detach(), dim=1), labels)
            absolute_clean = semantic_clean & geometric_clean
            absolute_noise = ~semantic_clean & ~geometric_clean
            uncertain = semantic_clean ^ geometric_clean
            counts += torch.tensor([absolute_clean.sum().item(), uncertain.sum().item(),
                                    absolute_noise.sum().item()])
            weights = torch.ones_like(labels, dtype=logits.dtype) if epoch <= args.warmup_epochs else (
                absolute_clean.float() + args.uncertain_weight * uncertain.float())
            ce = (F.cross_entropy(logits, labels, reduction="none") * weights).mean()
            contrastive = (model.nasc.loss(features[absolute_clean, :model.token_dim], labels[absolute_clean])
                           if absolute_clean.any() else ce * 0)
            loss = ce + args.nasc_weight * contrastive
            loss.backward()
            optimizer.step()
            model.nasc.update(features[:, :model.token_dim], labels, absolute_noise)
            total_ce += ce.item()
            total_nasc += contrastive.item()
        semantic.update(epoch, args.warmup_epochs, initial_momentum=0.5,
                        min_threshold=args.sct_min_threshold)
        accuracy = evaluate(model, val_loader, device)
        ratios = (counts.float() / counts.sum()).tolist()
        print(f"Epoch {epoch}: CE={total_ce / len(train_loader):.4f}, "
              f"NASC={total_nasc / len(train_loader):.4f}, "
              f"clean/uncertain/noise={ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}, "
              f"val_acc={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({"model": model.state_dict(), "class_to_idx": class_to_idx,
                        "semantic_global_threshold": semantic.global_threshold,
                        "semantic_class_confidence": semantic.class_confidence,
                        "epoch": epoch, "val_accuracy": accuracy}, args.output_dir / "stage2_best.pt")
    print(f"Stage 2 complete; best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
