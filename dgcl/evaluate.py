"""Evaluate a saved Stage 2 checkpoint on a compatible feature file."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from dgcl.stage2 import DGCLClassifier, evaluate, read_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    config = json.loads((args.checkpoint.parent / "stage2_config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    features, labels, class_to_idx = read_features(args.features)
    if class_to_idx != checkpoint["class_to_idx"]:
        parser.error("Feature class mapping does not match the checkpoint")
    model = DGCLClassifier(features.shape[1], len(class_to_idx), config["projection_dim"],
                           config["dropout"], config["queue_size"], config["contrastive_dim"],
                           0.999, config["temperature"])
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(TensorDataset(features, labels), batch_size=args.batch_size)
    accuracy = evaluate(model.to(device), loader, device)
    print(f"Accuracy: {accuracy * 100:.2f}% ({len(labels)} images)")


if __name__ == "__main__":
    main()
