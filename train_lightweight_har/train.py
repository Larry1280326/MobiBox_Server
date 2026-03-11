#!/usr/bin/env python3
"""
Standalone training script for TinierHAR model.

This script can be copied to a remote GPU machine for training.
It includes the model architecture and training loop without
dependencies on the main codebase.

Usage:
    python train.py --data train_data.npy --labels train_labels.npy --output checkpoints/
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


# =============================================================================
# Model Definition
# =============================================================================

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for efficient feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ConvBlock(nn.Module):
    """Convolutional block with depthwise separable conv, activation, and pooling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int, dropout: float = 0.1):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.activation = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class AttentionAggregation(nn.Module):
    """Attention-based temporal aggregation."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        output = (x * weights).sum(dim=1)
        return output


class TinierHAR(nn.Module):
    """TinierHAR: Ultra-lightweight HAR model for IMU-based activity recognition."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__()

        if config is None:
            config = self._default_config()

        self.config = config
        self.input_channels = config["input_channels"]
        self.window_size = config["window_size"]

        # Stage 1
        self.stage1 = ConvBlock(
            config["input_channels"], config["stage1_channels"],
            config["stage1_kernel"], config["stage1_pool"],
            config.get("dropout", 0.1),
        )

        # Stage 2
        self.stage2 = ConvBlock(
            config["stage1_channels"], config["stage2_channels"],
            config["stage2_kernel"], config["stage2_pool"],
            config.get("dropout", 0.1),
        )

        # GRU
        self.gru = nn.GRU(
            input_size=config["stage2_channels"],
            hidden_size=config["gru_hidden"],
            num_layers=config["gru_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=config["gru_dropout"] if config["gru_layers"] > 1 else 0,
        )

        gru_output_size = config["gru_hidden"] * 2

        # Attention
        self.attention = AttentionAggregation(gru_output_size)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_output_size),
            nn.Linear(gru_output_size, config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], config["num_classes"]),
        )

        self._init_weights()

    def _default_config(self) -> dict:
        return {
            "input_channels": 9,
            "window_size": 50,
            "stage1_channels": 64,
            "stage1_kernel": 5,
            "stage1_pool": 4,
            "stage2_channels": 128,
            "stage2_kernel": 3,
            "stage2_pool": 2,
            "gru_hidden": 64,
            "gru_layers": 1,
            "gru_dropout": 0.1,
            "attention_dim": 128,
            "hidden_dim": 64,
            "num_classes": 7,
            "dropout": 0.1,
        }

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle (B, T, C) input
        if x.size(-1) <= 12 and x.size(1) > 12:
            x = x.transpose(1, 2)

        x = self.stage1(x)
        x = self.stage2(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = self.attention(x)
        logits = self.classifier(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Dataset Definition
# =============================================================================

class IMUDataset(Dataset):
    """Dataset for IMU-based HAR with augmentation."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

        # Normalize
        mean = self.data.mean(dim=(0, 1), keepdim=True)
        std = self.data.std(dim=(0, 1), keepdim=True) + 1e-8
        self.data = (self.data - mean) / std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        x = self.data[idx]
        y = self.labels[idx]

        if self.augment:
            # Gaussian noise
            x = x + torch.randn_like(x) * 0.01
            # Random scaling
            x = x * np.random.uniform(0.9, 1.1)

        return x, y


# =============================================================================
# Training Functions
# =============================================================================

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights).float()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        total += y.size(0)

        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    return total_loss / total, correct / total, np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """Compute detailed metrics."""
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    label_names = ["unknown", "standing", "sitting", "lying", "walking", "climbing stairs", "running"]

    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)

    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    class_f1 = {label_names[i]: float(f1_per_class[i]) if i < len(f1_per_class) else 0.0 for i in range(num_classes)}

    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

    return {
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
        "f1_per_class": class_f1,
        "confusion_matrix": cm.tolist(),
    }


def train(args):
    """Main training function."""
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {args.data}")
    data = np.load(args.data)
    labels = np.load(args.labels)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Config
    config = {
        "input_channels": data.shape[2] if data.ndim == 3 else 9,
        "window_size": data.shape[1] if data.ndim == 3 else 50,
        "stage1_channels": 64,
        "stage1_kernel": 5,
        "stage1_pool": 4,
        "stage2_channels": 128,
        "stage2_kernel": 3,
        "stage2_pool": 2,
        "gru_hidden": 64,
        "gru_layers": 1,
        "gru_dropout": 0.1,
        "attention_dim": 128,
        "hidden_dim": 64,
        "num_classes": 7,
        "dropout": 0.1,
    }

    # Dataset
    dataset = IMUDataset(data, labels, augment=True)

    # Split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = TinierHAR(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Loss, optimizer, scheduler
    class_weights = compute_class_weights(labels, config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Training loop
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_preds, val_labels, config["num_classes"])
        val_f1 = val_metrics["f1_macro"]

        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Log
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "config": config,
            }
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"Saved best model with F1: {val_f1:.4f}")

            with open(output_dir / "best_metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2)
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    # Save final
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
    }, output_dir / "final.pt")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train TinierHAR model")
    parser.add_argument("--data", type=str, required=True, help="Path to data numpy file")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels numpy file")
    parser.add_argument("--output", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()