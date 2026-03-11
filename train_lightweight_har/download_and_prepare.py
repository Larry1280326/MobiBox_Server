#!/usr/bin/env python3
"""
Complete script to download public HAR dataset and train TinierHAR model.

This script is standalone and can be copied to a remote GPU machine.
It downloads the UCI-HAR dataset and trains the model.

Usage:
    python train_complete.py --output checkpoints/

    # With custom parameters
    python train_complete.py --output checkpoints/ --epochs 200 --batch-size 128

    # Download only (skip training)
    python train_complete.py --download-only --output data/
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

# =============================================================================
# Step 1: Download UCI-HAR Dataset
# =============================================================================

UCI_HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
UCI_HAR_MIRROR = "https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip"


def download_uci_har(output_dir: Path) -> Path:
    """Download and extract UCI-HAR dataset."""
    dataset_dir = output_dir / "UCI_HAR_Dataset"
    if dataset_dir.exists():
        print(f"UCI-HAR already downloaded at {dataset_dir}")
        return dataset_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "UCI_HAR.zip"

    # Try primary URL, fall back to mirror
    urls = [UCI_HAR_URL, UCI_HAR_MIRROR]
    for url in urls:
        try:
            print(f"Downloading from {url}...")
            urlretrieve(url, zip_path)
            break
        except Exception as e:
            print(f"Failed: {e}")
            continue
    else:
        raise RuntimeError("Failed to download UCI-HAR from all sources")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    # Rename folder
    extracted = output_dir / "UCI HAR Dataset"
    if extracted.exists():
        extracted.rename(dataset_dir)

    zip_path.unlink(missing_ok=True)
    print(f"Downloaded to {dataset_dir}")
    return dataset_dir


# =============================================================================
# Step 2: Convert to Training Format
# =============================================================================

# UCI labels (1-6) to our labels (0-6)
# UCI: 1=WALKING, 2=UPSTAIRS, 3=DOWNSTAIRS, 4=SITTING, 5=STANDING, 6=LAYING
# Ours: 0=unknown, 1=standing, 2=sitting, 3=lying, 4=walking, 5=stairs, 6=running
UCI_TO_OURS = {1: 4, 2: 5, 3: 5, 4: 2, 5: 1, 6: 3}
LABEL_NAMES = ["unknown", "standing", "sitting", "lying", "walking", "climbing_stairs", "running"]


def load_uci_har(dataset_dir: Path, window_size: int = 50):
    """Load UCI-HAR raw signals and convert to training format."""
    print("Loading UCI-HAR data...")

    def load_split(split: str):
        signals_dir = dataset_dir / split / "Inertial Signals"

        # Load accelerometer (body acceleration)
        acc_x = np.loadtxt(signals_dir / f"body_acc_x_{split}.txt")
        acc_y = np.loadtxt(signals_dir / f"body_acc_y_{split}.txt")
        acc_z = np.loadtxt(signals_dir / f"body_acc_z_{split}.txt")

        # Load gyroscope
        gyro_x = np.loadtxt(signals_dir / f"body_gyro_x_{split}.txt")
        gyro_y = np.loadtxt(signals_dir / f"body_gyro_y_{split}.txt")
        gyro_z = np.loadtxt(signals_dir / f"body_gyro_z_{split}.txt")

        # Load labels and subjects
        labels = np.loadtxt(dataset_dir / split / f"y_{split}.txt").astype(int)
        subjects = np.loadtxt(dataset_dir / split / f"subject_{split}.txt").astype(int)

        # Stack: (N, 128, 6) - UCI has 128 timesteps
        signals = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=-1)
        return signals, labels, subjects

    train_signals, train_labels, train_subjects = load_split("train")
    test_signals, test_labels, test_subjects = load_split("test")

    # Combine
    all_signals = np.concatenate([train_signals, test_signals], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    all_subjects = np.concatenate([train_subjects, test_subjects], axis=0)

    print(f"Combined data: {all_signals.shape}")

    # Convert labels
    converted_labels = np.array([UCI_TO_OURS.get(l, 0) for l in all_labels])
    print(f"Label distribution: {np.bincount(converted_labels)}")

    # Window: UCI has 128 samples, we use center 50
    start_idx = (128 - window_size) // 2
    all_signals = all_signals[:, start_idx:start_idx + window_size, :]
    print(f"Windowed data: {all_signals.shape}")

    # Add magnetometer placeholder (UCI doesn't have magnetometer)
    mag = np.zeros((all_signals.shape[0], window_size, 3), dtype=np.float32)
    all_signals = np.concatenate([all_signals, mag], axis=-1)
    print(f"With magnetometer: {all_signals.shape}")

    # Split by subjects for cross-subject validation
    unique_subjects = np.unique(all_subjects)
    np.random.seed(42)
    np.random.shuffle(unique_subjects)

    n_val = int(len(unique_subjects) * 0.2)
    val_subjects = set(unique_subjects[:n_val])

    train_mask = ~np.isin(all_subjects, list(val_subjects))
    val_mask = np.isin(all_subjects, list(val_subjects))

    train_data = all_signals[train_mask].astype(np.float32)
    train_labels = converted_labels[train_mask]
    val_data = all_signals[val_mask].astype(np.float32)
    val_labels = converted_labels[val_mask]

    print(f"Train: {len(train_data)} samples")
    print(f"Val: {len(val_data)} samples")

    return train_data, train_labels, val_data, val_labels


def prepare_training_data(dataset_dir: Path, output_dir: Path, window_size: int = 50):
    """Prepare and save training data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, train_labels, val_data, val_labels = load_uci_har(dataset_dir, window_size)

    # Save
    np.save(output_dir / "train_data.npy", train_data)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "val_data.npy", val_data)
    np.save(output_dir / "val_labels.npy", val_labels)

    # Metadata
    metadata = {
        "source": "UCI_HAR",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "data_shape": list(train_data.shape),
        "window_size": window_size,
        "num_channels": 9,
        "num_classes": 7,
        "label_names": LABEL_NAMES,
        "train_label_distribution": {str(i): int(np.sum(train_labels == i)) for i in range(7)},
        "val_label_distribution": {str(i): int(np.sum(val_labels == i)) for i in range(7)},
        "notes": "Magnetometer is placeholder zeros. UCI has 6 activities mapped to 7 labels.",
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved training data to {output_dir}")
    print(f"Train distribution: {metadata['train_label_distribution']}")
    print(f"Val distribution: {metadata['val_label_distribution']}")

    return output_dir


# =============================================================================
# Step 3: Model Definition
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return self.bn(self.pointwise(self.depthwise(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, pool_size, dropout=0.1):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.pool(self.act(self.conv(x))))


class AttentionAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


class TinierHAR(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {
                "input_channels": 9, "window_size": 50,
                "stage1_channels": 64, "stage1_kernel": 5, "stage1_pool": 4,
                "stage2_channels": 128, "stage2_kernel": 3, "stage2_pool": 2,
                "gru_hidden": 64, "gru_layers": 1, "gru_dropout": 0.1,
                "hidden_dim": 64, "num_classes": 7, "dropout": 0.1,
            }
        self.config = config

        self.stage1 = ConvBlock(config["input_channels"], config["stage1_channels"],
                                 config["stage1_kernel"], config["stage1_pool"], config["dropout"])
        self.stage2 = ConvBlock(config["stage1_channels"], config["stage2_channels"],
                                 config["stage2_kernel"], config["stage2_pool"], config["dropout"])

        self.gru = nn.GRU(config["stage2_channels"], config["gru_hidden"],
                          config["gru_layers"], batch_first=True, bidirectional=True,
                          dropout=config["gru_dropout"] if config["gru_layers"] > 1 else 0)

        gru_out = config["gru_hidden"] * 2
        self.attention = AttentionAggregation(gru_out)
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out),
            nn.Linear(gru_out, config["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_dim"], config["num_classes"]),
        )
        self._init_weights()

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

    def forward(self, x):
        if x.size(-1) <= 12 and x.size(1) > 12:
            x = x.transpose(1, 2)
        x = self.stage1(x)
        x = self.stage2(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = self.attention(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Step 4: Training
# =============================================================================

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class IMUDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment
        mean = self.data.mean(dim=(0, 1), keepdim=True)
        std = self.data.std(dim=(0, 1), keepdim=True) + 1e-8
        self.data = (self.data - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        if self.augment:
            x = x + torch.randn_like(x) * 0.01
            x = x * np.random.uniform(0.9, 1.1)
        return x, y


def train(args):
    """Main training function."""
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_dir = Path(args.data_dir)
    train_data = np.load(data_dir / "train_data.npy")
    train_labels = np.load(data_dir / "train_labels.npy")
    val_data = np.load(data_dir / "val_data.npy")
    val_labels = np.load(data_dir / "val_labels.npy")

    print(f"Train: {train_data.shape}, Val: {val_data.shape}")

    # Dataset
    train_ds = IMUDataset(train_data, train_labels, augment=True)
    val_ds = IMUDataset(val_data, val_labels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = TinierHAR().to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Class weights
    counts = np.bincount(train_labels, minlength=7).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = torch.from_numpy(1.0 / (counts + 1)).float()
    weights = weights / weights.sum() * 7
    weights = weights.to(device)

    # Optimizer
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Training loop
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += logits.argmax(1).eq(y).sum().item()
            train_total += y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += logits.argmax(1).eq(y).sum().item()
                val_total += y.size(0)
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # F1 score
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, output_dir / "best.pt")
            print(f"  -> Saved best model (F1: {val_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Final save
    torch.save({"model_state_dict": model.state_dict()}, output_dir / "final.pt")

    # Metrics
    metrics = {
        "best_f1": best_f1,
        "epochs_trained": epoch + 1,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    print(f"Checkpoint saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download UCI-HAR and train TinierHAR")
    parser.add_argument("--output", type=str, default="checkpoints/", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data/uci_har_processed/", help="Data directory")
    parser.add_argument("--download-only", action="store_true", help="Download data only, skip training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--window-size", type=int, default=50, help="Window size")

    args = parser.parse_args()

    # Step 1: Download
    base_dir = Path(args.data_dir).parent.parent
    dataset_dir = download_uci_har(base_dir / "downloads")

    # Step 2: Prepare data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        data_dir = prepare_training_data(dataset_dir, base_dir / "data" / "uci_har_processed", args.window_size)

    if args.download_only:
        print("Download complete. Exiting.")
        return

    # Step 3: Train
    train(args)


if __name__ == "__main__":
    main()