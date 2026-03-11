"""Training script for TinierHAR model.

Usage:
    python -m src.celery_app.services.lightweight_har.train --data path/to/data.npy --labels path/to/labels.npy

Or with Supabase:
    python -m src.celery_app.services.lightweight_har.train --from-supabase --users user1,user2
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .config import TINIER_HAR_CONFIG, HAR_LABELS
from .dataset import IMUDataset
from .model import TinierHAR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.

    Uses inverse frequency weighting with smoothing.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1)  # Avoid division by zero

    # Inverse frequency with smoothing
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * num_classes

    return torch.from_numpy(weights).float()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item() * x.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate the model.

    Returns:
        Tuple of (average_loss, accuracy, all_predictions, all_labels)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)

        all_preds.append(predicted.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return avg_loss, accuracy, all_preds, all_labels


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict:
    """Compute detailed metrics including per-class F1 scores."""
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    # Overall metrics
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    class_f1 = {
        HAR_LABELS[i]: float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
        for i in range(num_classes)
    }

    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))

    return {
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision": float(precision),
        "recall": float(recall),
        "f1_per_class": class_f1,
        "confusion_matrix": cm.tolist(),
    }


def train(
    data_path: str,
    labels_path: str,
    output_dir: str,
    config: Optional[dict] = None,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    patience: int = 15,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train the TinierHAR model.

    Args:
        data_path: Path to numpy data file (samples, timesteps, channels)
        labels_path: Path to numpy labels file (samples,)
        output_dir: Directory to save model checkpoints
        config: Model configuration (uses default if None)
        epochs: Maximum number of epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: AdamW weight decay
        device: Device to train on ("cuda" or "cpu")
        patience: Early stopping patience
        val_split: Validation split ratio
        seed: Random seed

    Returns:
        Training history dictionary
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use default config if not provided
    if config is None:
        config = TINIER_HAR_CONFIG

    num_classes = config["num_classes"]

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {data_path}")
    data = np.load(data_path)
    labels = np.load(labels_path)

    logger.info(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    logger.info(f"Label distribution: {np.bincount(labels)}")

    # Create dataset
    dataset = IMUDataset(data, labels, augment=True)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Create model
    model = TinierHAR(config)
    model.to(device)

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Model size: {model.model_size_mb():.2f} MB")

    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(labels, num_classes).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Training loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        # Compute metrics
        val_metrics = compute_metrics(val_preds, val_labels, num_classes)
        val_f1 = val_metrics["f1_macro"]

        # Update learning rate
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )

        # Save best model
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
            logger.info(f"Saved best model with F1: {val_f1:.4f}")

            # Save metrics
            with open(output_dir / "best_metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2)
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping after {epoch+1} epochs without improvement")
            break

    # Save final model
    final_checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "history": history,
    }
    torch.save(final_checkpoint, output_dir / "final.pt")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best Val F1: {best_val_f1:.4f}")

    return history


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

    train(
        data_path=args.data,
        labels_path=args.labels,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()