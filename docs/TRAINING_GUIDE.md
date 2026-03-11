# Training Guide for Lightweight HAR Model

This guide explains how to train the TinierHAR model on a remote GPU machine.

## Quick Start (Recommended)

The fastest way to train - auto-downloads UCI-HAR public dataset:

```bash
# On remote GPU machine
cd train_lightweight_har
pip install -r requirements.txt
python download_and_prepare.py --output checkpoints/
```

This automatically:
1. Downloads UCI-HAR dataset (public benchmark)
2. Converts to training format
3. Trains TinierHAR model
4. Saves checkpoint to `checkpoints/best.pt`

## Overview

Two options for training data:

| Option | Description | Best For |
|--------|-------------|----------|
| **Public Dataset** | UCI-HAR (auto-download) | Quick start, baseline model |
| **Custom Data** | Export from Supabase | Your specific use case |

## Option 1: Public Dataset (Recommended)

### Datasets Available

**UCI-HAR Dataset** (Recommended)
- 10,299 samples, 30 subjects
- 6 activities: Walking, Upstairs, Downstairs, Sitting, Standing, Laying
- Accelerometer + Gyroscope at 50Hz
- Auto-downloads from: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+

**WISDM Dataset** (Alternative)
- 1.1M samples
- Accelerometer only
- Download with: `python scripts/download_har_dataset.py --dataset wisdm`

### Quick Commands

```bash
# Download and train (UCI-HAR)
python download_and_prepare.py --output checkpoints/

# Download only (skip training)
python download_and_prepare.py --download-only --output data/

# Train with custom parameters
python download_and_prepare.py \
    --output checkpoints/ \
    --epochs 200 \
    --batch-size 128 \
    --lr 5e-4 \
    --device cuda
```

## Option 2: Custom Data from Supabase

If you have labeled IMU data in your database:

### Export from Supabase

```bash
# Activate virtual environment
source .venv/bin/activate

# Export all labeled data from last 30 days
python scripts/export_training_data.py --output data/

# Or specify parameters
python scripts/export_training_data.py \
    --output data/ \
    --min-samples-per-class 50 \
    --val-split 0.2 \
    --days-back 30
```

**Output files:**
```
data/
├── train_data.npy      # Training IMU data (N, 50, 9)
├── train_labels.npy    # Training labels (N,)
├── val_data.npy        # Validation IMU data
├── val_labels.npy      # Validation labels
└── metadata.json       # Dataset statistics
```

## Step 2: Transfer Data to Remote Machine

```bash
# Option 1: Using scp
scp -r data/ user@remote:/path/to/training/

# Option 2: Using rsync
rsync -avz data/ user@remote:/path/to/training/data/
```

## Step 3: Setup Remote Environment

SSH into the remote machine and set up the environment:

```bash
# SSH to remote
ssh user@remote

# Create directory
mkdir -p ~/har_training
cd ~/har_training

# Clone repository (or copy required files)
git clone <repo_url> .
# OR copy just the training files
# scp -r user@local:/path/to/MobiBox_Server/src/celery_app/services/lightweight_har .

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch numpy scikit-learn tqdm
```

## Step 4: Train Model

### Basic Training

```bash
# Use absolute paths to data files
python -m src.celery_app.services.lightweight_har.train \
    --data /path/to/data/train_data.npy \
    --labels /path/to/data/train_labels.npy \
    --output checkpoints/ \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --device cuda
```

### Full Training Command

```bash
python -m src.celery_app.services.lightweight_har.train \
    --data /path/to/data/train_data.npy \
    --labels /path/to/data/train_labels.npy \
    --output checkpoints/ \
    --epochs 200 \
    --batch-size 128 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --device cuda \
    --patience 20 \
    --val-split 0.2 \
    --seed 42
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | Required | Path to training data (.npy) |
| `--labels` | Required | Path to training labels (.npy) |
| `--output` | `checkpoints/` | Output directory |
| `--epochs` | 100 | Maximum epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--device` | `cuda` | Device (cuda/cpu) |
| `--patience` | 15 | Early stopping patience |
| `--val-split` | 0.2 | Validation split ratio |
| `--seed` | 42 | Random seed |

### Monitoring Training

Training logs are printed to console and saved to:
- `checkpoints/history.json` - Training history
- `checkpoints/best_metrics.json` - Best validation metrics

**Example output:**
```
Epoch 50/200 - Train Loss: 0.4521, Train Acc: 0.8234 - Val Loss: 0.5123, Val Acc: 0.8012, Val F1: 0.7856
Saved best model with F1: 0.7856
```

## Step 5: Validate Results

After training, check the metrics:

```bash
# View best metrics
cat checkpoints/best_metrics.json

# View training history
python -c "import json; print(json.dumps(json.load(open('checkpoints/history.json')), indent=2))"
```

**Expected metrics for a well-trained model:**
- F1 Macro: > 0.85
- Accuracy: > 0.88
- Per-class F1: > 0.80 for each class

## Step 6: Transfer Checkpoint Back

```bash
# Copy best checkpoint to local machine
scp user@remote:/path/to/training/checkpoints/best.pt \
    src/celery_app/services/lightweight_har/ckpts/
```

## Step 7: Update Configuration

Update the config to use the new checkpoint:

```python
# In src/celery_app/config.py
LIGHTWEIGHT_HAR_CHECKPOINT = "src/celery_app/services/lightweight_har/ckpts/best.pt"
```

## Alternative: Training with Public Datasets

If you don't have enough labeled data, you can pre-train on public datasets:

### UCI-HAR Dataset

1. Download from: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
2. Extract to `data/uci_har/`
3. Convert format:

```python
# scripts/convert_uci_har.py
import numpy as np

# Load UCI-HAR data
train_data = np.loadtxt('data/uci_har/train/X_train.txt')
train_labels = np.loadtxt('data/uci_har/train/y_train.txt')

# Reshape to (N, 50, 9) - UCI-HAR has 561 features, need to adapt
# This requires custom preprocessing based on your needs
```

### WISDM Dataset

1. Download from: https://researchdata.ands.org.au/human-activity-recognition-using-smartphone-sensor/1180375
2. Convert to our format

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 32

# Or use gradient accumulation (modify train.py)
```

### Poor Accuracy
1. Check data quality: `cat data/metadata.json`
2. Ensure balanced classes or use class weights (default)
3. Increase training data
4. Try different learning rates: `--lr 1e-4` or `--lr 5e-4`

### Slow Training
```bash
# Use larger batch size if memory allows
--batch-size 256

# Reduce validation frequency (modify train.py)
```

## Quick Reference

```bash
# Full pipeline on remote machine
cd ~/har_training
source .venv/bin/activate

# Train
python -m src.celery_app.services.lightweight_har.train \
    --data data/train_data.npy \
    --labels data/train_labels.npy \
    --output checkpoints/ \
    --epochs 200 \
    --batch-size 128 \
    --device cuda \
    --patience 20

# Check results
cat checkpoints/best_metrics.json
```

## Model Architecture

The TinierHAR model has ~93K parameters:

| Layer | Parameters | Output Shape |
|-------|------------|--------------|
| Stage 1 Conv | 9,389 | (B, 64, 12) |
| Stage 2 Conv | - | (B, 128, 6) |
| BiGRU | 74,496 | (B, 6, 128) |
| Attention | 129 | (B, 128) |
| Classifier | 8,967 | (B, 7) |
| **Total** | **92,981** | - |

## Next Steps

1. **Quantization**: Convert to INT8 for edge deployment
2. **ONNX Export**: For cross-platform inference
3. **Fine-tuning**: Continue training with new data

See `docs/lightweight_har_plan.md` for more details.