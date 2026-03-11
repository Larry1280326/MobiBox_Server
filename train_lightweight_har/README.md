# Lightweight HAR Training Package

This package contains standalone files for training the TinierHAR model on a remote GPU machine.

## Quick Start (One Command)

The fastest way to train - downloads UCI-HAR dataset automatically and trains the model:

```bash
# On remote GPU machine
python download_and_prepare.py --output checkpoints/
```

This will:
1. Download UCI-HAR dataset from official sources
2. Convert to training format
3. Train TinierHAR model
4. Save checkpoint to `checkpoints/best.pt`

## Step-by-Step Guide

### 1. Copy files to remote machine

```bash
# Copy the entire training package
scp -r train_lightweight_har/ user@remote:~/har_training/
```

### 2. Setup environment on remote

```bash
ssh user@remote
cd ~/har_training

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train (auto-downloads data)

```bash
# Automatic: downloads UCI-HAR and trains
python download_and_prepare.py --output checkpoints/

# With custom parameters
python download_and_prepare.py \
    --output checkpoints/ \
    --epochs 200 \
    --batch-size 128 \
    --lr 5e-4 \
    --device cuda
```

### 4. Copy checkpoint back

```bash
# From local machine
scp user@remote:~/har_training/checkpoints/best.pt \
    ../src/celery_app/services/lightweight_har/ckpts/
```

## Available Datasets

### UCI-HAR (Recommended)
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+)
- **Size:** 10,299 samples, 30 subjects
- **Activities:** Walking, Upstairs, Downstairs, Sitting, Standing, Laying
- **Sensors:** Accelerometer + Gyroscope (matches our use case)
- **Sampling:** 50Hz (same as our data)

### WISDM (Alternative)
- **Source:** [WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php)
- **Size:** 1.1M samples
- **Activities:** Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
- **Sensors:** Accelerometer only (no gyroscope)

## Label Mapping

UCI-HAR labels are mapped to our labels:

| UCI-HAR Activity | Our Label |
|-------------------|-----------|
| WALKING | walking (4) |
| WALKING_UPSTAIRS | climbing_stairs (5) |
| WALKING_DOWNSTAIRS | climbing_stairs (5) |
| SITTING | sitting (2) |
| STANDING | standing (1) |
| LAYING | lying (3) |

Note: UCI-HAR doesn't have "unknown" or "running" activities.

## Training Arguments

```
--output         Output directory (default: checkpoints/)
--data-dir       Data directory (default: data/uci_har_processed/)
--download-only  Download data only, skip training
--epochs         Maximum epochs (default: 100)
--batch-size     Batch size (default: 64)
--lr             Learning rate (default: 1e-3)
--weight-decay   Weight decay (default: 1e-4)
--device         Device: cuda or cpu (default: cuda)
--patience       Early stopping patience (default: 15)
--window-size    Window size (default: 50)
```

## Output Files

After training:

```
checkpoints/
├── best.pt             # Best model checkpoint (copy this to repo)
├── final.pt            # Final model checkpoint
└── metrics.json        # Training metrics
```

## Data Format

Training data files:
- `train_data.npy`: Shape `(N, 50, 9)` - N samples, 50 timesteps, 9 channels
- `train_labels.npy`: Shape `(N,)` - Integer labels 0-6
- `val_data.npy`: Validation data
- `val_labels.npy`: Validation labels
- `metadata.json`: Dataset statistics

Channel order (9 channels):
1. acc_X, acc_Y, acc_Z (accelerometer)
2. gyro_X, gyro_Y, gyro_Z (gyroscope)
3. mag_X, mag_Y, mag_Z (magnetometer - placeholder zeros in UCI-HAR)

## Model Architecture

```
TinierHAR (~93K parameters):
├── Stage 1: Depthwise Separable Conv (9 -> 64 channels, 4x pooling)
├── Stage 2: Depthwise Separable Conv (64 -> 128 channels, 2x pooling)
├── Stage 3: Bidirectional GRU (hidden=64)
├── Stage 4: Attention Aggregation
└── Classifier: Linear -> GELU -> Linear (7 classes)
```

## Training on Different Data

If you have your own labeled IMU data:

```bash
# 1. Prepare your data as numpy arrays
# train_data.npy: (N, 50, 9) float32
# train_labels.npy: (N,) int64 (labels 0-6)

# 2. Create val split
# val_data.npy, val_labels.npy

# 3. Train
python train.py \
    --data train_data.npy \
    --labels train_labels.npy \
    --output checkpoints/
```

## Files in This Package

| File | Description |
|------|-------------|
| `download_and_prepare.py` | Complete script: download + train |
| `train.py` | Standalone training script |
| `requirements.txt` | Python dependencies |

## Troubleshooting

### CUDA Out of Memory
```bash
--batch-size 32
```

### Slow Download
The UCI-HAR dataset is ~58MB. If the primary URL fails, the script automatically tries a mirror.

### Poor Accuracy
1. Check label distribution in `metadata.json`
2. Try different learning rate: `--lr 5e-4`
3. Increase epochs: `--epochs 200`

## Citation

If you use UCI-HAR dataset, cite:
> D. Anguita et al., "A Public Domain Dataset for Human Activity Recognition using Smartphones," ESANN 2013.