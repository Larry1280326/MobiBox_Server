# Lightweight HAR Training Package

This package contains standalone files for training the TinierHAR model on a remote GPU machine.

## Quick Start

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

### 3. Copy training data

```bash
# From local machine
scp data/train_data.npy data/train_labels.npy user@remote:~/har_training/data/
```

### 4. Train

```bash
# On remote machine
python train.py \
    --data data/train_data.npy \
    --labels data/train_labels.npy \
    --output checkpoints/ \
    --epochs 200 \
    --batch-size 128 \
    --device cuda
```

### 5. Copy checkpoint back

```bash
# From local machine
scp user@remote:~/har_training/checkpoints/best.pt \
    ../src/celery_app/services/lightweight_har/ckpts/
```

## Files

| File | Description |
|------|-------------|
| `train.py` | Standalone training script (includes model definition) |
| `requirements.txt` | Python dependencies |

## Training Arguments

```
--data          Path to training data (.npy file)
--labels        Path to training labels (.npy file)
--output        Output directory (default: checkpoints/)
--epochs        Maximum epochs (default: 100)
--batch-size    Batch size (default: 64)
--lr            Learning rate (default: 1e-3)
--weight-decay  AdamW weight decay (default: 1e-4)
--device        Device: cuda or cpu (default: cuda)
--patience      Early stopping patience (default: 15)
--val-split     Validation split (default: 0.2)
--seed          Random seed (default: 42)
```

## Output Files

After training, the output directory contains:

```
checkpoints/
├── best.pt             # Best model checkpoint (copy this to repo)
├── best_metrics.json   # Best validation metrics
├── final.pt            # Final model checkpoint
└── history.json        # Training history
```

## Data Format

Training data should be numpy arrays:

- `train_data.npy`: Shape `(N, 50, 9)` - N samples, 50 timesteps, 9 channels
- `train_labels.npy`: Shape `(N,)` - Integer labels 0-6

Label mapping:
```
0: unknown
1: standing
2: sitting
3: lying
4: walking
5: climbing stairs
6: running
```

## Model Architecture

```
TinierHAR (~93K parameters):
├── Stage 1: Depthwise Separable Conv (9 -> 64 channels, 4x pooling)
├── Stage 2: Depthwise Separable Conv (64 -> 128 channels, 2x pooling)
├── Stage 3: Bidirectional GRU (hidden=64)
├── Stage 4: Attention Aggregation
└── Classifier: Linear -> GELU -> Linear (7 classes)
```

## Example Training Session

```
$ python train.py --data data/train_data.npy --labels data/train_labels.npy --output checkpoints/
Using device: cuda
Loading data from data/train_data.npy
Data shape: (5000, 50, 9), Labels shape: (5000,)
Label distribution: [500 500 500 500 500 500 500]
Model parameters: 92,981

Epoch 1/100 - Train Loss: 1.8234, Train Acc: 0.2345 - Val Loss: 1.7234, Val Acc: 0.3012, Val F1: 0.2456
Epoch 10/100 - Train Loss: 0.6123, Train Acc: 0.7623 - Val Loss: 0.7234, Val Acc: 0.7412, Val F1: 0.7234
...
Saved best model with F1: 0.8567

Training complete. Best Val F1: 0.8567
Checkpoint saved to: checkpoints/
```