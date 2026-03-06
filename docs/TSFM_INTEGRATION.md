# TSFM Model Integration

This document describes the integration of the TSFM (Time Series Foundation Model) into the MobiBox backend for Human Activity Recognition (HAR).

## Overview

TSFM is a semantic-aligned foundation model for IMU-based activity recognition. It uses contrastive learning to align sensor embeddings with text embeddings, enabling zero-shot generalization to new activities.

### Training Data

The model was trained on **UCI-HAR dataset** with 6 activity labels:
- `walking`
- `walking_upstairs`
- `walking_downstairs`
- `sitting`
- `standing`
- `laying`

### Model Architecture

| Aspect | Previous Model | TSFM |
|--------|---------------|------|
| Architecture | Simple Transformer | Semantic-aligned encoder + contrastive learning |
| Classes | 7 (fixed) | 6 (trained), zero-shot capable |
| Input | 50 samples (1s @ 50Hz) | Variable length patches |
| Channels | 9 (acc/gyro/mag) | 6-52 (flexible) |
| Output | Log probabilities | Embedding + cosine similarity to labels |

## Files Added

### TSFM Model Package

```
src/celery_app/services/tsfm_model/
├── __init__.py              # Package exports
├── config.py                # Model configuration (SMALL_CONFIG, SMALL_DEEP_CONFIG)
├── encoder.py               # IMUActivityRecognitionEncoder
├── semantic_alignment.py    # SemanticAlignmentHead, CrossChannelFusion, etc.
├── token_text_encoder.py    # TokenTextEncoder, LearnableLabelBank
├── feature_extractor.py     # CNN feature extractors
├── transformer.py           # Transformer modules
├── positional_encoding.py   # Temporal and channel encoding
├── preprocessing.py        # IMU data preprocessing
├── label_groups.py          # Activity label groupings for zero-shot
├── model_loading.py         # Model loading utilities (adapted for MobiBox)
└── ckpts/
    ├── best.pt              # Trained model checkpoint (~400MB)
    └── hyperparameters.json # Model configuration
```

### TSFM Service

```
src/celery_app/services/tsfm_service.py
```

Provides:
- `run_tsfm_inference(imu_data)` - Run TSFM inference on IMU data
- `is_tsfm_available()` - Check if TSFM model is loaded
- Label mapping from TSFM labels to MobiBox classes

## Files Modified

### Configuration (`src/celery_app/config.py`)

Added TSFM configuration flags:

```python
# TSFM Model Configuration
USE_TSFM_MODEL = True  # Set to False to use legacy IMU transformer
TSFM_MIN_SAMPLES = 10  # Minimum IMU samples required for TSFM inference
```

### HAR Service (`src/celery_app/services/har_service.py`)

Updated `run_har_model()` to use TSFM with fallback chain:

```python
async def run_har_model(imu_data: list[dict]) -> tuple[str, float, str]:
    """
    Priority:
    1. TSFM model (if USE_TSFM_MODEL=True and checkpoint available)
    2. Legacy IMU transformer (if checkpoint configured)
    3. Mock model (fallback)
    """
```

### Dependencies (`environment.yml`)

Added:
```yaml
# TSFM model dependencies
- sentence-transformers
```

## Label Mapping

TSFM's trained labels (UCI-HAR) are mapped to MobiBox's 7 classes:

| MobiBox Label | TSFM Labels |
|---------------|-------------|
| walking | walking |
| climbing stairs | ascending_stairs, walking_upstairs, descending_stairs, walking_downstairs, stairs |
| sitting | sitting |
| standing | standing |
| lying | laying, lying |
| running | running, jogging (zero-shot) |
| unknown | Other predictions |

## Model Details

### Checkpoint Information

- **Source**: `/localdata/khliuae/tsfm/training_output/semantic_alignment/20260306_160938/best.pt`
- **Epoch**: 139
- **Architecture**: small_deep
- **Training Data**: UCI-HAR (6 classes)

### Hyperparameters

```json
{
  "model_size": "small_deep",
  "d_model": 384,
  "num_heads": 8,
  "num_temporal_layers": 8,
  "dim_feedforward": 1536,
  "feature_extractor_type": "spectral_temporal",
  "semantic_dim": 768,
  "per_patch_prediction": true
}
```

## Usage

### Running Inference

```python
from src.celery_app.services.tsfm_service import run_tsfm_inference, is_tsfm_available

# Check if model is available
if is_tsfm_available():
    # Run inference
    label, confidence, source = run_tsfm_inference(imu_data)
    # label: MobiBox label (walking, running, sitting, standing, lying, climbing stairs, unknown)
    # confidence: 0.0 to 1.0
    # source: "tsfm_model"
```

### Configuration

To disable TSFM and use legacy model:

```python
# In src/celery_app/config.py
USE_TSFM_MODEL = False
```

## Troubleshooting

### Common Issues

1. **Model not loading**: Ensure `best.pt` and `hyperparameters.json` are in `tsfm_model/ckpts/`

2. **MPS device errors**: The model is configured to use CPU to avoid Apple Silicon MPS compatibility issues with sentence-transformers

3. **Low confidence on predictions**: Normal for ambiguous activities. The model works best with 1+ seconds of IMU data at 50Hz

4. **Import errors**: Ensure `sentence-transformers` is installed:
   ```bash
   pip install sentence-transformers
   ```

## Performance Considerations

- **Memory**: TSFM model is larger (~400MB) than legacy transformer (~10MB)
- **Inference time**: First run downloads sentence-transformers model (~420MB for all-mpnet-base-v2)
- **CPU usage**: Model runs on CPU to avoid MPS compatibility issues
- **Input requirements**: Minimum 10 samples at 50Hz for meaningful inference

## Future Improvements

1. Train on additional HAR datasets for more activity classes
2. Add GPU support when MPS compatibility improves
3. Implement batch inference for multiple users
4. Add confidence threshold filtering
5. Fine-tune on MobiBox-specific data