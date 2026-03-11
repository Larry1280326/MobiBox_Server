# Lightweight HAR Model Implementation Plan

## Overview

Replace/augment the current TSFM model (~21M params) with a lightweight HAR model (~93K params) for efficient edge deployment while maintaining compatible 7-class output.

## Status: Complete ✅

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Model architecture implementation |
| Phase 2 | ✅ Complete | Data preparation pipeline |
| Phase 3 | ✅ Complete | Training pipeline and experiments |
| Phase 4 | ✅ Complete | Integration with existing HAR service |
| Phase 5 | 📋 Pending | Optimization and deployment prep (INT8 quantization) |

## Current State

- **Primary Model:** TSFM - ~21M parameters, zero-shot capable
- **Fallback Model:** Legacy IMU Transformer - 6-layer transformer
- **Input:** 9-channel IMU at 50Hz, 50-sample windows
- **Output:** 7 classes (unknown, standing, sitting, lying, walking, climbing stairs, running)

**New Addition:**
- **Lightweight HAR:** TinierHAR - ~93K parameters, depthwise separable conv + BiGRU + attention

---

## Recommended Models

### Option A: TinierHAR (Implemented ✓)
- **Parameters:** ~93K (225× smaller than TSFM, 2.4× smaller than legacy transformer)
- **Architecture:** Depthwise separable conv + bidirectional GRU + attention
- **Accuracy:** Expected comparable to DeepConvLSTM (needs training validation)
- **Source:** https://arxiv.org/html/2507.07949v1

### Option B: μBi-ConvLSTM
- **Parameters:** ~11.4K (smallest option)
- **Architecture:** 2-stage CNN + 4× temporal pooling + BiLSTM
- **Accuracy:** 93.41% F1 on UCI-HAR
- **Size:** 23KB (INT8 quantized)
- **Source:** https://arxiv.org/html/2602.06523v1

### Option C: DeepConvLSTM (Baseline)
- **Parameters:** ~130K
- **Architecture:** 4 conv layers + 2 LSTM layers
- **Accuracy:** 98.24% on WISDM
- **Well-documented:** Industry standard baseline

---

## Implementation Phases

### Phase 1: Model Architecture Implementation

**Files to create:**
```
src/celery_app/services/lightweight_har/
├── __init__.py
├── config.py                    # Model configuration
├── models/
│   ├── __init__.py
│   ├── tinier_har.py           # TinierHAR architecture
│   ├── micro_biconvlstm.py     # μBi-ConvLSTM architecture
│   └── deepconv_lstm.py        # DeepConvLSTM baseline
├── preprocessing.py            # Data preprocessing utilities
├── dataset.py                  # PyTorch dataset class
└── inference.py                # Inference service
```

**Tasks:**
1. Implement TinierHAR architecture
   - Depthwise separable convolutions
   - Temporal pooling (4× reduction)
   - Bidirectional GRU layer
   - Attention-based temporal aggregation
   - 7-class output head

2. Implement μBi-ConvLSTM as alternative
   - Two-stage convolution with pooling
   - Single bidirectional LSTM layer
   - Classification head

3. Create unified inference interface
   - Match existing HAR service API
   - Support model switching

### Phase 2: Data Preparation

**Tasks:**
1. Map current IMU format to model input
   - Current: 9 channels (acc, gyro, mag) at 50Hz
   - May need to adapt for models trained on 6 channels (acc + gyro only)

2. Create preprocessing pipeline
   - Z-score normalization (already have)
   - Windowing (already have 50-sample windows)
   - Handle missing channels if using pre-trained weights

3. Prepare training/validation data
   - Extract labeled data from Supabase
   - Create train/val/test splits (subject-independent)
   - Data augmentation: jittering, scaling, rotation

### Phase 3: Training Pipeline

**Files to create:**
```
src/celery_app/services/lightweight_har/
├── train.py                    # Training script
├── losses.py                   # Loss functions
└── evaluate.py                 # Evaluation metrics
```

**Tasks:**
1. Set up training loop with:
   - Cross-entropy loss with class weights (handle imbalance)
   - AdamW optimizer
   - Learning rate scheduling
   - Early stopping

2. Evaluation metrics:
   - Per-class F1 scores
   - Confusion matrix
   - Latency measurement

3. Cross-subject validation
   - Leave-one-subject-out (LOSO) validation
   - Ensure generalization to new users

### Phase 4: Integration

**Files to modify:**
- `src/celery_app/services/har_service.py` - Add lightweight model to fallback chain
- `src/celery_app/config.py` - Add configuration options

**Tasks:**
1. Update HAR service fallback chain:
   ```
   TSFM -> Lightweight HAR -> Legacy IMU Transformer -> Mock
   ```

2. Add configuration options:
   ```python
   LIGHTWEIGHT_HAR_MODEL = "tinier_har"  # or "micro_biconvlstm"
   LIGHTWEIGHT_HAR_CHECKPOINT = "path/to/checkpoint.pth"
   LIGHTWEIGHT_HAR_CONFIDENCE_THRESHOLD = 0.7
   ```

3. Create checkpoint loading utilities

### Phase 5: Optimization & Deployment

**Tasks:**
1. Model quantization
   - Post-training INT8 quantization
   - Measure accuracy degradation
   - Target: <1% loss with 4× size reduction

2. Performance profiling
   - Inference latency (target: <10ms)
   - Memory footprint
   - CPU/GPU utilization

3. Edge deployment preparation
   - Export to ONNX format
   - Export to TensorFlow Lite (optional)
   - Create mobile-friendly inference code

---

## Architecture Details

### TinierHAR Architecture

```
Input: (batch, 9, 50) - 9 channels, 50 timesteps

┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Depthwise Separable Conv Block                     │
│   - DepthwiseConv1d(9, 9, kernel=5, groups=9)               │
│   - PointwiseConv1d(9, 64)                                   │
│   - BatchNorm + GELU                                         │
│   - MaxPool1d(4) -> (batch, 64, 12)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Depthwise Separable Conv Block                     │
│   - DepthwiseConv1d(64, 64, kernel=3, groups=64)            │
│   - PointwiseConv1d(64, 128)                                 │
│   - BatchNorm + GELU                                         │
│   - MaxPool1d(2) -> (batch, 128, 6)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Bidirectional GRU                                  │
│   - BiGRU(128, 64, batch_first=True)                        │
│   - Output: (batch, 6, 128)                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Attention Temporal Aggregation                     │
│   - Attention weights: Linear(128, 1) + Softmax             │
│   - Weighted sum: (batch, 128)                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Classification Head                                         │
│   - Linear(128, 64) + GELU + Dropout                        │
│   - Linear(64, 7) -> 7 classes                               │
└─────────────────────────────────────────────────────────────┘

Total Parameters: ~34,000
```

### μBi-ConvLSTM Architecture

```
Input: (batch, 9, 50) - 9 channels, 50 timesteps

┌─────────────────────────────────────────────────────────────┐
│ Stage 1: First Conv Block                                   │
│   - Conv1d(9, 32, kernel=5, stride=1)                       │
│   - BatchNorm + ReLU                                         │
│   - MaxPool1d(4) -> (batch, 32, 12)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Second Conv Block                                  │
│   - Conv1d(32, 64, kernel=3, stride=1)                      │
│   - BatchNorm + ReLU                                         │
│   - MaxPool1d(2) -> (batch, 64, 6)                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Bidirectional LSTM                                 │
│   - BiLSTM(64, 32, batch_first=True)                        │
│   - Take last hidden state: (batch, 64)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Classification Head                                         │
│   - Linear(64, 32) + ReLU + Dropout                         │
│   - Linear(32, 7) -> 7 classes                               │
└─────────────────────────────────────────────────────────────┘

Total Parameters: ~11,400
```

---

## Estimated Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Phase 1 | 2-3 days | Model architecture implementation |
| Phase 2 | 1-2 days | Data preparation pipeline |
| Phase 3 | 2-3 days | Training pipeline and experiments |
| Phase 4 | 1-2 days | Integration with existing HAR service |
| Phase 5 | 1-2 days | Optimization and deployment prep |
| **Total** | **7-12 days** | Complete implementation |

---

## Success Criteria

1. **Model Size:** < 50K parameters (vs. 21M for TSFM)
2. **Accuracy:** > 90% F1 on validation set (7 classes)
3. **Latency:** < 10ms inference on CPU
4. **Compatibility:** Drop-in replacement for existing HAR pipeline
5. **Generalization:** < 5% accuracy drop on cross-subject validation

---

## References

1. [TinierHAR: Ultra-Lightweight HAR Models](https://arxiv.org/html/2507.07949v1)
2. [μBi-ConvLSTM: Ultra-Lightweight Model](https://arxiv.org/html/2602.06523v1)
3. [Efficient HAR on Edge Devices](https://www.nature.com/articles/s41598-025-98571-2)
4. [Current TSFM Implementation](./tsfm_model/)
5. [Current Legacy IMU Transformer](./imu_model_utils/)