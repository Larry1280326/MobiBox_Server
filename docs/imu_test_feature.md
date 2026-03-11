# IMU Test Feature - Complete Documentation

## Overview

The IMU Test feature enables evaluation of the TSFM (Time Series Foundation Model) activity recognition model by allowing users to collect labeled IMU data on an Android device and compare model predictions against ground truth labels.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Backend Implementation](#backend-implementation)
3. [Frontend Implementation](#frontend-implementation)
4. [API Reference](#api-reference)
5. [Database Schema](#database-schema)
6. [Model Fallback Chain](#model-fallback-chain)
7. [Configuration](#configuration)
8. [Setup Instructions](#setup-instructions)
9. [Troubleshooting](#troubleshooting)
10. [Git Branches](#git-branches)

---

## Architecture

```
┌──────────────────────────────────────┐         ┌──────────────────────────────────────┐
│         Android App (Frontend)        │         │         Backend API (FastAPI)        │
│                                      │         │                                      │
│  ┌─────────────────────────────────┐ │         │  ┌─────────────────────────────────┐ │
│  │     ImuTestActivity             │ │         │  │   /imu_test/predict             │ │
│  │  ┌───────────────────────────┐  │ │         │  │   - Validates IMU data         │ │
│  │  │  Activity Selection      │  │ │         │  │   - Runs HAR model inference    │ │
│  │  │  (walking, running, etc) │  │ │         │  │   - Compares with ground truth │ │
│  │  └───────────────────────────┘  │ │         │  │   - Stores result in database  │ │
│  │  ┌───────────────────────────┐  │ │         │  └─────────────────────────────────┘ │
│  │  │  IMU Data Collection      │  │ │         │                                      │
│  │  │  - 50Hz sampling rate    │  │ │         │  ┌─────────────────────────────────┐ │
│  │  │  - 3 seconds duration   │  │ │  HTTP   │  │   /imu_test/statistics          │ │
│  │  │  - ~150 samples expected │  │ │ ──────▶ │  │   - Accuracy metrics          │ │
│  │  └───────────────────────────┘  │ │ ◀────── │  │   - Per-label breakdown        │ │
│  │  ┌───────────────────────────┐  │ │         │  └─────────────────────────────────┘ │
│  │  │  Results Display         │  │ │         │                                      │
│  │  │  - Predicted label       │  │ │         │  ┌─────────────────────────────────┐ │
│  │  │  - Confidence score      │  │ │         │  │   HAR Model Chain              │ │
│  │  │  - Model source          │  │ │         │  │   1. TSFM (primary)             │ │
│  │  │  - Accuracy comparison   │  │ │         │  │   2. Legacy IMU Transformer     │ │
│  │  └───────────────────────────┘  │ │         │  │   3. Mock (fallback)           │ │
│  └─────────────────────────────────┘ │         │  └─────────────────────────────────┘ │
└──────────────────────────────────────┘         └──────────────────────────────────────┘
                                                          │
                                                          ▼
                                                 ┌───────────────────┐
                                                 │   Supabase DB     │
                                                 │   imu_test_results│
                                                 └───────────────────┘
```

---

## Backend Implementation

### File Structure

```
MobiBox_Server/
├── src/imu_test/
│   ├── __init__.py          # Module initialization
│   ├── router.py            # API endpoints
│   ├── schemas.py           # Request/Response schemas
│   └── service.py           # Business logic
├── src/celery_app/services/
│   ├── har_service.py       # HAR model orchestration (modified)
│   └── tsfm_service.py       # TSFM model inference
├── src/celery_app/config.py # Configuration (TSFM_MIN_SAMPLES, etc.)
├── src/test/
│   └── test_imu_model_loading.py  # Unit tests for IMU model
└── migrations/
    └── 006_imu_test_results.sql  # Database migration
```

### Key Components

#### 1. Router (`src/imu_test/router.py`)

```python
@router.post("/imu_test/predict")
async def imu_predict(request: IMUTestRequest):
    """Predict activity from IMU data."""
    result = await predict_activity(request)
    await save_test_result(result)
    return result

@router.get("/imu_test/statistics")
async def get_statistics(user: Optional[str] = None):
    """Get accuracy statistics."""
    return await get_test_statistics(user)

@router.get("/imu_test/labels")
async def get_valid_labels():
    """Get valid activity labels."""
    return {"labels": VALID_ACTIVITY_LABELS}
```

#### 2. Service (`src/imu_test/service.py`)

```python
async def predict_activity(request: IMUTestRequest, timeout_seconds: float = 10.0):
    """Run HAR model with timeout protection."""
    try:
        predicted_label, confidence, source = await asyncio.wait_for(
            run_har_model(imu_data),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        predicted_label = "unknown"
        confidence = 0.0
        source = "timeout"
    # ... validation and comparison logic
```

#### 3. HAR Service (`src/celery_app/services/har_service.py`)

```python
async def run_har_model(imu_data: list[dict]) -> tuple[str, float, str]:
    """
    Model fallback chain:
    1. TSFM model (if available and enough samples)
    2. Legacy IMU transformer (if checkpoint exists)
    3. Mock model (random based on acceleration magnitude)
    """
    if USE_TSFM_MODEL and is_tsfm_available() and len(imu_data) >= TSFM_MIN_SAMPLES:
        return await asyncio.to_thread(run_tsfm_inference, imu_data)

    model, available = _get_imu_model()
    if available:
        return await _run_imu_model(imu_data)

    return await run_mock_har_model(imu_data)
```

---

## Frontend Implementation

### File Structure

```
MobiQA-Android/
├── app/src/main/java/com/example/mobibox/
│   ├── ImuTestActivity.java       # Main activity for IMU testing
│   ├── MainActivity.java          # Added IMU test button (modified)
│   ├── Constants.java             # Added ENDPOINT_IMU_TEST_PREDICT (modified)
│   └── network/HttpApiClient.java # Added getHttpClient() (modified)
├── app/src/main/res/layout/
│   ├── activity_imu_test.xml      # IMU test activity layout
│   └── activity_main.xml          # Added IMU test button (modified)
└── app/src/main/AndroidManifest.xml # Registered ImuTestActivity (modified)
```

### Key Implementation Details

#### 1. Accurate 50Hz Sampling

The original implementation used `Handler.postDelayed()` which was inaccurate due to UI thread overhead. Fixed using `ScheduledExecutorService`:

```java
// Accurate 50Hz sampling (20ms intervals)
scheduledExecutor.scheduleAtFixedRate(() -> {
    // Collect sample from SensorDataManager
    float[] accel = sensorDataManager.getAccelData();
    float[] gyro = sensorDataManager.getGyroData();
    float[] mag = sensorDataManager.getMagData();

    IMUSample sample = new IMUSample(System.currentTimeMillis(), accel, gyro, mag);
    collectedSamples.add(sample);

    // Throttled UI updates (every 10 samples)
    if (collectedSamples.size() % 10 == 0) {
        mainHandler.post(() -> updateProgress());
    }
}, 0, 20, TimeUnit.MILLISECONDS);  // 50Hz = 20ms interval
```

#### 2. Data Collection Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Sampling Rate | 50 Hz | Matches backend HAR model |
| Collection Duration | 3 seconds | Standard evaluation window |
| Expected Samples | 150 | 50 Hz × 3 seconds |
| Minimum Required | 50 | Backend TSFM_MIN_SAMPLES |

#### 3. Activity Labels

```java
private String formatSource(String source) {
    switch (source) {
        case "tsfm_model": return "TSFM (AI Model)";
        case "imu_model": return "Legacy IMU Model";
        case "mock_har": return "Mock (Debug)";
        case "timeout": return "Timeout";
        case "insufficient_data": return "Insufficient Data";
        default: return source;
    }
}
```

---

## API Reference

### POST `/imu_test/predict`

**Request:**
```json
{
  "user": "user123",
  "ground_truth_label": "walking",  // Optional - for evaluation
  "imu_data": [
    {
      "timestamp": "2024-03-08T12:00:00.000Z",
      "acc_X": 0.12,
      "acc_Y": 9.81,
      "acc_Z": -0.05,
      "gyro_X": 0.001,
      "gyro_Y": 0.002,
      "gyro_Z": -0.001,
      "mag_X": 45.2,
      "mag_Y": -12.3,
      "mag_Z": 38.7
    }
    // ... minimum 50 samples
  ]
}
```

**Response:**
```json
{
  "user": "user123",
  "predicted_label": "walking",
  "confidence": 0.85,
  "source": "tsfm_model",
  "ground_truth_label": "walking",
  "is_correct": true,
  "sample_count": 150,
  "timestamp": "2024-03-08T12:00:03.000Z"
}
```

### GET `/imu_test/statistics?user=<user_id>`

**Response:**
```json
{
  "total": 10,
  "correct": 8,
  "accuracy": 0.8,
  "per_label": {
    "walking": {"total": 3, "correct": 3, "accuracy": 1.0},
    "running": {"total": 2, "correct": 1, "accuracy": 0.5},
    "sitting": {"total": 5, "correct": 4, "accuracy": 0.8}
  }
}
```

### GET `/imu_test/labels`

**Response:**
```json
{
  "labels": [
    "walking",
    "running",
    "sitting",
    "standing",
    "lying",
    "climbing stairs",
    "unknown"
  ]
}
```

---

## Database Schema

### Table: `imu_test_results`

```sql
CREATE TABLE imu_test_results (
    id BIGSERIAL PRIMARY KEY,
    "user" TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'tsfm_model',
    ground_truth_label TEXT,
    is_correct BOOLEAN,
    sample_count INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for query performance
CREATE INDEX idx_imu_test_results_user ON imu_test_results("user");
CREATE INDEX idx_imu_test_results_timestamp ON imu_test_results(timestamp);
CREATE INDEX idx_imu_test_results_ground_truth ON imu_test_results(ground_truth_label);
CREATE INDEX idx_imu_test_results_is_correct ON imu_test_results(is_correct);
```

**Note:** `user` is a reserved keyword in PostgreSQL, so it must be quoted.

---

## Model Fallback Chain

```
┌─────────────────────────────────────────────────────────────────┐
│                     HAR Model Selection                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 1. TSFM Model (Primary)                                  │    │
│  │    - Zero-shot activity recognition                      │    │
│  │    - Requires: USE_TSFM_MODEL=True                       │    │
│  │    - Requires: TSFM checkpoint exists                    │    │
│  │    - Requires: sentence-transformers cached              │    │
│  │    - Requires: len(imu_data) >= TSFM_MIN_SAMPLES (10)    │    │
│  │    - Returns: label, confidence, "tsfm_model"            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                       │
│                          │ Failed                               │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 2. Legacy IMU Transformer                                │    │
│  │    - 7-class classifier                                   │    │
│  │    - Requires: HAR_IMU_MODEL_CHECKPOINT exists           │    │
│  │    - Returns: label, confidence, "imu_model"             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                       │
│                          │ Failed                               │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 3. Mock Model (Fallback)                                 │    │
│  │    - Random based on acceleration magnitude              │    │
│  │    - For testing only                                    │    │
│  │    - Returns: label, confidence, "mock_har"              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Label Mapping

TSFM model outputs are mapped to MobiBox labels:

| TSFM Labels | MobiBox Label |
|-------------|---------------|
| walking, nordic_walking, walking_* | walking |
| running, jogging, running_* | running |
| sitting, sitting_down, transport_sit | sitting |
| standing, standing_*, transport_stand | standing |
| lying, laying, sleeping | lying |
| ascending_stairs, climbing_stairs, walking_upstairs | climbing stairs |
| *other* | unknown |

---

## Configuration

### Backend Environment Variables

```bash
# Model cache directories
export HF_HOME=/root/.cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface
```

### Backend Config (`src/celery_app/config.py`)

```python
# TSFM Configuration
USE_TSFM_MODEL = True
TSFM_MIN_SAMPLES = 10  # Minimum samples for TSFM inference

# Legacy IMU Model
HAR_IMU_MODEL_CHECKPOINT = "src/celery_app/services/imu_model_utils/ckpts/run_05_06_25_14_16_final_no_cycling_7_class8_25.pth"
HAR_IMU_WINDOW_SIZE = 50  # Samples at 50Hz for 1 second
HAR_IMU_INPUT_CHANNELS = 9  # acc_X/Y/Z, gyro_X/Y/Z, mag_X/Y/Z
```

### TSFM Checkpoint Requirements

```
src/celery_app/services/tsfm_model/ckpts/
├── best.pt                 # Model checkpoint
└── hyperparameters.json    # Model configuration
```

### Sentence Transformers Models Required

```
~/.cache/huggingface/hub/
├── models--sentence-transformers--all-MiniLM-L6-v2/   # 384-dim, token encoding
└── models--sentence-transformers--all-mpnet-base-v2/  # 768-dim, label bank
```

---

## Setup Instructions

### 1. Backend Setup

```bash
# Clone and checkout branch
cd MobiBox_Server
git checkout imu_test
git pull origin imu_test

# Run database migration
# Execute 006_imu_test_results.sql in Supabase SQL editor

# Download sentence-transformers models (on machine with internet)
./scripts/download_models.sh

# Copy models to server
scp -r ~/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2 \
    root@server:~/.cache/huggingface/hub/
scp -r ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2 \
    root@server:~/.cache/huggingface/hub/

# On server, set environment variables
export HF_HOME=/root/.cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface

# Restart backend
systemctl restart mobibox-backend
# or
docker restart <container_name>
```

### 2. Frontend Setup

```bash
# Clone and checkout branch
cd MobiQA-Android
git checkout imu_test
git pull origin imu_test

# Build and install
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 3. Verify Setup

```bash
# Check backend logs for model loading
tail -f /var/log/mobibox/backend.log | grep -i "tsfm\|sentence"

# Should see:
# INFO - TSFM model loaded successfully from .../best.pt
# Loaded all-mpnet-base-v2 (768-dim, frozen)
```

---

## Troubleshooting

### Issue: Predictions always return "unknown"

**Cause:** TSFM model not loaded, falling back to mock model.

**Diagnosis:**
1. Check backend logs:
   ```
   WARNING: TSFM model not available, falling back to legacy model
   ```

2. Check if sentence-transformers models are cached:
   ```bash
   ls -la ~/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/
   ```

3. Verify environment variables:
   ```bash
   echo $HF_HOME
   echo $SENTENCE_TRANSFORMERS_HOME
   ```

**Solution:**
```bash
# Download and copy models (see Setup Instructions)
./scripts/download_models.sh
scp -r ~/.cache/huggingface/hub/models--sentence-transformers--* root@server:~/.cache/huggingface/hub/
```

### Issue: Frontend shows "Mock (Debug)" as model source

**Cause:** TSFM and legacy IMU model both failed to load.

**Solution:**
1. Verify TSFM checkpoint exists: `ls src/celery_app/services/tsfm_model/ckpts/best.pt`
2. Verify legacy checkpoint exists: `ls src/celery_app/services/imu_model_utils/ckpts/*.pth`
3. Check sentence-transformers cache

### Issue: Only ~105 samples collected instead of 150

**Cause:** Original implementation used `Handler.postDelayed()` on main thread.

**Solution:** Already fixed in commit `f8a08bb` using `ScheduledExecutorService`.

### Issue: Timeout errors on prediction

**Cause:** First inference takes longer due to model loading.

**Solution:**
- First request may take 5-10 seconds for model initialization
- Subsequent requests should be fast (~100-500ms)
- Timeout is set to 10 seconds (configurable in `service.py`)

### Issue: Network unreachable for HuggingFace

**Cause:** Server cannot access huggingface.co.

**Solution:**
- Pre-cache models on machine with internet access
- Copy cache to server (see Setup Instructions)

---

## Git Branches

| Repository | Branch | Description |
|------------|--------|-------------|
| MobiBox_Server | `imu_test` | Backend implementation |
| MobiQA-Android | `imu_test` | Frontend implementation |

### Key Commits

**Backend (MobiBox_Server):**
- `3487227` - feat: add IMU test endpoint for TSFM model evaluation
- `b9fcdb2` - fix: remove "unknown" from mock model choices for debugging
- `66cc9fa` - docs: add IMU test feature documentation and improve logging
- `895b1a3` - test: add comprehensive tests for IMU model loading and inference

**Frontend (MobiQA-Android):**
- `ede3b22` - feat: add IMU test activity for model evaluation
- `f8a08bb` - fix: use ScheduledExecutorService for accurate 50Hz IMU sampling
- `58dca4d` - feat: display model source in IMU test results

---

## Testing

### Test File Structure

```
src/test/
└── test_imu_model_loading.py    # Comprehensive tests for IMU model
```

### Test Coverage

The test suite covers:

| Test Class | Description |
|------------|-------------|
| `TestTSFMModelLoading` | TSFM model loading, caching, checkpoint verification |
| `TestLegacyIMUModelLoading` | Legacy IMU model path resolution and loading |
| `TestModelFallbackChain` | HAR model fallback (TSFM → Legacy → Mock) |
| `TestIMUTestEndpoint` | Request validation, label normalization |
| `TestIMUDataConversion` | IMU data to numpy array conversion |
| `TestLabelMapping` | TSFM label to MobiBox label mapping |

### Running Tests

```bash
# Run all IMU model tests
pytest src/test/test_imu_model_loading.py -v

# Run with coverage
pytest src/test/test_imu_model_loading.py --cov=src/celery_app/services --cov=src/imu_test

# Run specific test class
pytest src/test/test_imu_model_loading.py::TestTSFMModelLoading -v
```

### Test Requirements

- Tests use `pytest` and `pytest-asyncio`
- TSFM and legacy model tests skip gracefully if models not available
- Tests are safe to run in CI/CD environments

---

## Future Improvements

1. **Batch Testing:** Allow multiple tests in one session
2. **Export Results:** Download test results as CSV
3. **Confusion Matrix:** Visualize per-class accuracy
4. **Model Comparison:** Compare TSFM vs legacy vs ground truth side-by-side
5. **Real-time Feedback:** Show prediction confidence during collection
6. **Extended Labels:** Support more activity types for fine-grained evaluation