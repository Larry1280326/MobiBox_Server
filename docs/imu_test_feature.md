# IMU Test Feature Documentation

## Overview

The IMU Test feature allows users to collect IMU sensor data on an Android device, send it to the backend for activity prediction using the TSFM model, and evaluate model accuracy by comparing predictions against ground truth labels.

## Branches

- **Backend**: `imu_test` (MobiBox_Server)
- **Frontend**: `imu_test` (MobiQA-Android)

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│   Android App       │         │    Backend API      │
│   ImuTestActivity   │────────▶│   /imu_test/predict │
│                     │         │                     │
│  - Select activity  │         │  - Run TSFM model   │
│  - Collect IMU 50Hz │         │  - Compare labels   │
│  - Send to server   │◀────────│  - Store results    │
│  - Display result   │         │                     │
└─────────────────────┘         └─────────────────────┘
                                          │
                                          ▼
                                  ┌───────────────────┐
                                  │  Supabase DB      │
                                  │  imu_test_results │
                                  └───────────────────┘
```

## Backend Changes

### New Files

| File | Description |
|------|-------------|
| `src/imu_test/__init__.py` | Module initialization |
| `src/imu_test/router.py` | API endpoints (`/imu_test/predict`, `/imu_test/statistics`, `/imu_test/labels`) |
| `src/imu_test/schemas.py` | Request/Response schemas |
| `src/imu_test/service.py` | Business logic with 10-second inference timeout |
| `migrations/006_imu_test_results.sql` | Database migration for test results table |

### API Endpoints

#### POST `/imu_test/predict`

**Request:**
```json
{
  "user": "user_id",
  "ground_truth_label": "walking",  // optional
  "imu_data": [
    {
      "timestamp": "2024-01-01T00:00:00.000Z",
      "acc_X": 0.1, "acc_Y": 9.8, "acc_Z": 0.0,
      "gyro_X": 0.0, "gyro_Y": 0.0, "gyro_Z": 0.0,
      "mag_X": 0.0, "mag_Y": 0.0, "mag_Z": 0.0
    }
    // ... minimum 50 samples
  ]
}
```

**Response:**
```json
{
  "user": "user_id",
  "predicted_label": "walking",
  "confidence": 0.85,
  "source": "tsfm_model",
  "ground_truth_label": "walking",
  "is_correct": true,
  "sample_count": 150,
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

#### GET `/imu_test/statistics?user=<user_id>`

Returns accuracy statistics for all test results.

#### GET `/imu_test/labels`

Returns valid activity labels: `walking`, `running`, `sitting`, `standing`, `lying`, `climbing stairs`, `unknown`

### Database Schema

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
```

## Frontend Changes

### New Files

| File | Description |
|------|-------------|
| `ImuTestActivity.java` | Activity for IMU testing |
| `res/layout/activity_imu_test.xml` | Layout for IMU test screen |

### Modified Files

| File | Changes |
|------|---------|
| `MainActivity.java` | Added button to launch ImuTestActivity |
| `res/layout/activity_main.xml` | Added "IMU Model Test" button |
| `Constants.java` | Added `ENDPOINT_IMU_TEST_PREDICT` endpoint |
| `HttpApiClient.java` | Added `getHttpClient()` method |
| `AndroidManifest.xml` | Registered ImuTestActivity |

### Data Collection

- **Sampling Rate**: 50Hz (20ms intervals)
- **Collection Duration**: 3 seconds (150 samples expected)
- **Minimum Samples Required**: 50

### Activity Labels

| Label | Display Name |
|-------|--------------|
| `walking` | Walking |
| `running` | Running |
| `sitting` | Sitting |
| `standing` | Standing |
| `lying` | Lying |
| `climbing stairs` | Climbing Stairs |

## Configuration

### Backend Environment

Ensure sentence-transformers models are cached:
```bash
export HF_HOME=/root/.cache/huggingface
export SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface
```

### Model Requirements

The TSFM model requires:
- `all-MiniLM-L6-v2` (384-dim, for token encoding)
- `all-mpnet-base-v2` (768-dim, for label bank)

Download with:
```bash
./scripts/download_models.sh
```

## Testing

1. Open app → tap "IMU Model Test"
2. Select ground truth activity
3. Tap "Start Collection" → wait 3 seconds
4. Tap "Send to Server"
5. View prediction result and accuracy comparison

## Troubleshooting

### "unknown" Prediction

If predictions always return "unknown", check:
1. TSFM model checkpoint exists at `src/celery_app/services/tsfm_model/ckpts/best.pt`
2. `hyperparameters.json` exists alongside checkpoint
3. Sentence-transformers models are cached (check HF_HOME/SENTENCE_TRANSFORMERS_HOME)
4. Check backend logs for model loading errors

### Timeout Errors

The inference has a 10-second timeout. If exceeded:
- Check model loading time (first request may be slow)
- Check sentence-transformers model cache

### Low Sample Count

If getting fewer than expected samples (e.g., 105 instead of 150):
- Fixed in commit `f8a08bb` using `ScheduledExecutorService` for accurate timing