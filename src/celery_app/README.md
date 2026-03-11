# Celery-based Activity Recognition & Intervention System

This module implements a scalable Celery pipeline for processing mobile sensor data, generating atomic activities, and producing health interventions and summary logs.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI APPLICATION                                │
│  /upload/documents ──┐                                                      │
│  /upload/imu ────────┼──→ Insert to DB ──→ Trigger Celery Tasks            │
└─────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CELERY WORKERS                                     │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ process_har_batch   │  │ process_atomic  │  │ Scheduled Tasks (Beat)  │  │
│  │ (on IMU upload)     │  │ (on doc upload) │  │ - process_har_periodic  │  │
│  │                     │  │                 │  │   (every 2 seconds)     │  │
│  │ process_har_periodic│  │ 7 dimensions:  │  │ - hourly_interventions  │  │
│  │ (every 2 seconds)   │  │ - HAR (LLM)     │  │ - hourly_summary        │  │
│  │                     │  │ - APP (LLM)     │  │ - daily_summary         │  │
│  │ IMU Transformer or  │  │ - Steps         │  │                         │  │
│  │ Mock HAR Model      │  │ - Phone         │  │ Compress → Generate     │  │
│  │                     │  │ - Social        │  │                         │  │
│  │                     │  │ - Movement      │  │                         │  │
│  │                     │  │ - Location(LLM) │  │                         │  │
│  └──────────┬──────────┘  └────────┬────────┘  └────────────┬────────────┘  │
│             │                      │                          │               │
│             ▼                      ▼                          ▼               │
│        har table           atomic_activities table    interventions/summary   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation & Setup

### Prerequisites
- RabbitMQ running on localhost:5672
- Celery and celery-beat packages installed
- Supabase database configured

### Start Services

```bash
# Start RabbitMQ
docker run -d -p 5672:5672 rabbitmq

# Start Celery worker with beat scheduler (recommended)
celery -A src.celery_app.celery_app worker --beat \
    -Q default,har,atomic,summary \
    --loglevel=INFO

# OR run worker and beat separately
# Terminal 1: Worker
celery -A src.celery_app.celery_app worker \
    -Q default,har,atomic,summary \
    --loglevel=INFO

# Terminal 2: Beat scheduler
celery -A src.celery_app.celery_app beat --loglevel=INFO

# Start FastAPI
uvicorn src.main:app --reload
```

> **Note:** The worker must listen to all queues (`default,har,atomic,summary`) because tasks are routed to specific queues based on their type.

## Configuration

### Environment Variables

Add to your `.env` file:

```env
# RabbitMQ
RABBITMQ_URL=amqp://guest:guest@localhost:5672//

# Celery
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672//
CELERY_RESULT_BACKEND=rpc://
```

### Config Options (src/celery_app/config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `HAR_TASK_RATE_LIMIT` | 30/m | Rate limit for HAR tasks |
| `ATOMIC_TASK_RATE_LIMIT` | 10/m | Rate limit for atomic activity tasks |
| `HAR_IMU_WINDOW_SECONDS` | 10 | IMU data window for HAR (seconds) |
| `HAR_DATA_DELAY_SECONDS` | 126 | Delay before fetching IMU data (accounts for batch upload) |
| `HAR_IMU_WINDOW_SIZE` | 50 | Samples per window (2s @ 25Hz, must match model) |
| `HAR_IMU_INPUT_CHANNELS` | 9 | Number of IMU channels (acc/gyro/mag × X/Y/Z) |
| `HAR_DEBOUNCE_SECONDS` | 2 | Minimum time between HAR processing per user |
| `ATOMIC_DEBOUNCE_SECONDS` | 5 | Minimum time between atomic processing per user |
| `ATOMIC_*_WINDOW_SECONDS` | 2-120 | Data windows for each atomic dimension |

### HAR Model Configuration

| Setting | Description |
|---------|-------------|
| `HAR_IMU_MODEL_CHECKPOINT` | Path to IMU Transformer checkpoint (.pth file) |
| `HAR_IMU_MODEL_CONFIG` | Model architecture config (must match checkpoint) |

Set `HAR_IMU_MODEL_CHECKPOINT` to `None` to use the mock HAR model instead.

## Task Pipelines

### 1. HAR Processing Pipeline

**Triggers:**
1. IMU data upload via `/upload/imu` → `process_har_batch`
2. Periodic task every 2 seconds → `process_har_periodic`

```
IMU Upload / Periodic (2s)
    │
    ▼
process_har_batch / process_har_periodic
    │
    ├── Check debounce (2s minimum between runs per user)
    │
    ├── Fetch IMU data from delayed window
    │   └── Window: (now - 126s - 10s) to (now - 126s)
    │   └── Delay accounts for batch IMU upload every 2 minutes
    │
    ├── Run HAR Model
    │   ├── If checkpoint exists: IMU Transformer (7 classes)
    │   └── Otherwise: Mock HAR model (heuristic-based)
    │
    └── Insert result to har table
```

**Why the 126-second delay?**
- IMU data is uploaded in batches every 2 minutes from the mobile client
- The 126s delay (2 min - 6s buffer) ensures we query data that has already been inserted
- Without this delay, the HAR task would find no data and skip processing

**HAR Labels (7 classes):**

| Index | Label |
|-------|-------|
| 0 | unknown |
| 1 | standing |
| 2 | sitting |
| 3 | lying |
| 4 | walking |
| 5 | climbing stairs |
| 6 | running |

**Tasks:**

| Task | Name | Queue | Description |
|------|------|-------|-------------|
| `process_har_batch` | `process_har_batch` | har | Process HAR for a list of users (triggered by upload) |
| `process_har_single` | `process_har_single` | har | Process HAR for a single user |
| `process_har_periodic` | `process_har_periodic` | har | Periodic HAR for all active users (every 2s) |

### 2. Atomic Activities Pipeline

**Trigger:** Document data upload via `/upload/documents`

```
Document Upload → process_atomic_activities_batch
               → Fetch data from last X seconds (per dimension)
               → Generate 7 labels
               → Insert to atomic_activities table
```

**Task:** `process_atomic_activities_batch(user_list: List[str])`

Generates 7 dimensions of atomic activity labels:

| Dimension | Source | Method | Window |
|-----------|--------|--------|--------|
| `har_label` | HAR table | LLM | 2s |
| `app_category` | current_app | LLM | 10s |
| `step_label` | stepcount_sensor | if-else | 10s |
| `phone_usage` | screen_on_ratio | if-else | 10s |
| `social_label` | bluetooth, apps | if-else | 10s |
| `movement_label` | GPS coordinates | if-else | 120s |
| `location_label` | GPS, address, POI | LLM | 120s |

### 3. Interventions & Summaries Pipeline

**Trigger:** Celery Beat (scheduled)

```
Celery Beat (hourly/daily) → compress_atomic_activities
                           → generate_interventions
                           → generate_summary_logs
```

**Scheduled Tasks:**

| Task | Name | Schedule | Queue |
|------|------|----------|-------|
| `process_har_periodic` | `process_har_periodic` | Every 2 seconds | har |
| `generate_hourly_interventions` | `generate_hourly_interventions` | Every hour at minute 5 | summary |
| `generate_hourly_summary` | `generate_hourly_summary` | Every hour at minute 0 | summary |
| `generate_daily_summary` | `generate_daily_summary` | Daily at midnight | summary |

## Database Tables

### har
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| user | text | User identifier |
| har_label | text | Activity label (enum) |
| timestamp | timestamptz | Record timestamp |

### atomic_activities
| Column | Type | Description |
|--------|------|-------------|
| user | text | User identifier |
| timestamp | timestamptz | Record timestamp |
| har_label | text | HAR activity label |
| app_category | text | App usage category |
| step_label | text | Step activity label |
| phone_usage | text | Phone usage pattern |
| social_label | text | Social context label |
| movement_label | text | Movement pattern label |
| location_label | text | Location context |

### interventions
| Column | Type | Description |
|--------|------|-------------|
| user | text | User identifier |
| intervention_type | text | Type of intervention |
| message | text | Intervention message |
| priority | text | Priority (low/medium/high) |
| category | text | Category (physical/mental/social/digital_wellbeing) |
| timestamp | timestamptz | Record timestamp |

### summary_logs
| Column | Type | Description |
|--------|------|-------------|
| user | text | User identifier |
| log_type | text | hourly or daily |
| title | text | Summary title |
| summary | text | Summary narrative |
| highlights | jsonb | Key highlights |
| recommendations | jsonb | Recommendations |
| timestamp | timestamptz | Record timestamp |

## Usage Examples

### Trigger HAR Processing Manually

```python
from src.celery_app.tasks.har_tasks import process_har_batch, process_har_single

# Process HAR for a list of users
result = process_har_batch.delay(["user1", "user2", "user3"])

# Process HAR for a single user
result = process_har_single.delay("user1")
```

### Trigger Atomic Activities Manually

```python
from src.celery_app.tasks.atomic_tasks import process_atomic_activities_batch

# Generate atomic activities for a user
result = process_atomic_activities_batch.delay(["user1"])
```

### Manual Intervention Generation

```python
from src.celery_app.tasks.summary_tasks import trigger_intervention_for_user

# Generate intervention for a specific user
result = trigger_intervention_for_user.delay("user1")
```

### Manual Summary Generation

```python
from src.celery_app.tasks.summary_tasks import trigger_summary_for_user

# Generate hourly summary
result = trigger_summary_for_user.delay("user1", hours=1)

# Generate daily summary
result = trigger_summary_for_user.delay("user1", hours=24)
```

## Queue Routing

Tasks are automatically routed to specific queues:

| Queue | Tasks |
|-------|-------|
| `default` | Default queue for unmapped tasks |
| `har` | All HAR tasks (`process_har_batch`, `process_har_periodic`, etc.) |
| `atomic` | All atomic activity tasks |
| `summary` | All summary and intervention tasks |

**Important:** Workers must listen to all queues to process all task types:
```bash
celery -A src.celery_app.celery_app worker -Q default,har,atomic,summary
```

## Debugging

### Enable Debug Logging

Debug logging is enabled for HAR service by default. To see detailed IMU query information:

```bash
celery -A src.celery_app.celery_app worker --loglevel=INFO
```

You will see logs like:
```
[DEBUG] Fetching IMU window for user1: 2026-03-04T15:53:06+08:00 to 2026-03-04T15:53:16+08:00 (delay=126s, window=10s)
[DEBUG] Found 25 IMU records for user1
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| HAR always skipped | No IMU data in window | Check `HAR_DATA_DELAY_SECONDS` matches upload frequency |
| Periodic task not running | Task name mismatch | Ensure beat schedule uses correct task name |
| Tasks not processed | Worker not listening to queue | Add `-Q default,har,atomic,summary` to worker command |
| Beat scheduler not triggering | Beat not running | Start beat with `--beat` flag or separate `celery beat` process |

## Scalability Considerations

1. **Rate Limiting:** Tasks use `rate_limit` to prevent overload
2. **Batching:** Multiple users processed in single task
3. **Debouncing:** In-memory tracking prevents redundant processing per worker
4. **Worker Concurrency:** Configurable via `--concurrency` flag
5. **Queue Separation:** Tasks can be routed to different queues (har, atomic, summary)

For production with multiple workers, consider:
- Using Redis instead of in-memory for debounce tracking
- Implementing distributed locks for concurrency control
- Adding monitoring (Celery flower)

## Testing

```bash
# Run unit tests
pytest src/test/ -v

# Test specific task
celery -A src.celery_app.celery_app call process_har_batch --args='["test_user"]'

# Check task status
celery -A src.celery_app.celery_app inspect active

# List registered tasks
celery -A src.celery_app.celery_app inspect registered
```