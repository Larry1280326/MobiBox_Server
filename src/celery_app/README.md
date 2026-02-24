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
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ process_har     │    │ process_atomic  │    │ Scheduled Tasks (Beat)  │  │
│  │ (on IMU upload)│    │ (on doc upload)│    │ - hourly_interventions │  │
│  │                 │    │                 │    │ - hourly_summary       │  │
│  │ Runs mock HAR   │    │ 7 dimensions:  │    │ - daily_summary        │  │
│  │ every 2s window │    │ - HAR (LLM)     │    │                         │  │
│  │                 │    │ - APP (LLM)     │    │ Compress → Generate    │  │
│  │                 │    │ - Steps         │    │                         │  │
│  │                 │    │ - Phone         │    │                         │  │
│  │                 │    │ - Social        │    │                         │  │
│  │                 │    │ - Movement      │    │                         │  │
│  │                 │    │ - Location(LLM) │    │                         │  │
│  └────────┬────────┘    └────────┬────────┘    └────────────┬────────────┘  │
│           │                      │                          │               │
│           ▼                      ▼                          ▼               │
│      har table           atomic_activities table    interventions/summary   │
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

# Start Celery worker
celery -A src.celery_app.celery_app worker --loglevel=info

# Start Celery beat (for scheduled tasks)
celery -A src.celery_app.celery_app beat --loglevel=info

# Start FastAPI
uvicorn src.main:app --reload
```

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
| `HAR_IMU_WINDOW_SECONDS` | 2 | IMU data window for HAR |
| `ATOMIC_*_WINDOW_SECONDS` | 10-120 | Data windows for each atomic dimension |
| `HAR_DEBOUNCE_SECONDS` | 2 | Minimum time between HAR processing |
| `ATOMIC_DEBOUNCE_SECONDS` | 5 | Minimum time between atomic processing |

## Task Pipelines

### 1. HAR Processing Pipeline

**Trigger:** IMU data upload via `/upload/imu`

```
IMU Upload → process_har_batch (debounced, 2s window)
          → Mock HAR Model
          → Insert to har table
```

**Task:** `process_har_batch(user_list: List[str])`

- Fetches IMU data for each user from the last 2 seconds
- Runs mock HAR model to classify activity
- Inserts HAR label to database

**Mock HAR Labels:**
- walking, running, sitting, standing, lying_down
- climbing_stairs, descending_stairs, cycling, driving, unknown

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

| Task | Schedule | Description |
|------|----------|-------------|
| `generate_hourly_interventions` | Every hour at minute 0 | Generate health interventions |
| `generate_hourly_summary` | Every hour at minute 0 | Generate hourly activity summary |
| `generate_daily_summary` | Daily at midnight | Generate daily activity summary |

## Database Tables

### har
| Column | Type | Description |
|--------|------|-------------|
| user | text | User identifier |
| label | text | Activity label |
| confidence | float | Confidence score (0-1) |
| source | text | Source of label (mock_har) |
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
from src.celery_app.tasks.har_tasks import process_har_batch

# Process HAR for a list of users
result = process_har_batch.delay(["user1", "user2", "user3"])
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
```
