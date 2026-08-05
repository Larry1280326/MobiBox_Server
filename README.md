# MobiBox Backend

A FastAPI-based backend for the MobiBox behavioral monitoring study (HKUST).
Processes IMU sensor data with an ML activity recognition model, generates
LLM-powered health interventions and summaries, and serves them to the Android
companion app via a REST API.

## Architecture

```
Android App ──HTTP──▶ FastAPI (port 8001) ──▶ MongoDB
                           │
                    ┌──────┴──────┐
                    ▼             ▼
               RabbitMQ      Celery Worker
               (queue)       ├─ HAR (SelfSupEncoder)
                             ├─ Atomic Activities
                             ├─ Summaries (LLM)
                             └─ Interventions (LLM)
                              ▲
                       Celery Beat (scheduler)
```

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate Mobibox_backend

# 2. Configure
cp .env.example .env
# Edit .env with your MongoDB URL and OpenRouter API key

# 3. Start services (recommended: tmux)
./scripts/tmux_start.sh

# 4. Verify
curl http://localhost:8001/health
```

**Manual start** (if not using scripts):

```bash
# Infrastructure
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# API server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8001

# Celery worker
celery -A src.celery_app.celery_app worker --loglevel=info -Q default,har,atomic,summary

# Celery beat (scheduler)
celery -A src.celery_app.celery_app beat --loglevel=info
```

---

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| Python 3.11 + [Conda](https://docs.conda.io/en/latest/miniconda.html) | Runtime |
| [MongoDB](https://www.mongodb.com/) | Database (local or Atlas) |
| [Docker](https://www.docker.com/) | RabbitMQ message broker |
| [OpenRouter](https://openrouter.ai/) API key | LLM-powered interventions & summaries |
| tmux (optional) | Process management on servers |

---

## Configuration

Copy `.env.example` to `.env` and configure:

### MongoDB

```env
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=mobibox
```

### LLM (OpenRouter)

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
DEFAULT_TEMPERATURE=0.1
```

The current default is `google/gemma-4-26b-a4b-it:free` — a free model with
native structured-output support, suitable for generating health interventions
and activity summaries. Other free options:
`meta-llama/llama-3.3-70b-instruct:free`, `mistralai/mistral-7b-instruct:free`.

### RabbitMQ / Celery

```env
RABBITMQ_URL=amqp://guest:guest@localhost:5672//
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672//
CELERY_RESULT_BACKEND=rpc://
```

### Baidu Maps (optional)

```env
BAIDU_MAPS_API_KEY=your-key
BAIDU_MAPS_ENABLED=true
```

---

## API Endpoints

### Health
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — pings MongoDB |
| `GET` | `/mongodb-test` | Direct MongoDB connection test |

### Registration
| Method | Path | Body |
|--------|------|------|
| `POST` | `/register` | `{"name": "username"}` |

User IDs are **strings**, not integers. Applies to all endpoints below.

### Data Upload
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload/documents` | Bulk upload sensor/survey data (17 fields: volume, screen_on_ratio, wifi, gps, battery, current_app, bluetooth, etc.) |
| `POST` | `/upload/imu` | Bulk upload IMU data (9 channels: acc XYZ + gyro XYZ + mag XYZ) |

Uploads trigger Celery HAR and atomic-activity processing.

### Query
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/get_summary_log` | Latest hourly/daily summary with polling support |
| `POST` | `/get_intervention` | Latest health intervention |
| `POST` | `/get_compressed_atomic_activities` | Grouped activity labels by dimension |
| `POST` | `/get_encoded_atomic_activities` | Level-1 (temporal) + Level-2 (aggregated) encoded activities |
| `POST` | `/send_intervention_feedback` | Submit feedback on an intervention (6 MC + text) |
| `POST` | `/send_log_feedback` | Submit feedback on a summary log |

### IMU Test
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/imu_test/predict` | Predict activity from IMU data (min 50 samples) |
| `GET` | `/imu_test/statistics` | Accuracy statistics per label |
| `GET` | `/imu_test/labels` | Valid activity labels |

Full API reference: `http://localhost:8001/docs` (Swagger UI).

---

## HAR Model — IMU-SelfSupEncoder-v1

Human Activity Recognition is powered by
**[IMU-SelfSupEncoder-v1](https://huggingface.co/NikoKKK/IMU-SelfSupEncoder-v1)**,
a lightweight (1.6M params) self-supervised Vision Transformer trained on the
WISDM dataset. It classifies 6 IMU channels (acc + gyro) into 7 activity labels.

| Property | Value |
|----------|-------|
| Model | `NikoKKK/IMU-SelfSupEncoder-v1` (HuggingFace) |
| Architecture | ViT-style Conv-stem + time-frequency fusion |
| Parameters | 1.6 M |
| Input | 6-channel (acc XYZ + gyro XYZ), 200 timesteps @ 20 Hz |
| Output | 192-dim CLS embedding → prototype-based classifier |
| Classes | walking, running, sitting, standing, lying, climbing stairs, unknown |
| Inference | ~3 ms on CPU (~340 inferences/sec) |
| Size on disk | ~5 MB |

### Fallback chain

```
SelfSupEncoder (prototype classifier)
        │
        ▼ (if model unavailable or < 50 samples)
Mock model (acceleration-magnitude heuristic)
```

### Offline deployment (China servers)

The model downloads from HuggingFace on first use. For servers without internet:

```bash
# On a machine WITH internet:
python scripts/download_selfsup_model.py
# → downloads to models/imu_selfsup/ (~5 MB)

# Copy to server:
scp -r models/imu_selfsup user@server:~/MobiBox_Server/models/imu_selfsup
```

The service auto-detects `models/imu_selfsup/` and loads locally — no network needed.

### Testing the model

```bash
python scripts/test_selfsup_model.py          # full evaluation
python scripts/test_selfsup_model.py --quick  # download + basic checks
```

---

## Data Pipeline

```
IMU Upload                 Document Upload
    │                            │
    ▼                            ▼
[imu collection]          [uploads collection]
    │                            │
    ▼ (Celery: HAR)              ▼ (Celery: Atomic)
[har collection]          [atomic_activities collection]
                               │
                               ▼ (Celery: 20 min + daily)
                          LLM Summary Generation
                               │
                               ▼
                          [summary_logs collection]
                               │
                               ▼ (Celery: 20 min)
                          LLM Intervention Generation
                               │
                               ▼
                          [interventions collection]
                               │
                               ▼
                      Android polls /get_summary_log
                           and /get_intervention
```

---

## Database (MongoDB)

### Collections

| Collection | TTL | Purpose |
|-----------|-----|---------|
| `users` | — | User registration |
| `uploads` | 30 days | Sensor/survey data |
| `imu` | 7 days | Raw IMU readings |
| `har` | 30 days | Activity labels + confidence |
| `atomic_activities` | 30 days | 7-dimension activity labels |
| `summary_logs` | 90 days | LLM-generated summaries |
| `interventions` | 90 days | LLM-generated health interventions |
| `intervention_feedbacks` | — | User feedback on interventions |
| `summary_log_feedbacks` | — | User feedback on summaries |
| `app_categories` | — | App-name → category cache |
| `user_processing_state` | — | Per-user processing timestamps |
| `imu_test_results` | — | IMU prediction test results |

Indexes (including TTL expiry) are created automatically at startup by `database_indexes.py`.

### Key document shapes

**imu**: `user`, `timestamp`, `acc_X/Y/Z`, `gyro_X/Y/Z`, `mag_X/Y/Z`
**har**: `user`, `timestamp`, `har_label`, `confidence`, `source` (`selfsup_model` \| `mock_har` \| `selfsup_insufficient`)
**atomic_activities**: `user`, `timestamp`, `har_label`, `app_category`, `app_name`, `step_count`, `phone_usage`, `social`, `movement`, `location`
**interventions**: `user`, `timestamp`, `intervention_content`, `start_timestamp`, `end_timestamp`
**summary_logs**: `user`, `log_type`, `summary`, `start_timestamp`, `end_timestamp`, `timestamp`

---

## Celery Tasks

| Pipeline | Trigger | Schedule |
|----------|---------|----------|
| HAR Processing | IMU upload | Every 2 s (periodic) |
| Atomic Activities | Document upload | Every 10 s (periodic) |
| Hourly Summary | Beat scheduler | Every 20 min |
| Hourly Intervention | Beat scheduler | Every 20 min |
| Daily Summary | Beat scheduler | Midnight |
```bash
# Start worker
celery -A src.celery_app.celery_app worker --loglevel=info -Q default,har,atomic,summary

# Start scheduler
celery -A src.celery_app.celery_app beat --loglevel=info

# Manual trigger
celery -A src.celery_app.celery_app call process_har_batch --args='["user1"]'
```

---

## LLM Integration

Summaries and interventions are generated via OpenRouter's OpenAI-compatible API
using `langchain-openai`. The `src/llm_utils/services.py` module provides:

| Function | Use |
|----------|-----|
| `query_llm()` | Simple text generation |
| `generate_structured_output()` | Pydantic schema-constrained JSON output |
| `summarize_long_text()` | Chunked summarization |
| `get_llm()` | Configured `ChatOpenAI` instance |

Built-in rate limiting: 60 requests/minute.

---

## Service Management

### tmux (recommended for servers)

```bash
./scripts/tmux_start.sh      # Start all services in tmux session 'mobibox'
./scripts/tmux_attach.sh     # Attach to view live output
./scripts/tmux_status.sh     # Check health of all services
./scripts/tmux_stop.sh       # Stop everything
```

Session layout: `api` | `worker` | `beat` | `logs` (3-pane live monitor)

### nohup (alternative)

```bash
./scripts/start_services.sh
./scripts/status.sh
./scripts/stop_services.sh
./scripts/restart_services.sh
```

### Logs

Logs are written to `logs/` with Python `RotatingFileHandler` (10 MB × 5 backups):

| File | Service |
|------|---------|
| `logs/api.log` | FastAPI |
| `logs/celery_worker.log` | Celery worker |
| `logs/celery_beat.log` | Celery beat |

---

## Testing

```bash
# All unit tests (~65 tests, <1 s)
pytest src/test/ -v \
  --ignore=src/test/test_llm_integration.py \
  --ignore=src/test/test_intervention_pipeline_integration.py

# LLM integration tests (requires OPENROUTER_API_KEY)
pytest src/test/test_llm_integration.py -v -m integration
```

### Test files

| File | Description |
|------|-------------|
| `test_upload.py` | Upload API endpoints |
| `test_query.py` | Query API endpoints |
| `test_celery_services.py` | HAR, atomic, summary, intervention services |
| `test_celery_tasks.py` | Celery task definitions |
| `test_llm_utils.py` | LLM utilities (mocked) |
| `test_llm_integration.py` | LLM integration (real API) |
| `test_intervention_pipeline_integration.py` | Intervention pipeline |

### Fixtures (`conftest.py`)

| Fixture | Description |
|---------|-------------|
| `mongodb_mock` | Mock MongoDB with auto-creating collections |
| `client` | FastAPI `TestClient` with mocked MongoDB + Celery |
| `mock_get_database` | Patch `get_database()` for Celery service tests |

---

## Project Structure

```
MobiBox_Server/
├── environment.yml
├── .env.example
├── pyproject.toml
├── README.md
├── scripts/
│   ├── tmux_start.sh / tmux_stop.sh / tmux_attach.sh / tmux_status.sh
│   ├── start_services.sh / stop_services.sh / restart_services.sh / status.sh
│   ├── download_selfsup_model.py
│   └── test_selfsup_model.py
├── models/                     # Offline model cache (gitignored)
├── logs/                       # Rotating log files
├── docs/
│   └── mongodb-access.md
├── src/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # App settings (from .env)
│   ├── database.py             # MongoDB (Motor async + PyMongo sync)
│   ├── database_indexes.py     # Auto-created indexes + TTL
│   ├── logging_config.py       # Rotating file handlers
│   ├── register/               # POST /register
│   ├── upload/                 # POST /upload/*
│   ├── query/                  # POST /get_summary_log, /get_intervention, etc.
│   ├── imu_test/               # POST /imu_test/predict, GET /statistics, /labels
│   ├── llm_utils/              # OpenRouter LLM integration
│   ├── services/               # Baidu Maps API client
│   ├── celery_app/
│   │   ├── celery_app.py       # Celery instance + beat schedule
│   │   ├── config.py           # Model config, windows, thresholds
│   │   ├── tasks/              # har_tasks, atomic_tasks, summary_tasks
│   │   ├── services/           # har_service, atomic_service, summary_service,
│   │   │                       # intervention_service, imu_selfsup_service,
│   │   │                       # app_category_service, processing_state_service
│   │   └── schemas/            # HAR + atomic activity Pydantic schemas
│   └── test/                   # pytest test suite
└── .gitignore
```

---

## Feature Details

### App Category Lookup
`src/celery_app/services/app_category_service.py` — 70+ predefined apps in
memory + DB cache (`app_categories`). LLM fallback for unknown apps.

### Incremental Processing
`src/celery_app/services/processing_state_service.py` — per-user timestamps
in `user_processing_state` collection prevent re-processing the same data.

### Summary Gating
Summaries are generated only when thresholds are met (≥60 atomic records,
≥3 unique activity types, ≥1 hour of data).

### Mobile Polling
`POST /get_summary_log` supports `last_log_id` polling — returns
`has_new_log: false` when the latest log matches, avoiding redundant transfers.

### Baidu Maps Integration
`src/services/baidu_maps.py` — reverse geocoding with 1-hour cache.
Falls back to provided address/POI data when API is unavailable.
