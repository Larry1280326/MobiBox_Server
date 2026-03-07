# MobiBox Backend

A FastAPI-based backend server for MobiBox with Supabase integration, Celery task processing, and LLM-powered health interventions.

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate Mobibox_backend
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials (see Configuration section below)
```

### 3. Start All Services (Recommended)

Use the provided startup scripts to manage all services:

```bash
# Start all services (RabbitMQ, FastAPI, Celery worker, Celery beat)
./scripts/start_services.sh

# Check service status
./scripts/status.sh

# Stop all services
./scripts/stop_services.sh

# Restart all services
./scripts/restart_services.sh
```

### 4. Verify Services

```bash
# Check API health
curl http://localhost:8000/health

# Check all service statuses
./scripts/status.sh
```

### Manual Start (Alternative)

If you prefer to start services manually:

```bash
# Start RabbitMQ (required for Celery)
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# Terminal 1: FastAPI Server
conda activate Mobibox_backend
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Celery Worker (for async tasks)
conda activate Mobibox_backend
celery -A src.celery_app.celery_app worker --loglevel=info

# Terminal 3: Celery Beat (for scheduled tasks - optional)
conda activate Mobibox_backend
celery -A src.celery_app.celery_app beat --loglevel=info
```

---

## Environment Setup

### Prerequisites

| Requirement | Purpose |
|-------------|---------|
| [Conda](https://docs.conda.io/en/latest/miniconda.html) | Python environment management |
| Python 3.11 | Runtime environment |
| [Docker](https://www.docker.com/) | Running RabbitMQ |
| [Supabase](https://supabase.com/) | Database backend |
| [OpenRouter](https://openrouter.ai/) API | LLM integration for interventions |

### Configuration

Copy `.env.example` to `.env` and configure the following:

```bash
cp .env.example .env
```

#### Supabase Configuration

1. Create a Supabase project at https://supabase.com/
2. Get your project URL, anon key, and service role key from **Project Settings → API**
3. Update `.env`:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here  # Optional, for admin operations
```

#### OpenRouter LLM Configuration (for LLM features)

1. Get your OpenRouter API key from [OpenRouter Keys](https://openrouter.ai/keys)
2. Update `.env`:

```env
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen3-vl-30b-a3b-thinking
DEFAULT_TEMPERATURE=0.1
```

**Free Models Available:**
- `qwen/qwen3-vl-30b-a3b-thinking` (recommended)
- `meta-llama/llama-3.2-3b-instruct:free`
- `google/gemma-2-9b-it:free`
- `mistralai/mistral-7b-instruct:free`

See [OpenRouter Models](https://openrouter.ai/models) for all available models.

#### RabbitMQ / Celery Configuration

Default configuration works for local development:

```env
RABBITMQ_URL=amqp://guest:guest@localhost:5672//
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672//
CELERY_RESULT_BACKEND=rpc://
```

#### Baidu Maps Configuration (Optional)

For GPS-to-location reverse geocoding:

```env
BAIDU_MAPS_API_KEY=your-baidu-api-key
BAIDU_MAPS_ENABLED=true
```

If not configured, the system falls back to using provided address/POI data.

### Create and Activate Environment

#### Option 1: Create from YAML file (Recommended)

```bash
# Create the conda environment from the YAML file
conda env create -f environment.yml

# Activate the environment
conda activate Mobibox_backend
```

#### Option 2: Create manually

```bash
# Create a new conda environment with Python 3.11
conda create -n Mobibox_backend python=3.11

# Activate the environment
conda activate Mobibox_backend

# Install dependencies
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings sqlalchemy alembic asyncpg psycopg2-binary aiomysql aiosqlite supabase "python-jose[cryptography]" "passlib[bcrypt]" python-multipart httpx aiohttp pytest pytest-asyncio python-dotenv pyyaml orjson black isort flake8 mypy
```

### Verify Installation

```bash
# Activate the environment
conda activate Mobibox_backend

# Check Python version
python --version

# Verify FastAPI is installed
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

# Test Supabase connection
python -c "from src.config import get_settings; print('Settings loaded:', get_settings().app_name)"
```

### Deactivate Environment

```bash
conda deactivate
```

### Export Environment (for sharing)

If you add new packages and want to update the YAML file:

```bash
# Export the current environment to a YAML file
conda env export > environment.yml

# Or export only the packages you explicitly installed (no build info)
pip freeze > requirements.txt
```

### Remove Environment

```bash
# Remove the environment completely
conda env remove -n Mobibox_backend
```

## Running the Server

### Service Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI       │     │   RabbitMQ      │     │   Supabase      │
│   (Port 8000)   │────▶│   (Port 5672)   │     │   (Cloud)       │
│                 │     │                 │     │                 │
│  REST API       │     │  Message Queue  │     │  Database       │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │  Celery Worker  │──────────────┘
         │              │                 │
         │              │  - HAR Tasks    │
         │              │  - Atomic Tasks │
         │              │  - Summaries    │
         │              └─────────────────┘
         │                       ▲
         │              ┌─────────────────┐
         └──────────────│  Celery Beat    │
                        │  (Scheduler)    │
                        │                 │
                        │  - Hourly tasks │
                        │  - Daily tasks  │
                        └─────────────────┘
```

### Starting Services

#### Step 1: Start RabbitMQ (Required)

```bash
# Start RabbitMQ container
docker run -d --name rabbitmq -p 5672:5672 rabbitmq

# To stop RabbitMQ
docker stop rabbitmq

# To restart RabbitMQ
docker start rabbitmq

# To remove RabbitMQ container
docker rm -f rabbitmq
```

#### Step 2: Start FastAPI Server

```bash
conda activate Mobibox_backend

# Development mode with auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at http://localhost:8000

#### Step 3: Start Celery Worker (Required for async tasks)

```bash
conda activate Mobibox_backend

# Start worker with default concurrency
celery -A src.celery_app.celery_app worker --loglevel=info

# Start with specific concurrency
celery -A src.celery_app.celery_app worker --loglevel=info --concurrency=4

# Start with specific queues
celery -A src.celery_app.celery_app worker --loglevel=info -Q har,atomic,summary
```

#### Step 4: Start Celery Beat (Optional - for scheduled tasks)

```bash
conda activate Mobibox_backend

# Start beat scheduler
celery -A src.celery_app.celery_app beat --loglevel=info
```

**Scheduled Tasks:**
| Task | Schedule | Description |
|------|----------|-------------|
| `generate_hourly_interventions` | Every hour | Generate health interventions |
| `generate_hourly_summary` | Every hour | Generate hourly activity summary |
| `generate_daily_summary` | Daily at midnight | Generate daily activity summary |
| `archive_data_periodic` | Daily at 3 AM | Archive old data to storage |

### Running Multiple Services Together

Use multiple terminal windows or a process manager:

#### Option 1: Multiple Terminals

```bash
# Terminal 1: API Server
uvicorn src.main:app --reload

# Terminal 2: Celery Worker
celery -A src.celery_app.celery_app worker --loglevel=info

# Terminal 3: Celery Beat (optional)
celery -A src.celery_app.celery_app beat --loglevel=info
```

#### Option 2: Using tmux (recommended for development)

```bash
# Create a new tmux session
tmux new -s mobibox

# Start API server
uvicorn src.main:app --reload

# Split pane: Ctrl+b %
# Start Celery worker in new pane
celery -A src.celery_app.celery_app worker --loglevel=info

# Split pane again: Ctrl+b %
# Start Celery beat (optional)
celery -A src.celery_app.celery_app beat --loglevel=info

# Detach: Ctrl+b d
# Reattach: tmux attach -t mobibox
```

#### Option 3: Background Mode

```bash
# Start services in background
uvicorn src.main:app --port 8000 &
celery -A src.celery_app.celery_app worker --loglevel=info &
celery -A src.celery_app.celery_app beat --loglevel=info &

# View logs
tail -f nohup.out

# Stop all background processes
pkill -f "uvicorn\|celery"
```

### Verification

```bash
# Check API is running
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# Check Supabase connection
curl http://localhost:8000/supabase-test

# Check Celery worker is running
celery -A src.celery_app.celery_app inspect active
# Expected: -> celery@hostname: OK

# Check registered tasks
celery -A src.celery_app.celery_app inspect registered
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `Connection refused` to RabbitMQ | Ensure RabbitMQ is running: `docker ps \| grep rabbitmq` |
| Celery tasks not executing | Check worker logs for errors, verify environment variables |
| LLM errors | Verify `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` in `.env` |
| Supabase connection errors | Verify `SUPABASE_URL` and `SUPABASE_ANON_KEY` in `.env` |
| Port 8000 already in use | Kill existing process: `lsof -i :8000` then `kill -9 <PID>` |

## API Endpoints

### Health Check
- `GET /health` - Returns `{"status": "healthy"}`

### Supabase Connection Test
- `GET /supabase-test` - Tests Supabase connection

### User Registration
- `POST /register` - Register a new user
  ```json
  {
    "name": "unique_username"
  }
  ```

### Data Upload
- `POST /upload/documents` - Bulk upload document data
  ```json
  {
    "items": [
      {
        "user": "username",
        "timestamp": "2024-01-01T00:00:00Z",
        "volume": 80,
        "screen_on_ratio": 0.5,
        "wifi_connected": true,
        "wifi_ssid": "MyWiFi",
        "network_traffic": 1024.5,
        "Rx_traffic": 512.0,
        "Tx_traffic": 512.5,
        "stepcount_sensor": 1500,
        "gpsLat": 37.7749,
        "gpsLon": -122.4194,
        "battery": 85,
        "current_app": "com.example.app",
        "bluetooth_devices": ["device1", "device2"],
        "address": "123 Main St",
        "poi": ["Coffee Shop", "Restaurant"],
        "nearbyBluetoothCount": 3,
        "topBluetoothDevices": ["device1", "device2", "device3"]
      }
    ]
  }
  ```

- `POST /upload/imu` - Bulk upload IMU sensor data
  ```json
  {
    "items": [
      {
        "user": "username",
        "timestamp": "2024-01-01T00:00:00Z",
        "acc_X": 0.1,
        "acc_Y": -0.2,
        "acc_Z": 9.8,
        "gyro_X": 0.01,
        "gyro_Y": -0.02,
        "gyro_Z": 0.03,
        "mag_X": 1.0,
        "mag_Y": 2.0,
        "mag_Z": 3.0
      }
    ]
  }
  ```

### Query Endpoints

- `POST /get_summary_log` - Fetch the most recent summary log for a user
  ```json
  {
    "user": "username",
    "log_type": "hourly"  // or "daily"
  }
  ```

- `POST /get_intervention` - Fetch the most recent intervention for a user
  ```json
  {
    "user": "username"
  }
  ```

- `POST /send_intervention_feedback` - Submit feedback for an intervention
  ```json
  {
    "user": "username",
    "intervention_id": 123,
    "feedback": "This intervention was helpful",
    "mc1": "Strongly agree",
    "mc2": "Agree",
    "mc3": "Neutral",
    "mc4": "Disagree",
    "mc5": "Strongly disagree",
    "mc6": "Not applicable"
  }
  ```

- `POST /send_log_feedback` - Submit feedback for a summary log
  ```json
  {
    "user": 123,
    "summary_logs_id": 456,
    "feedback": "The summary was accurate"
  }
  ```

## Testing

### Run Test Suite

```bash
conda activate Mobibox_backend

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest src/test/test_celery_services.py

# Run specific test class
pytest src/test/test_celery_services.py::TestSummaryService -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Categories

| Test File | Description |
|-----------|-------------|
| `test_upload.py` | API endpoint tests |
| `test_llm_utils.py` | LLM service tests |
| `test_celery_services.py` | Celery service tests |
| `test_celery_tasks.py` | Celery task tests |
| `test_query.py` | Query endpoint tests |
| `test_intervention_pipeline_integration.py` | Intervention pipeline integration tests |

### Manual Task Testing

```bash
# Test HAR processing for a user
celery -A src.celery_app.celery_app call process_har_batch --args='["user1"]'

# Test atomic activities for a user
celery -A src.celery_app.celery_app call process_atomic_activities_batch --args='["user1"]'

# Check active tasks
celery -A src.celery_app.celery_app inspect active

# Check registered tasks
celery -A src.celery_app.celery_app inspect registered
```

## Project Structure

```
MobiBox_server/
├── environment.yml           # Conda environment configuration
├── .env.example              # Example environment variables
├── README.md                 # This file
├── scripts/                  # Service management scripts
│   ├── start_services.sh    # Start all services
│   ├── stop_services.sh     # Stop all services
│   ├── restart_services.sh  # Restart all services
│   └── status.sh            # Check service status
├── logs/                     # Service logs (created at runtime)
│   ├── api.log              # FastAPI logs
│   ├── celery_worker.log    # Celery worker logs
│   └── celery_beat.log      # Celery beat logs
├── migrations/               # Database migrations
│   ├── 002_add_missing_features_tables.sql
│   ├── 003_database_improvements.sql
│   ├── 004_add_log_feedback_fields.sql
│   └── 005_storage_archival.sql
├── docs/                     # Documentation
│   └── TSFM_INTEGRATION.md  # TSFM model documentation
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Application configuration
│   ├── database.py          # Supabase client initialization
│   ├── register/            # User registration module
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── routes.py        # Registration API routes
│   │   ├── schemas.py       # Pydantic schemas
│   │   └── service.py       # Business logic
│   ├── upload/              # Data upload module
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── routes.py        # Upload API routes
│   │   ├── schemas.py       # Pydantic schemas
│   │   └── service.py       # Business logic
│   ├── celery_app/          # Celery tasks for async processing
│   │   ├── __init__.py
│   │   ├── celery_app.py    # Celery app instance
│   │   ├── config.py        # Celery configuration
│   │   ├── tasks/           # Celery task modules
│   │   │   ├── har_tasks.py
│   │   │   ├── atomic_tasks.py
│   │   │   ├── summary_tasks.py
│   │   │   └── archive_tasks.py    # Data archival tasks (new)
│   │   ├── services/        # Business logic
│   │   │   ├── har_service.py
│   │   │   ├── atomic_service.py
│   │   │   ├── summary_service.py
│   │   │   ├── app_category_service.py  # App category lookup (new)
│   │   │   ├── processing_state_service.py  # User state tracking (new)
│   │   │   ├── archive_service.py     # Data archival service (new)
│   │   │   ├── intervention_service.py
│   │   │   ├── tsfm_service.py         # TSFM model wrapper
│   │   │   └── tsfm_model/             # TSFM model code
│   │   │       ├── __init__.py
│   │   │       ├── config.py
│   │   │       ├── encoder.py
│   │   │       ├── semantic_alignment.py
│   │   │       ├── token_text_encoder.py
│   │   │       ├── feature_extractor.py
│   │   │       ├── transformer.py
│   │   │       ├── positional_encoding.py
│   │   │       ├── preprocessing.py
│   │   │       ├── label_groups.py
│   │   │       ├── model_loading.py
│   │   │       └── ckpts/              # Model checkpoints
│   │   │           ├── best.pt
│   │   │           └── hyperparameters.json
│   │   ├── schemas/         # Pydantic schemas
│   │   │   ├── har_schemas.py
│   │   │   └── atomic_schemas.py
│   │   └── README.md        # Celery documentation
│   ├── llm_utils/           # LLM integration utilities
│   │   └── services.py
│   ├── services/             # External services (new)
│   │   ├── __init__.py
│   │   └── baidu_maps.py    # Baidu Maps API client
│   ├── query/               # Query module for summary logs and interventions
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── routes.py        # Query API routes
│   │   ├── schemas.py       # Pydantic schemas
│   │   └── service.py       # Business logic
│   └── test/                # Test suite
│       ├── __init__.py
│       ├── conftest.py      # Pytest fixtures
│       ├── test_upload.py   # Upload endpoint tests
│       ├── test_llm_utils.py # LLM service tests
│       ├── test_celery_services.py # Celery service tests
│       ├── test_celery_tasks.py # Celery task tests
│       ├── test_query.py    # Query endpoint tests
│       ├── test_tsfm.py     # TSFM model tests
│       └── test_intervention_pipeline_integration.py # Intervention pipeline tests
└── .gitignore
```

## Installed Libraries

| Library | Purpose |
|---------|---------|
| fastapi | Web framework for building APIs |
| uvicorn | ASGI server |
| pydantic | Data validation using Python type hints |
| pydantic-settings | Settings management |
| supabase | Supabase Python client |
| celery | Distributed task queue |
| celery-beat | Celery scheduler for periodic tasks |
| pytest | Testing framework |
| pytest-asyncio | Async testing support |
| python-dotenv | Environment variable management |
| httpx | HTTP client for testing |
| black | Code formatter |
| isort | Import sorter |
| flake8 | Style guide enforcement |
| mypy | Static type checker |
| pyarrow | Parquet file format for data archival |

## Database Schema

The application connects to Supabase and uses the following tables:

### users table
- `name` (text, primary key) - Unique user identifier

### uploads table
- `user` (character varying) - User identifier (foreign key to user.name)
- `timestamp` (timestamp with time zone) - Record timestamp, defaults to now()
- `volume` (smallint) - Audio volume level (0-100)
- `screen_on_ratio` (real) - Screen on time ratio (0.0-1.0)
- `wifi_connected` (boolean) - WiFi connection status
- `wifi_ssid` (character varying) - WiFi network SSID
- `network_traffic` (real) - Total network traffic in bytes
- `Rx_traffic` (real) - Received traffic in bytes
- `Tx_traffic` (real) - Transmitted traffic in bytes
- `stepcount_sensor` (smallint) - Step count
- `gpsLat` (double precision) - GPS latitude
- `gpsLon` (double precision) - GPS longitude
- `battery` (smallint) - Battery percentage (0-100)
- `current_app` (character varying) - Current foreground app package name
- `bluetooth_devices` (character varying[]) - Array of Bluetooth device names
- `address` (character varying) - Physical address
- `poi` (character varying[]) - Array of points of interest
- `nearbyBluetoothCount` (smallint) - Count of nearby Bluetooth devices
- `topBluetoothDevices` (character varying[]) - Array of top Bluetooth device names

### imu table
- `user` (text) - User identifier
- `timestamp` (timestamptz) - Record timestamp
- `acc_X`, `acc_Y`, `acc_Z` (float) - Accelerometer readings
- `gyro_X`, `gyro_Y`, `gyro_Z` (float) - Gyroscope readings
- `mag_X`, `mag_Y`, `mag_Z` (float) - Magnetometer readings

### har table
- `id` (bigint, auto-generated) - Primary key
- `timestamp` (timestamptz) - Record timestamp
- `user` (varchar) - User identifier
- `har_label` (enum) - Activity label (walking, running, sitting, etc.)
- `confidence` (real, new) - Confidence score (0-1)
- `source` (varchar, new) - Source: 'tsfm_model', 'imu_model', 'mock_har', 'insufficient_data'

### atomic_activities table
- `user` (text) - User identifier
- `timestamp` (timestamptz) - Record timestamp
- `har_label` (text) - HAR activity label
- `app_category` (text) - App usage category
- `step_label` (text) - Step activity label
- `phone_usage` (text) - Phone usage pattern
- `social_label` (text) - Social context label
- `movement_label` (text) - Movement pattern label
- `location_label` (text) - Location context

### interventions table
- `user` (text) - User identifier
- `intervention_type` (text) - Type of intervention
- `message` (text) - Intervention message
- `priority` (text) - Priority (low/medium/high)
- `category` (text) - Category
- `timestamp` (timestamptz) - Record timestamp

### summary_logs table
- `user` (text) - User identifier
- `log_type` (text) - hourly or daily
- `title` (text) - Summary title
- `summary` (text) - Summary narrative
- `highlights` (jsonb) - Key highlights
- `recommendations` (jsonb) - Recommendations
- `timestamp` (timestamptz) - Record timestamp

### app_categories table (new)
- `id` (serial, primary key) - Auto-increment ID
- `app_name` (text, unique) - App package name
- `category` (text) - Category classification
- `source` (text) - 'lookup' (predefined) or 'llm' (learned)
- `created_at` (timestamptz) - Record timestamp

### user_processing_state table (new)
- `user` (text, primary key) - User identifier
- `last_har_timestamp` (timestamptz) - Last HAR processing time
- `last_atomic_timestamp` (timestamptz) - Last atomic activity time
- `last_upload_timestamp` (timestamptz) - Last data upload time
- `data_collection_start` (timestamptz) - Data collection start time
- `last_summary_generated` (timestamptz) - Last summary generation time
- `updated_at` (timestamptz) - Record update time

## Celery Tasks

The system uses Celery for asynchronous processing of sensor data and generating health interventions.

### Task Pipelines

| Pipeline | Trigger | Description |
|----------|---------|-------------|
| **HAR Processing** | IMU data upload | Classify human activity from sensor data |
| **Atomic Activities** | Document upload | Generate 7-dimensional activity labels |
| **Interventions** | Scheduled (hourly) | LLM-generated health suggestions |
| **Summaries** | Scheduled (hourly/daily) | Activity summary logs |

### Quick Reference

```bash
# Start Celery worker
celery -A src.celery_app.celery_app worker --loglevel=info

# Start Celery beat scheduler
celery -A src.celery_app.celery_app beat --loglevel=info

# Manual task execution
celery -A src.celery_app.celery_app call process_har_batch --args='["user1"]'
```

For detailed Celery documentation, see [src/celery_app/README.md](src/celery_app/README.md).

## TSFM Model Integration

The backend uses a TSFM (Time Series Foundation Model) for Human Activity Recognition (HAR). The model is a semantic-aligned encoder trained on UCI-HAR dataset with zero-shot capability.

### Model Details

| Property | Value |
|----------|-------|
| Architecture | Semantic-aligned encoder with contrastive learning |
| Training Data | UCI-HAR dataset (6 activities) |
| Input | 9-channel IMU data (accelerometer, gyroscope, magnetometer) |
| Labels | walking, walking_upstairs, walking_downstairs, sitting, standing, laying |
| Checkpoint | `src/celery_app/services/tsfm_model/ckpts/best.pt` |

### Label Mapping

TSFM labels are mapped to MobiBox activity labels:

| TSFM Label | MobiBox Label |
|------------|---------------|
| walking | walking |
| walking_upstairs | climbing stairs |
| walking_downstairs | climbing stairs |
| sitting | sitting |
| standing | standing |
| laying | lying |

### Setup

1. **Download the checkpoint** from remote server:
   ```bash
   # Create checkpoint directory
   mkdir -p src/celery_app/services/tsfm_model/ckpts

   # Download checkpoint and hyperparameters
   scp user@server:/path/to/best.pt src/celery_app/services/tsfm_model/ckpts/
   scp user@server:/path/to/hyperparameters.json src/celery_app/services/tsfm_model/ckpts/
   ```

2. **Verify the model loads correctly**:
   ```bash
   python -c "from src.celery_app.services.tsfm_service import _get_tsfm_model; model, _, available = _get_tsfm_model(); print(f'TSFM available: {available}')"
   ```

### Testing

Run TSFM-specific tests:
```bash
# Run all TSFM tests
pytest src/test/test_tsfm.py -v

# Run specific test categories
pytest src/test/test_tsfm.py::TestTSFMPreprocessing -v
pytest src/test/test_tsfm.py::TestTSFMLabelMapping -v
pytest src/test/test_tsfm.py::TestTSFMInference -v
pytest src/test/test_tsfm.py::TestTSFMBatchInference -v
```

### Inference Flow

```
IMU Data (N samples × 9 channels)
        ↓
    Preprocessing
  (patching, normalization)
        ↓
    TSFM Encoder
  (semantic embeddings)
        ↓
  Cosine Similarity
  (with label embeddings)
        ↓
    Activity Label
```

For detailed TSFM documentation, see [docs/TSFM_INTEGRATION.md](docs/TSFM_INTEGRATION.md).

## Feature Implementation Details

This section documents the key features implemented in the MobiBox backend.

### 1. App Category Table Lookup

**Purpose:** Reduce LLM API calls for app classification by using a cached lookup table.

**Files:**
- `src/celery_app/services/app_category_service.py` - App category lookup service
- `src/celery_app/services/atomic_service.py` - Integration point

**How it works:**
1. In-memory cache with 70+ predefined common apps
2. Database cache (`app_categories` table) for learned classifications
3. LLM fallback for unknown apps
4. Results cached in database for future use

**Categories:**
- social communication app
- video and music app
- games or gaming platform
- e-commerce/shopping platform
- office/working app
- learning and education app
- health management/self-discipline app
- financial services app
- news/reading app
- tool/engineering/functional app
- uncertain

---

### 2. Last Processed Timestamp Tracking

**Purpose:** Enable incremental data processing to avoid reprocessing the same data.

**Files:**
- `src/celery_app/services/processing_state_service.py` - State tracking service

**Database Table:** `user_processing_state`

| Column | Description |
|--------|-------------|
| user | User identifier (primary key) |
| last_har_timestamp | Last HAR processing timestamp |
| last_atomic_timestamp | Last atomic activity timestamp |
| last_upload_timestamp | Last data upload timestamp |
| data_collection_start | When user started collecting data |
| last_summary_generated | Last summary log timestamp |

---

### 3. Hourly Log Threshold Check

**Purpose:** Only generate summary logs when sufficient data is available.

**Configuration** (`src/celery_app/config.py`):
```python
MIN_ATOMIC_RECORDS_FOR_HOURLY_LOG = 60  # At least 60 records
MIN_UNIQUE_LABELS_FOR_LOG = 3  # At least 3 unique activity types
```

**Function:** `should_generate_summary()` in `summary_service.py`

---

### 4. Per-User Hourly Timer

**Purpose:** Generate logs based on user's data accumulation, not fixed schedule.

**Configuration:**
```python
MIN_DATA_COLLECTION_HOURS = 1  # Minimum 1 hour of data
MIN_HOURS_BETWEEN_SUMMARIES = 1  # Minimum 1 hour between logs
```

**Function:** `check_user_hourly_ready()` in `summary_service.py`

**Flow:**
1. User starts collecting data → `data_collection_start` is set
2. Wait until user has 1+ hour of data
3. Wait until at least 1 hour since last summary
4. Check data threshold requirements
5. Generate and store summary

---

### 5. Mobile Polling Mechanism

**Purpose:** Allow mobile app to efficiently detect new logs via polling.

**API Changes:**

**Request:**
```json
POST /get_summary_log
{
  "user": "username",
  "log_type": "hourly",
  "last_log_id": 123  // Optional: ID of last received log
}
```

**Response:**
```json
{
  "status": "success",
  "data": { ... },  // Null if no new log
  "has_new_log": true  // False if last_log_id matches latest
}
```

**Usage:**
1. Mobile calls `/get_summary_log` without `last_log_id` to get initial log
2. Mobile stores `data.id` as `last_log_id`
3. Mobile polls with `last_log_id` to check for new logs
4. If `has_new_log` is `false`, no new data to download

---

### 6. Baidu Map API Integration

**Purpose:** Enrich GPS coordinates with location names via reverse geocoding.

**Files:**
- `src/services/baidu_maps.py` - Baidu Maps API client
- `src/config.py` - Configuration settings

**Configuration:**
```env
BAIDU_MAPS_API_KEY=your-api-key
BAIDU_MAPS_ENABLED=true
```

**Features:**
- Reverse geocoding (GPS → address, POI, city, district)
- 1-hour cache to minimize API calls
- Automatic fallback to provided address/POI if API fails

**Response fields:**
- `address` - Formatted address
- `poi` - List of nearby POIs
- `city` - City name
- `district` - District name
- `business` - Business area

---

## Database Migration

Run migrations in order:

```bash
# Migration 002: Add missing features tables
psql -f migrations/002_add_missing_features_tables.sql

# Migration 003: Database improvements (indexes, constraints, triggers)
psql -f migrations/003_database_improvements.sql

# Migration 004: Add log feedback fields
psql -f migrations/004_add_log_feedback_fields.sql

# Migration 005: Storage archival setup
psql -f migrations/005_storage_archival.sql
```

### Migration 002: New Tables

Creates tables for app categories cache and user processing state:

```sql
-- App categories cache
CREATE TABLE IF NOT EXISTS app_categories (
    id SERIAL PRIMARY KEY,
    app_name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    source TEXT DEFAULT 'llm',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User processing state
CREATE TABLE IF NOT EXISTS user_processing_state (
    "user" TEXT PRIMARY KEY,
    last_har_timestamp TIMESTAMPTZ,
    last_atomic_timestamp TIMESTAMPTZ,
    last_upload_timestamp TIMESTAMPTZ,
    data_collection_start TIMESTAMPTZ,
    last_summary_generated TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Migration 003: Performance Improvements

Adds indexes, constraints, and triggers for better performance:

- **Foreign Keys**: Links `user_processing_state.user` to `user.name`
- **Indexes**: Optimizes common queries on `atomic_activities`, `har`, `summary_logs`
- **Triggers**: Auto-updates `updated_at` timestamps
- **Constraints**: Validates `log_type`, `source` values
- **Views**: `v_user_recent_activity`, `v_users_ready_for_summary`

---

## Database Schema Reference

### Core Tables

#### `har` Table
Human Activity Recognition labels from IMU sensor data.

| Column | Type | Description |
|--------|------|-------------|
| `id` | bigint | Auto-generated primary key |
| `timestamp` | timestamptz | When the activity was detected |
| `user` | varchar | User identifier (FK to user.name) |
| `har_label` | enum | Activity label (walking, running, sitting, etc.) |
| `confidence` | real | Model confidence score (0.0-1.0) |
| `source` | varchar | Source: 'tsfm_model', 'imu_model', 'mock_har', 'insufficient_data' |

#### `atomic_activities` Table
7-dimensional atomic activity labels.

| Column | Type | Description |
|--------|------|-------------|
| `id` | bigint | Auto-generated primary key |
| `timestamp` | timestamptz | When the activity was generated |
| `user` | varchar | User identifier (FK to user.name) |
| `har_label` | enum | HAR activity label |
| `app_category` | enum | App usage category |
| `app_name` | varchar | Specific app package name (new) |
| `step_count` | enum | Step activity label |
| `phone_usage` | enum | Phone usage pattern |
| `social` | enum | Social context label |
| `movement` | enum | Movement pattern label |
| `location` | varchar | Location context |

#### `summary_logs` Table
Hourly and daily activity summaries.

| Column | Type | Description |
|--------|------|-------------|
| `id` | bigint | Auto-generated primary key (used for polling) |
| `timestamp` | timestamptz | When the summary was generated |
| `user` | varchar | User identifier |
| `log_type` | enum | 'hourly' or 'daily' |
| `summary` | text | Summary content |
| `start_timestamp` | timestamptz | Window start time |
| `end_timestamp` | timestamptz | Window end time |

#### `app_categories` Table
Cache for app category classifications.

| Column | Type | Description |
|--------|------|-------------|
| `id` | serial | Primary key |
| `app_name` | text | App package name (unique) |
| `category` | text | Category classification |
| `source` | text | 'lookup' or 'llm' |
| `created_at` | timestamptz | When cached |

#### `user_processing_state` Table
Per-user processing timestamps for incremental processing.

| Column | Type | Description |
|--------|------|-------------|
| `user` | text | User identifier (PK, FK to user.name) |
| `last_har_timestamp` | timestamptz | Last HAR processing time |
| `last_atomic_timestamp` | timestamptz | Last atomic activity time |
| `last_upload_timestamp` | timestamptz | Last data upload time |
| `data_collection_start` | timestamptz | When user started collecting data |
| `last_summary_generated` | timestamptz | Last summary generation time |
| `updated_at` | timestamptz | Auto-updated timestamp |

### Indexes

Key indexes for performance:

| Table | Index | Purpose |
|-------|-------|---------|
| `atomic_activities` | `(user, timestamp DESC)` | User activity queries |
| `har` | `(user, timestamp DESC)` | HAR label queries |
| `summary_logs` | `(user, log_type, timestamp DESC)` | Summary polling |
| `imu` | `(user, timestamp DESC)` | IMU data queries |
| `app_categories` | `(app_name)` | App category lookup |

### Views

#### `v_user_recent_activity`
Recent activity summary per user:
```sql
SELECT user, MAX(timestamp) as last_activity,
       COUNT(*) as activity_count,
       MODE() WITHIN GROUP (ORDER BY har_label) as dominant_activity
FROM atomic_activities GROUP BY user;
```

#### `v_users_ready_for_summary`
Users ready for summary generation (1+ hour of data, 1+ hour since last summary):
```sql
SELECT user, data_collection_start, last_summary_generated,
       EXTRACT(EPOCH FROM (NOW() - data_collection_start))/3600 as hours_since_start
FROM user_processing_state
WHERE data_collection_start IS NOT NULL
  AND hours_since_start >= 1
  AND (last_summary_generated IS NULL OR hours_since_last_summary >= 1);
```

---

## Data Archival System

The system automatically archives old data to Supabase Storage to reduce database size and costs. Archived data is stored in **Parquet format with Snappy compression**, achieving **10-100x compression** compared to CSV.

### Overview

| Component | Purpose |
|------------|---------|
| Archive Service | Exports old records to Parquet files |
| Celery Beat | Schedules daily archival at 3 AM |
| Storage Bucket | Holds archived Parquet files |
| Archival Logs | Audit trail of all archival operations |

### Storage Format

Archives are stored as Parquet files with Snappy compression:

```
mobibox-archive/
└── archives/
    ├── imu/
    │   └── 2026/
    │       └── 03/
    │           └── 2026-03-01.parquet
    ├── har/
    │   └── 2026/
    │       └── 03/
    │           └── 2026-03-01.parquet
    └── atomic_activities/
        └── ...
```

**Benefits of Parquet:**
- **10-100x smaller** than CSV (columnar + dictionary encoding)
- **Type preservation** (timestamps, numbers remain typed)
- **Query optimization** (built-in statistics)
- **Industry standard** for data lakes and analytics

### Retention Policy

| Table | Retention | Reason |
|-------|------------|--------|
| `imu` | 7 days | Highest volume sensor data |
| `har` | 30 days | Derived from IMU, lower volume |
| `atomic_activities` | 30 days | Activity summaries |
| `uploads` | 30 days | Document uploads |
| `summary_logs` | 90 days | Important user summaries |
| `interventions` | 90 days | Health interventions |

### Configuration

Add to `.env`:

```env
# Storage bucket name (create in Supabase dashboard)
STORAGE_BUCKET=mobibox-archive

# Retention days (optional, defaults shown)
RETENTION_IMU_DAYS=7
RETENTION_HAR_DAYS=30
RETENTION_ATOMIC_DAYS=30
RETENTION_UPLOADS_DAYS=30
RETENTION_SUMMARY_LOGS_DAYS=90
RETENTION_INTERVENTIONS_DAYS=90

# Archival settings
ARCHIVE_ENABLED=true
ARCHIVE_BATCH_SIZE=10000
```

### Setup

#### 1. Create Storage Bucket

In Supabase Dashboard:
1. Go to **Storage** → **Create a new bucket**
2. Name: `mobibox-archive`
3. Public: **No** (private bucket)

Or via SQL:
```sql
INSERT INTO storage.buckets (id, name, public)
VALUES ('mobibox-archive', 'mobibox-archive', false)
ON CONFLICT (id) DO NOTHING;
```

#### 2. Add Service Role Key

The archival service needs admin access to storage:

```env
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIs...
```

Get this from **Project Settings → API → service_role** (keep secret!).

#### 3. Run Migration

```sql
-- Migration 005: Create archival_logs table
\i migrations/005_storage_archival.sql
```

#### 4. Install PyArrow

```bash
conda activate Mobibox_backend
conda install -c conda-forge pyarrow
# or
pip install pyarrow
```

### Manual Archival

Trigger archival manually:

```bash
# Archive all tables
celery -A src.celery_app.celery_app call archive_data_periodic

# Archive specific table
celery -A src.celery_app.celery_app call archive_table_manual --args='["imu", 7]'

# Check archive statistics
celery -A src.celery_app.celery_app call get_archive_stats
```

### Monitoring

#### Check Archival Logs

```sql
-- Recent archival operations
SELECT * FROM v_archival_history;

-- Recent failures
SELECT * FROM v_recent_archival_failures;

-- Storage efficiency over time
SELECT
    table_name,
    archival_timestamp,
    records_archived,
    file_size_bytes,
    ROUND(file_size_bytes::numeric / NULLIF(records_archived, 0), 2) as bytes_per_record
FROM archival_logs
WHERE status = 'completed'
ORDER BY archival_timestamp DESC
LIMIT 20;
```

#### Storage Efficiency

```sql
-- Average bytes per record by table
SELECT
    table_name,
    COUNT(*) as archive_count,
    SUM(records_archived) as total_records,
    SUM(file_size_bytes) as total_bytes,
    ROUND(SUM(file_size_bytes)::numeric / NULLIF(SUM(records_archived), 0), 2) as avg_bytes_per_record,
    pg_size_pretty(SUM(file_size_bytes)) as total_size
FROM archival_logs
WHERE status = 'completed'
GROUP BY table_name;
```

### Restoring Data

To restore archived data:

```python
import pandas as pd
from supabase import create_client

# Download Parquet file
supabase = create_client(url, service_role_key)
response = supabase.storage.from_('mobibox-archive').download('archives/imu/2026/03/2026-03-01.parquet')

# Read Parquet
df = pd.read_parquet(io.BytesIO(response))

# Insert back to database
# ... (your restore logic)
```

### Archival Logs Table

```sql
CREATE TABLE archival_logs (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    records_archived INTEGER DEFAULT 0,
    records_deleted INTEGER DEFAULT 0,
    storage_path VARCHAR(500),
    file_size_bytes BIGINT,  -- Parquet file size
    archival_timestamp TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20),  -- 'completed', 'failed', 'partial'
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Service Management Scripts

The `scripts/` directory contains utilities for managing services:

| Script | Description |
|--------|-------------|
| `start_services.sh` | Start all services (RabbitMQ, FastAPI, Celery worker/beat) |
| `stop_services.sh` | Stop all running services |
| `restart_services.sh` | Restart all services |
| `status.sh` | Check status of all services |

### Usage

```bash
# Start all services
./scripts/start_services.sh

# Check service status
./scripts/status.sh

# Stop all services (will prompt to stop RabbitMQ)
./scripts/stop_services.sh

# Restart all services
./scripts/restart_services.sh
```

### Logs

Service logs are stored in the `logs/` directory:
- `logs/api.log` - FastAPI server logs
- `logs/celery_worker.log` - Celery worker logs
- `logs/celery_beat.log` - Celery beat scheduler logs

```bash
# View API logs
tail -f logs/api.log

# View Celery worker logs
tail -f logs/celery_worker.log
```
