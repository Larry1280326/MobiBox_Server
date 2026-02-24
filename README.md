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

### 3. Start Infrastructure Services

```bash
# Start RabbitMQ (required for Celery)
docker run -d --name rabbitmq -p 5672:5672 rabbitmq

# Verify RabbitMQ is running
docker ps | grep rabbitmq
```

### 4. Start the Application

Open separate terminals for each service:

```bash
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

### 5. Verify Services

```bash
# Check API health
curl http://localhost:8000/health

# Check Celery worker status
celery -A src.celery_app.celery_app inspect active
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
| Azure OpenAI API | LLM integration for interventions |

### Configuration

Copy `.env.example` to `.env` and configure the following:

```bash
cp .env.example .env
```

#### Supabase Configuration

1. Create a Supabase project at https://supabase.com/
2. Get your project URL and anon key from **Project Settings → API**
3. Update `.env`:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
```

#### Azure OpenAI Configuration (for LLM features)

1. Get your Azure OpenAI credentials from [HKUST Azure OpenAI Service](https://itso.hkust.edu.hk/services/it-infrastructure/azure-openai-api-service)
2. Update `.env`:

```env
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://hkust.azure-api.net
AZURE_OPENAI_API_VERSION=2024-10-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
DEFAULT_TEMPERATURE=0.1
```

#### RabbitMQ / Celery Configuration

Default configuration works for local development:

```env
RABBITMQ_URL=amqp://guest:guest@localhost:5672//
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672//
CELERY_RESULT_BACKEND=rpc://
```

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
│   │   │   └── summary_tasks.py
│   │   ├── services/        # Business logic
│   │   │   ├── har_service.py
│   │   │   ├── atomic_service.py
│   │   │   └── summary_service.py
│   │   ├── schemas/         # Pydantic schemas
│   │   │   ├── har_schemas.py
│   │   │   └── atomic_schemas.py
│   │   └── README.md        # Celery documentation
│   ├── llm_utils/           # LLM integration utilities
│   │   └── services.py
│   └── test/                # Test suite
│       ├── __init__.py
│       ├── conftest.py      # Pytest fixtures
│       └── test_upload.py   # Upload endpoint tests
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
- `user` (text) - User identifier
- `label` (text) - Activity label (walking, running, sitting, etc.)
- `confidence` (float) - Confidence score (0-1)
- `source` (text) - Source of label
- `timestamp` (timestamptz) - Record timestamp

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
