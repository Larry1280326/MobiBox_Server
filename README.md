# MobiBox Backend

A FastAPI-based backend server for MobiBox with Supabase integration.

## Environment Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Python 3.11
- [Supabase](https://supabase.com/) account and project

### Supabase Configuration

1. Create a Supabase project at https://supabase.com/
2. Get your project URL and anon key from Project Settings → API
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Update `.env` with your Supabase credentials:
   ```
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=your-anon-key-here
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

```bash
# Activate the environment first
conda activate Mobibox_backend

# Run the development server
uvicorn src.main:app --reload

# Run with specific host and port
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

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
        "volume": 0.8,
        "screen_on_ratio": 0.5,
        "wifi_connected": "Yes",
        "wifi_ssid": "MyWiFi",
        "network_traffic": 1024.5,
        "Rx_traffic": 512.0,
        "Tx_traffic": 512.5,
        "stepcount_sensor": 1500,
        "gpsLat": 37.7749,
        "gpsLon": -122.4194,
        "battery": 85.0,
        "current_app": "com.example.app",
        "bluetooth_devices": "device1,device2",
        "address": "123 Main St",
        "poi": "Coffee Shop",
        "nearbyBluetoothCount": 3,
        "topBluetoothDevices": "device1,device2,device3"
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

Run the test suite:

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest src/test/test_upload.py
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
- `user` (text) - User identifier
- `timestamp` (timestamptz) - Record timestamp
- `volume` (float) - Audio volume level
- `screen_on_ratio` (float) - Screen on time ratio
- `wifi_connected` (text) - WiFi connection status
- `wifi_ssid` (text) - WiFi network SSID
- `network_traffic` (float) - Total network traffic
- `Rx_traffic` (float) - Received traffic
- `Tx_traffic` (float) - Transmitted traffic
- `stepcount_sensor` (integer) - Step count
- `gpsLat` (float) - GPS latitude
- `gpsLon` (float) - GPS longitude
- `battery` (float) - Battery percentage
- `current_app` (text) - Current foreground app
- `bluetooth_devices` (text) - Comma-separated Bluetooth devices
- `address` (text) - Physical address
- `poi` (text) - Point of interest
- `nearbyBluetoothCount` (integer) - Nearby Bluetooth device count
- `topBluetoothDevices` (text) - Top Bluetooth devices

### imu table
- `user` (text) - User identifier
- `timestamp` (timestamptz) - Record timestamp
- `acc_X`, `acc_Y`, `acc_Z` (float) - Accelerometer readings
- `gyro_X`, `gyro_Y`, `gyro_Z` (float) - Gyroscope readings
- `mag_X`, `mag_Y`, `mag_Z` (float) - Magnetometer readings
