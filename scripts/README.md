# MobiBox Service Management Scripts

This directory contains scripts for managing MobiBox backend services.

## Script Inventory

| Script | Linux/macOS | Windows | Description |
|--------|-------------|---------|-------------|
| Start Services | `start_services.sh` | `start_services.ps1` | Start RabbitMQ, FastAPI, Celery Worker, Celery Beat |
| Stop Services | `stop_services.sh` | `stop_services.ps1` | Stop all services (optional RabbitMQ prompt) |
| Restart Services | `restart_services.sh` | `restart_services.ps1` | Stop all services, wait 3s, start all services |
| Status Check | `status.sh` | `status.ps1` | Check status of all services, ports, and models |
| Download Models | `download_models.sh` | `download_models.ps1` | Download sentence-transformers models for offline use |
| Download Models (Python) | `download_sentence_transformers.py` | *(cross-platform)* | Standalone Python script with `--verify` mode |

### Shared Module

- `common.ps1` — Shared helper functions dot-sourced by all PowerShell scripts (process management, port checking, conda helpers, etc.)

## Platform Quick-Start

### Windows (PowerShell)

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for RabbitMQ)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.x installed in the `Mobibox_backend` conda environment
- PowerShell 5.1 or later (included with Windows 10+)

**First-time setup:**
```powershell
# Allow script execution (run once, as Administrator if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Initialize conda for PowerShell (run once)
conda init powershell

# Create the conda environment
conda env create -f environment.yml
```

**Usage:**
```powershell
# Start all services
.\scripts\start_services.ps1

# Check service status
.\scripts\status.ps1

# Stop all services
.\scripts\stop_services.ps1

# Restart all services
.\scripts\restart_services.ps1

# Download ML models
.\scripts\download_models.ps1

# Verify models are cached
.\scripts\download_models.ps1 -Verify
```

### Linux / macOS (bash)

**Prerequisites:**
- Docker (for RabbitMQ)
- Miniconda or Anaconda
- Python 3.x installed in the `Mobibox_backend` conda environment
- `lsof`, `pgrep`, `pkill` (usually pre-installed)

**First-time setup:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Create the conda environment
conda env create -f environment.yml
```

**Usage:**
```bash
# Start all services
./scripts/start_services.sh

# Check service status
./scripts/status.sh

# Stop all services
./scripts/stop_services.sh

# Restart all services
./scripts/restart_services.sh

# Download ML models
./scripts/download_models.sh
```

## Services Managed

| Service | Port | Description |
|---------|------|-------------|
| RabbitMQ | 5672, 15672 (Web UI) | Message queue broker (Docker container) |
| FastAPI | 8000 | Main API server (uvicorn) |
| Celery Worker | — | Background task worker |
| Celery Beat | — | Periodic task scheduler |

## Logs and PID Files

All log files and PID files are written to the `logs/` directory at the project root.

| File | Purpose |
|------|---------|
| `logs/api.pid` | FastAPI server process ID |
| `logs/celery_worker.pid` | Celery worker process ID |
| `logs/celery_beat.pid` | Celery beat process ID |
| `logs/api.log` | FastAPI server logs |
| `logs/celery_worker.log` | Celery worker logs |
| `logs/celery_beat.log` | Celery beat logs |

## Key Differences Between Platforms

| Feature | Linux/macOS (bash) | Windows (PowerShell) |
|---------|-------------------|---------------------|
| Process detection | `pgrep -f "pattern"` | PID files (primary) + WMI (fallback) |
| Port checking | `lsof -Pi :$port` | `Get-NetTCPConnection` (or `netstat`) |
| Background processes | `nohup command &` | `Start-Process -WindowStyle Hidden` |
| Process termination | `pkill -f "pattern"` | `taskkill /PID /T /F` (tree kill) |
| Conda activation | `eval "$(conda shell.bash hook)"` | `conda activate` (via `conda init powershell`) |
| Interactive timeout | `read -t 5` | `$host.UI.RawUI.KeyAvailable` polling loop |
| HTTP requests | `curl -s` | `Invoke-RestMethod` |
| Log tailing | `tail -f` | `Get-Content -Wait` |

## Troubleshooting

### "Running scripts is disabled on this system" (Windows)

Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "conda is not recognized" (Windows)

Make sure conda is in your PATH, or initialize it for PowerShell:
```powershell
conda init powershell
```
Then restart your PowerShell session.

### "Get-NetTCPConnection" access denied (Windows)

Port checking via `Get-NetTCPConnection` may require administrator privileges on some systems. The scripts fall back to `netstat -ano` if the cmdlet fails.

### "Error: Failed to activate conda environment"

Make sure the `Mobibox_backend` environment exists:
```bash
conda env create -f environment.yml
```
