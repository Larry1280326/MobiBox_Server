# MobiBox Service Management Scripts

This directory contains scripts for managing MobiBox backend services.

## Script Inventory

| Script | Linux/macOS | Windows | Description |
|--------|-------------|---------|-------------|
| Start Services (nohup) | `start_services.sh` | `start_services.ps1` | Start services with nohup in background |
| Start Services (tmux)  | `tmux_start.sh`    | — | **Recommended for servers** — tmux session with live output |
| Stop Services | `stop_services.sh` | `stop_services.ps1` | Stop all services (optional RabbitMQ prompt) |
| Stop Services (tmux)   | `tmux_stop.sh`     | — | Stop tmux-managed services |
| Restart Services | `restart_services.sh` | `restart_services.ps1` | Stop all services, wait 3s, start all services |
| Status Check | `status.sh` | `status.ps1` | Check status of all services, ports, and models |
| Status Check (tmux)    | `tmux_status.sh`   | — | Check tmux session and service health |
| Attach tmux            | `tmux_attach.sh`   | — | Attach to running tmux session |
| Download Models | `download_models.sh` | `download_models.ps1` | Download sentence-transformers models for offline use |
| Download Models (Python) | `download_sentence_transformers.py` | *(cross-platform)* | Standalone Python script with `--verify` mode |

### Shared Module

- `common.ps1` — Shared helper functions dot-sourced by all PowerShell scripts (process management, port checking, conda helpers, etc.)

## tmux Setup (Recommended for Linux Servers)

The tmux scripts provide a **proper terminal multiplexer** approach — each service runs in its own tmux window with live visible output, and services survive SSH disconnects.

### Why tmux over nohup?

| Feature | nohup | tmux |
|---------|-------|------|
| Live output | ❌ Redirected to /dev/null | ✅ Each service in its own window |
| Reattach to view logs | `tail -f logs/*.log` | `tmux attach` — see all services at once |
| Graceful restart | `pkill` + restart | Kill window, start fresh in same session |
| Service isolation | One PID per process | Separate windows, independent scrollback |
| Log rotation | Python RotatingFileHandler | Same Python handler + visible pane output |

### Quick Start

```bash
# First-time: make scripts executable
chmod +x scripts/tmux_*.sh

# Start all services in tmux
./scripts/tmux_start.sh

# Attach to the session to view live output
./scripts/tmux_attach.sh

# Detach (services keep running): Ctrl-b d
# Stop everything:
./scripts/tmux_stop.sh

# Check status:
./scripts/tmux_status.sh
```

### tmux Session Layout

```
Session: mobibox
├── Window 0: api      — FastAPI server (uvicorn, port 8001)
├── Window 1: worker   — Celery worker (all queues)
├── Window 2: beat     — Celery beat scheduler
└── Window 3: logs     — Live log monitor (3-pane split)
                         ├── api.log (top)
                         ├── celery_worker.log (bottom-left)
                         └── celery_beat.log (bottom-right)
```

### tmux Keybindings

| Key | Action |
|-----|--------|
| `Ctrl-b 0`–`3` | Switch to window by number |
| `Ctrl-b n` / `p` | Next / previous window |
| `Ctrl-b d` | Detach (services keep running) |
| `Ctrl-b [` | Scroll mode (`q` to quit, arrows/PgUp/PgDn to scroll) |
| `Ctrl-b w` | Window list with preview |
| `Ctrl-b ,` | Rename current window |

### Log Rotation

Logs are handled at two levels:

1. **Python `RotatingFileHandler`** (automatic) — Configured in `src/logging_config.py`:
   - 10 MB max per log file
   - 5 backup files retained (`api.log`, `api.log.1`, …, `api.log.5`)
   - Applies to `logs/api.log`, `logs/celery_worker.log`, `logs/celery_beat.log`

2. **System `logrotate`** (optional) — Drop this into `/etc/logrotate.d/mobibox` for an extra safety net:
   ```
   /path/to/MobiBox_Server/logs/*.log {
       daily
       rotate 14
       maxsize 50M
       compress
       delaycompress
       missingok
       notifempty
       copytruncate
   }
   ```

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
| FastAPI | 8001 | Main API server (uvicorn) |
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
