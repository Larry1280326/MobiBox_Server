<#
.SYNOPSIS
    MobiBox Backend - Check Service Status
.DESCRIPTION
    Checks the status of all MobiBox backend services:
    MongoDB, RabbitMQ, FastAPI, Celery Worker, Celery Beat, TSFM model, and port usage.
.NOTES
    Windows PowerShell equivalent of status.sh
#>

# Dot-source shared helpers
. "$PSScriptRoot\common.ps1"

Write-Banner -Title "MobiBox Backend - Service Status"

# =========================================
# Check MongoDB
# =========================================
Write-Host "MongoDB:" -ForegroundColor $BLUE
if (-not (Test-DockerAvailable)) {
    Write-Host "  Docker Desktop not running - cannot check MongoDB" -ForegroundColor $YELLOW
} else {
    $mongoRunning = Invoke-DockerCommand -Command {
        docker ps --filter "name=mobibox-mongo" --format "{{.Names}}"
    } -ErrorMessage "Cannot check MongoDB."
    if ($mongoRunning.Trim() -eq "mobibox-mongo") {
        Write-Host "  Running (Docker container)" -ForegroundColor $GREEN
        Write-Host "    URL: mongodb://localhost:27017"
    } else {
        $mongoExists = Invoke-DockerCommand -Command {
            docker ps -a --filter "name=mobibox-mongo" --format "{{.Names}}"
        } -ErrorMessage "Cannot check MongoDB status."
        if ($mongoExists.Trim() -eq "mobibox-mongo") {
            Write-Host "  Stopped (container exists)" -ForegroundColor $YELLOW
        } else {
            Write-Host "  Not found" -ForegroundColor $RED
        }
    }
}
Write-Host ""

# =========================================
# Check RabbitMQ
# =========================================
Write-Host "RabbitMQ:" -ForegroundColor $BLUE
if (-not (Test-DockerAvailable)) {
    Write-Host "  Docker Desktop not running - cannot check RabbitMQ" -ForegroundColor $YELLOW
} else {
    $rabbitRunning = Invoke-DockerCommand -Command {
        docker ps --filter "name=rabbitmq" --format "{{.Names}}"
    } -ErrorMessage "Cannot check RabbitMQ."
    if ($rabbitRunning.Trim() -eq "rabbitmq") {
        Write-Host "  Running (Docker container)" -ForegroundColor $GREEN
        Write-Host "    Web UI: http://localhost:15672 (guest/guest)"
    } else {
        $rabbitExists = Invoke-DockerCommand -Command {
            docker ps -a --filter "name=rabbitmq" --format "{{.Names}}"
        } -ErrorMessage "Cannot check RabbitMQ status."
        if ($rabbitExists.Trim() -eq "rabbitmq") {
            Write-Host "  Stopped (container exists)" -ForegroundColor $YELLOW
        } else {
            Write-Host "  Not found" -ForegroundColor $RED
        }
    }
}
Write-Host ""

# =========================================
# Check FastAPI
# =========================================
Write-Host "FastAPI Server:" -ForegroundColor $BLUE
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "  Running" -ForegroundColor $GREEN
    Write-Host "    URL: http://localhost:8000"
    Write-Host "    Health: $($health | ConvertTo-Json -Compress)"
} catch {
    # Check if process exists but not responding
    $apiPidFile = Join-Path $LogsDir "api.pid"
    if (Test-ProcessFromPidFile -PidFilePath $apiPidFile) {
        Write-Host "  Process running but not responding" -ForegroundColor $YELLOW
    } else {
        Write-Host "  Not running" -ForegroundColor $RED
    }
}
Write-Host ""

# =========================================
# Check Celery Worker
# =========================================
Write-Host "Celery Worker:" -ForegroundColor $BLUE
$workerPidFile = Join-Path $LogsDir "celery_worker.pid"
$workerRunning = Test-ProcessFromPidFile -PidFilePath $workerPidFile

# Fallback: WMI
if (-not $workerRunning) {
    $workers = Find-ProcessesByCommandLine -Pattern "celery.*worker"
    $workerRunning = ($workers.Count -gt 0)
}

if ($workerRunning) {
    Write-Host "  Running" -ForegroundColor $GREEN
    Write-Host "    Inspect registered tasks:"
    Write-Host "    conda run -n $CondaEnvName celery -A src.celery_app.celery_app inspect registered"
} else {
    Write-Host "  Not running" -ForegroundColor $RED
}
Write-Host ""

# =========================================
# Check Celery Beat
# =========================================
Write-Host "Celery Beat:" -ForegroundColor $BLUE
$beatPidFile = Join-Path $LogsDir "celery_beat.pid"
$beatRunning = Test-ProcessFromPidFile -PidFilePath $beatPidFile

# Fallback: WMI
if (-not $beatRunning) {
    $beats = Find-ProcessesByCommandLine -Pattern "celery.*beat"
    $beatRunning = ($beats.Count -gt 0)
}

if ($beatRunning) {
    Write-Host "  Running" -ForegroundColor $GREEN
} else {
    Write-Host "  Not running" -ForegroundColor $RED
}
Write-Host ""

# =========================================
# Check TSFM Model
# =========================================
Write-Host "TSFM Model:" -ForegroundColor $BLUE
$tsfmCheckpoint = Join-Path $ProjectRoot "src\celery_app\services\tsfm_model\ckpts\best.pt"
if (Test-Path $tsfmCheckpoint) {
    $sizeBytes = (Get-Item $tsfmCheckpoint).Length
    if ($sizeBytes -gt 1GB) {
        $sizeStr = "{0:N2} GB" -f ($sizeBytes / 1GB)
    } elseif ($sizeBytes -gt 1MB) {
        $sizeStr = "{0:N2} MB" -f ($sizeBytes / 1MB)
    } else {
        $sizeStr = "{0:N2} KB" -f ($sizeBytes / 1KB)
    }
    Write-Host "  Checkpoint found ($sizeStr)" -ForegroundColor $GREEN
} else {
    Write-Host "  Checkpoint not found" -ForegroundColor $YELLOW
    Write-Host "    Download from remote server to: $tsfmCheckpoint"
}
Write-Host ""

# =========================================
# Check Ports
# =========================================
Write-Host "Port Usage:" -ForegroundColor $BLUE
$port27017 = Test-PortInUse -Port 27017
$port5672 = Test-PortInUse -Port 5672
$port8000 = Test-PortInUse -Port 8000
Write-Host "  27017 (MongoDB):     " -NoNewline
if ($port27017) { Write-Host "In Use" -ForegroundColor $GREEN } else { Write-Host "Free" -ForegroundColor $RED }
Write-Host "  5672  (RabbitMQ):    " -NoNewline
if ($port5672) { Write-Host "In Use" -ForegroundColor $GREEN } else { Write-Host "Free" -ForegroundColor $RED }
Write-Host "  8000  (FastAPI):     " -NoNewline
if ($port8000) { Write-Host "In Use" -ForegroundColor $GREEN } else { Write-Host "Free" -ForegroundColor $RED }
Write-Host ""

# =========================================
# Quick Commands
# =========================================
Write-Host "========================================" -ForegroundColor $BLUE
Write-Host "  Quick Commands" -ForegroundColor $BLUE
Write-Host "========================================" -ForegroundColor $BLUE
Write-Host "  Start all:    .\scripts\start_services.ps1"
Write-Host "  Stop all:     .\scripts\stop_services.ps1"
Write-Host "  Restart all:  .\scripts\restart_services.ps1"
Write-Host "  View logs:    Get-Content logs\api.log -Wait"
Write-Host "  Test API:     Invoke-RestMethod http://localhost:8000/health"
Write-Host ""
