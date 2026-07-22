<#
.SYNOPSIS
    MobiBox Backend - Stop All Services
.DESCRIPTION
    Stops all running MobiBox backend services (Celery Beat, Celery Worker,
    FastAPI) and optionally RabbitMQ and MongoDB.
.NOTES
    Windows PowerShell equivalent of stop_services.sh
#>

# Dot-source shared helpers
. "$PSScriptRoot\common.ps1"

Write-Banner -Title "MobiBox Backend - Service Shutdown"

# =========================================
# Stop Celery Beat
# =========================================
Write-Host "Stopping Celery Beat..." -ForegroundColor $YELLOW
$beatPid = Join-Path $LogsDir "celery_beat.pid"
if (Test-ProcessFromPidFile -PidFilePath $beatPid) {
    Stop-ProcessFromPidFile -PidFilePath $beatPid -ServiceName "Celery Beat"
} else {
    Write-Host "Celery Beat was not running" -ForegroundColor $YELLOW
}

# =========================================
# Stop Celery Worker
# =========================================
Write-Host "Stopping Celery Worker..." -ForegroundColor $YELLOW
$workerPid = Join-Path $LogsDir "celery_worker.pid"
if (Test-ProcessFromPidFile -PidFilePath $workerPid) {
    Stop-ProcessFromPidFile -PidFilePath $workerPid -ServiceName "Celery Worker"
} else {
    Write-Host "Celery Worker was not running" -ForegroundColor $YELLOW
}

# =========================================
# Stop FastAPI Server
# =========================================
Write-Host "Stopping FastAPI Server..." -ForegroundColor $YELLOW
$apiPid = Join-Path $LogsDir "api.pid"
if (Test-ProcessFromPidFile -PidFilePath $apiPid) {
    Stop-ProcessFromPidFile -PidFilePath $apiPid -ServiceName "FastAPI Server"
} else {
    Write-Host "FastAPI Server was not running" -ForegroundColor $YELLOW
}

# =========================================
# Stop RabbitMQ (Optional)
# =========================================
Write-Host ""
$stopRabbit = Read-ChoiceWithTimeout -Prompt "Do you want to stop RabbitMQ?" -TimeoutSeconds 5 -DefaultChoice "N"
if ($stopRabbit) {
    if (-not (Test-DockerAvailable)) {
        Write-Host "  Docker Desktop not running. Cannot stop RabbitMQ." -ForegroundColor $YELLOW
    } else {
        Write-Host "Stopping RabbitMQ..." -ForegroundColor $YELLOW
        $stopResult = Invoke-DockerCommand -Command {
            docker stop rabbitmq
        } -ErrorMessage "Cannot stop RabbitMQ."
        if ($stopResult.Trim() -ne "") {
            Write-Host "  RabbitMQ stopped" -ForegroundColor $GREEN
        } else {
            Write-Host "  RabbitMQ container not found or not running" -ForegroundColor $YELLOW
        }
    }
} else {
    Write-Host "Keeping RabbitMQ running (use 'docker stop rabbitmq' to stop manually)" -ForegroundColor $YELLOW
}

# =========================================
# Stop MongoDB (Optional)
# =========================================
Write-Host ""
$stopMongo = Read-ChoiceWithTimeout -Prompt "Do you want to stop MongoDB?" -TimeoutSeconds 5 -DefaultChoice "N"
if ($stopMongo) {
    if (-not (Test-DockerAvailable)) {
        Write-Host "  Docker Desktop not running. Cannot stop MongoDB." -ForegroundColor $YELLOW
    } else {
        Write-Host "Stopping MongoDB..." -ForegroundColor $YELLOW
        $stopResult = Invoke-DockerCommand -Command {
            docker stop mobibox-mongo
        } -ErrorMessage "Cannot stop MongoDB."
        if ($stopResult.Trim() -ne "") {
            Write-Host "  MongoDB stopped" -ForegroundColor $GREEN
        } else {
            Write-Host "  MongoDB container not found or not running" -ForegroundColor $YELLOW
        }
    }
} else {
    Write-Host "Keeping MongoDB running (use 'docker stop mobibox-mongo' to stop manually)" -ForegroundColor $YELLOW
}

# =========================================
# Clean up PID files
# =========================================
Write-Host "Cleaning up PID files..." -ForegroundColor $YELLOW
Remove-Item (Join-Path $LogsDir "*.pid") -Force -ErrorAction SilentlyContinue

# =========================================
# Summary
# =========================================
Write-Host ""
Write-Host "========================================" -ForegroundColor $GREEN
Write-Host "  All services stopped" -ForegroundColor $GREEN
Write-Host "========================================" -ForegroundColor $GREEN
Write-Host ""
Write-Host "To restart services:" -ForegroundColor $BLUE
Write-Host "  .\scripts\start_services.ps1" -ForegroundColor $BLUE
Write-Host ""
