<#
.SYNOPSIS
    MobiBox Backend - Start All Services
.DESCRIPTION
    Starts all required services for MobiBox backend:
    1. MongoDB (via Docker)
    2. RabbitMQ (via Docker)
    3. FastAPI server (uvicorn)
    4. Celery Worker
    5. Celery Beat (scheduler)
.NOTES
    Windows PowerShell equivalent of start_services.sh
    Requires: Docker Desktop, conda (with Mobibox_backend env), Python dependencies
#>

$ErrorActionPreference = "Stop"

# Dot-source shared helpers
. "$PSScriptRoot\common.ps1"

# Create logs directory
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

Write-Banner -Title "MobiBox Backend - Service Startup"

# =========================================
# Pre-flight Checks
# =========================================

# Check conda
if (-not (Test-CondaAvailable)) {
    Write-Host "Error: conda is not installed or not in PATH" -ForegroundColor $RED
    exit 1
}

# Activate conda environment (validates it exists)
try {
    Initialize-CondaEnvironment
} catch {
    exit 1
}

# Check for .env file
try {
    Initialize-EnvFile
} catch {
    exit 1
}

# =========================================
# Step 1: Check/Start MongoDB
# =========================================
Write-Host "Step 1: MongoDB (Database)" -ForegroundColor $BLUE

if (-not (Test-DockerAvailable)) {
    Write-Host "  Docker Desktop is not running. Skipping MongoDB." -ForegroundColor $YELLOW
    Write-Host "  Start Docker Desktop and re-run this script to start MongoDB." -ForegroundColor $YELLOW
} else {
    $mongoRunning = Invoke-DockerCommand -Command {
        docker ps --filter "name=mobibox-mongo" --format "{{.Names}}"
    } -ErrorMessage "Cannot check MongoDB status."

    if ($mongoRunning.Trim() -eq "mobibox-mongo") {
        Write-Host "  MongoDB container is already running" -ForegroundColor $GREEN
    } else {
        Write-Host "Starting MongoDB container..." -ForegroundColor $YELLOW
        $runResult = Invoke-DockerCommand -Command {
            docker run -d --name mobibox-mongo -p 27017:27017 -v mobibox_mongo_data:/data/db mongo:7
        } -ErrorMessage "Cannot create MongoDB container."

        if ($runResult.Trim() -eq "") {
            # Container may already exist, try starting it
            Write-Host "Container may already exist, starting it..." -ForegroundColor $YELLOW
            $startResult = Invoke-DockerCommand -Command {
                docker start mobibox-mongo
            } -ErrorMessage "Cannot start MongoDB container."
            if ($startResult.Trim() -eq "") {
                Write-Host "Warning: Failed to start MongoDB. Check Docker Desktop." -ForegroundColor $YELLOW
                Write-Host "Continuing without MongoDB — endpoints requiring DB will fail." -ForegroundColor $YELLOW
            } else {
                Write-Host "  MongoDB started" -ForegroundColor $GREEN
            }
        } else {
            Write-Host "  MongoDB started" -ForegroundColor $GREEN
        }
        Write-Host "Waiting for MongoDB to initialize..." -ForegroundColor $YELLOW
        Start-Sleep -Seconds 3
    }
}

# =========================================
# Step 2: Check/Start RabbitMQ
# =========================================
Write-Host "Step 2: RabbitMQ (Message Queue)" -ForegroundColor $BLUE

if (-not (Test-DockerAvailable)) {
    Write-Host "  Docker Desktop is not running. Skipping RabbitMQ." -ForegroundColor $YELLOW
    Write-Host "  Start Docker Desktop and re-run this script to start RabbitMQ." -ForegroundColor $YELLOW
} else {
    $rabbitRunning = Invoke-DockerCommand -Command {
        docker ps --filter "name=rabbitmq" --format "{{.Names}}"
    } -ErrorMessage "Cannot check RabbitMQ status."

    if ($rabbitRunning.Trim() -eq "rabbitmq") {
        Write-Host "  RabbitMQ container is already running" -ForegroundColor $GREEN
    } else {
        Write-Host "Starting RabbitMQ container..." -ForegroundColor $YELLOW
        $runResult = Invoke-DockerCommand -Command {
            docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
        } -ErrorMessage "Cannot create RabbitMQ container."

        if ($runResult.Trim() -eq "") {
            # Container may already exist, try starting it
            Write-Host "Container may already exist, starting it..." -ForegroundColor $YELLOW
            $startResult = Invoke-DockerCommand -Command {
                docker start rabbitmq
            } -ErrorMessage "Cannot start RabbitMQ container."
            if ($startResult.Trim() -eq "") {
                Write-Host "Error: Failed to start RabbitMQ. Check Docker Desktop." -ForegroundColor $RED
                Write-Host "Continuing without RabbitMQ..." -ForegroundColor $YELLOW
            } else {
                Write-Host "  RabbitMQ started" -ForegroundColor $GREEN
            }
        } else {
            Write-Host "  RabbitMQ started" -ForegroundColor $GREEN
        }
        Write-Host "Waiting for RabbitMQ to initialize..." -ForegroundColor $YELLOW
        Start-Sleep -Seconds 5
    }
}

# =========================================
# Step 3: Start FastAPI Server
# =========================================
Write-Host "Step 3: FastAPI Server" -ForegroundColor $BLUE

if (Test-PortInUse -Port 8000) {
    Write-Host "Port 8000 is already in use. Skipping FastAPI server startup." -ForegroundColor $YELLOW
    Write-Host "To restart, stop the process on port 8000 first." -ForegroundColor $YELLOW
} else {
    # Build argument list for: conda run -n Mobibox_backend python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
    $uvicornArgs = (Get-CondaRunPrefix) + @(
        "python", "-m", "uvicorn", "src.main:app",
        "--host", "0.0.0.0", "--port", "8000"
    )

    Write-Host "Starting FastAPI server..." -ForegroundColor $YELLOW
    $apiPidFile = "api.pid"
    $apiPidPath = Join-Path $LogsDir $apiPidFile

    $apiProc = Start-Process -FilePath "conda" `
        -ArgumentList $uvicornArgs `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru

    $apiProc.Id | Out-File -FilePath $apiPidPath -NoNewline

    if (-not (Wait-ForService -Name "FastAPI" -Port 8000)) {
        Write-Host "Warning: FastAPI may still be starting. Check logs at: $LogsDir\api.log" -ForegroundColor $YELLOW
    }
}

# =========================================
# Step 4: Start Celery Worker
# =========================================
Write-Host "Step 4: Celery Worker" -ForegroundColor $BLUE

$workerPidFile = Join-Path $LogsDir "celery_worker.pid"
$workerRunning = $false

# Check if already running via PID file
if (Test-ProcessFromPidFile -PidFilePath $workerPidFile) {
    $workerRunning = $true
}

# Fallback: check via WMI command-line inspection
if (-not $workerRunning) {
    $existingWorkers = Find-ProcessesByCommandLine -Pattern "celery.*worker"
    if ($existingWorkers.Count -gt 0) {
        $workerRunning = $true
    }
}

if ($workerRunning) {
    Write-Host "Celery worker is already running. Skipping." -ForegroundColor $YELLOW
    Write-Host "To restart, run: .\scripts\stop_services.ps1 then .\scripts\start_services.ps1" -ForegroundColor $YELLOW
} else {
    # Build argument list for: conda run -n Mobibox_backend celery -A src.celery_app.celery_app worker --loglevel=info
    $celeryWorkerArgs = (Get-CondaRunPrefix) + @(
        "celery", "-A", "src.celery_app.celery_app", "worker",
        "--loglevel=info"
    )

    Write-Host "Starting Celery worker..." -ForegroundColor $YELLOW
    $workerProc = Start-Process -FilePath "conda" `
        -ArgumentList $celeryWorkerArgs `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru

    $workerProc.Id | Out-File -FilePath $workerPidFile -NoNewline
    Start-Sleep -Seconds 3
    Write-Host "  Celery worker started" -ForegroundColor $GREEN
}

# =========================================
# Step 5: Start Celery Beat (Scheduler)
# =========================================
Write-Host "Step 5: Celery Beat (Scheduler)" -ForegroundColor $BLUE

$beatPidFile = Join-Path $LogsDir "celery_beat.pid"
$beatRunning = $false

# Check if already running via PID file
if (Test-ProcessFromPidFile -PidFilePath $beatPidFile) {
    $beatRunning = $true
}

# Fallback: check via WMI command-line inspection
if (-not $beatRunning) {
    $existingBeats = Find-ProcessesByCommandLine -Pattern "celery.*beat"
    if ($existingBeats.Count -gt 0) {
        $beatRunning = $true
    }
}

if ($beatRunning) {
    Write-Host "Celery beat is already running. Skipping." -ForegroundColor $YELLOW
    Write-Host "To restart, run: .\scripts\stop_services.ps1 then .\scripts\start_services.ps1" -ForegroundColor $YELLOW
} else {
    # Build argument list for: conda run -n Mobibox_backend celery -A src.celery_app.celery_app beat --loglevel=info
    $celeryBeatArgs = (Get-CondaRunPrefix) + @(
        "celery", "-A", "src.celery_app.celery_app", "beat",
        "--loglevel=info"
    )

    Write-Host "Starting Celery beat..." -ForegroundColor $YELLOW
    $beatProc = Start-Process -FilePath "conda" `
        -ArgumentList $celeryBeatArgs `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru

    $beatProc.Id | Out-File -FilePath $beatPidFile -NoNewline
    Start-Sleep -Seconds 2
    Write-Host "  Celery beat started" -ForegroundColor $GREEN
}

# =========================================
# Summary
# =========================================
Write-Host ""
Write-Host "========================================" -ForegroundColor $GREEN
Write-Host "  All services started successfully!" -ForegroundColor $GREEN
Write-Host "========================================" -ForegroundColor $GREEN
Write-Host ""
Write-Host "Service Status:" -ForegroundColor $BLUE
Write-Host "  MongoDB:        mongodb://localhost:27017" -ForegroundColor $GREEN
Write-Host "  RabbitMQ:       http://localhost:15672 (guest/guest)" -ForegroundColor $GREEN
Write-Host "  FastAPI:         http://localhost:8000" -ForegroundColor $GREEN
Write-Host "  Celery Worker:   Running" -ForegroundColor $GREEN
Write-Host "  Celery Beat:     Running" -ForegroundColor $GREEN
Write-Host ""
Write-Host "Logs Location:" -ForegroundColor $BLUE
Write-Host "  API Server:     $LogsDir\api.log"
Write-Host "  Celery Worker:  $LogsDir\celery_worker.log"
Write-Host "  Celery Beat:    $LogsDir\celery_beat.log"
Write-Host ""
Write-Host "PID Files:" -ForegroundColor $BLUE
Write-Host "  API Server:     $LogsDir\api.pid"
Write-Host "  Celery Worker:  $LogsDir\celery_worker.pid"
Write-Host "  Celery Beat:    $LogsDir\celery_beat.pid"
Write-Host ""
Write-Host "Quick Commands:" -ForegroundColor $BLUE
Write-Host "  View API logs:      Get-Content $LogsDir\api.log -Wait"
Write-Host "  View worker logs:   Get-Content $LogsDir\celery_worker.log -Wait"
Write-Host "  Stop all services:  .\scripts\stop_services.ps1"
Write-Host "  Check health:       Invoke-RestMethod http://localhost:8000/health"
Write-Host ""
