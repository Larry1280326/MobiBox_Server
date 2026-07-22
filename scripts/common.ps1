<#
.SYNOPSIS
    Shared helper functions for MobiBox service management scripts.
.DESCRIPTION
    Dot-sourced by all MobiBox .ps1 scripts. Provides centralized:
    - Path discovery (PROJECT_ROOT, LOGS_DIR)
    - Terminal color constants
    - Conda environment management
    - Process lifecycle (start/stop/check via PID files)
    - Port checking
    - Service readiness polling
    - Interactive timeout prompt
.NOTES
    This module MUST be dot-sourced, not imported as a module:
        . "$PSScriptRoot\common.ps1"
#>

#Requires -Version 5.1

# ============================================
# Terminal Colors
# ============================================
$script:RED    = "Red"
$script:GREEN  = "Green"
$script:YELLOW = "Yellow"
$script:BLUE   = "Blue"
$script:CYAN   = "Cyan"
$script:NC     = $null  # placeholder; Write-Host resets per-call

# ============================================
# Path Discovery
# ============================================
$script:ProjectRoot = Split-Path -Parent $PSScriptRoot
$script:LogsDir     = Join-Path $script:ProjectRoot "logs"

# ============================================
# Conda Environment Management
# ============================================
$script:CondaEnvName = "Mobibox_backend"

function Test-CondaAvailable {
    [CmdletBinding()]
    param()
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    return ($null -ne $condaCmd)
}

function Initialize-CondaEnvironment {
    [CmdletBinding()]
    param()
    if (-not (Test-CondaAvailable)) {
        Write-Host "Error: conda is not installed or not in PATH" -ForegroundColor $RED
        throw "conda not found"
    }

    # Check if the environment exists
    $envList = conda env list 2>$null | Out-String
    if ($envList -notmatch [regex]::Escape($CondaEnvName)) {
        Write-Host "Error: conda environment '$CondaEnvName' not found" -ForegroundColor $RED
        Write-Host "Make sure you've created the environment:" -ForegroundColor $YELLOW
        Write-Host "  conda env create -f environment.yml"
        throw "conda environment not found"
    }

    # Activate the environment (harmless if already active)
    Write-Host "Activating conda environment '$CondaEnvName'..." -ForegroundColor $YELLOW
    try {
        conda activate $CondaEnvName 2>$null
    } catch {
        # If activate fails but env exists, warn but continue
        Write-Host "Warning: Could not activate conda environment, but it exists. Continuing..." -ForegroundColor $YELLOW
    }
}

# ============================================
# Docker Availability Check
# ============================================
function Test-DockerAvailable {
    [CmdletBinding()]
    param()
    try {
        $null = docker ps 2>&1
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Invoke-DockerCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [scriptblock]$Command,

        [string]$ErrorMessage = "Docker command failed. Is Docker Desktop running?"
    )
    try {
        $result = & $Command 2>&1
        if ($LASTEXITCODE -ne 0) {
            $errStr = "$result"
            if ($errStr -match "Cannot connect|error during connect|docker daemon|Is the docker") {
                Write-Host "Error: Cannot connect to Docker. Is Docker Desktop running?" -ForegroundColor $RED
            } else {
                Write-Host "Warning: $ErrorMessage" -ForegroundColor $YELLOW
            }
            return ""
        }
        return "$result"
    } catch {
        Write-Host "Warning: Docker not available. $ErrorMessage" -ForegroundColor $YELLOW
        return ""
    }
}

function Get-CondaRunPrefix {
    [CmdletBinding()]
    param()
    # Returns the argument list prefix for running commands in the conda environment.
    # Uses "conda run" which is more reliable than relying on shell state.
    return @("run", "--no-capture-output", "-n", $script:CondaEnvName)
}

# ============================================
# Environment File Validation
# ============================================
function Initialize-EnvFile {
    [CmdletBinding()]
    param()
    $envFile = Join-Path $script:ProjectRoot ".env"
    $envExample = Join-Path $script:ProjectRoot ".env.example"

    if (-not (Test-Path $envFile)) {
        Write-Host "Warning: .env file not found. Copying from .env.example..." -ForegroundColor $YELLOW
        if (Test-Path $envExample) {
            Copy-Item $envExample $envFile
            Write-Host "Please edit .env with your credentials before running again." -ForegroundColor $YELLOW
        } else {
            Write-Host "Error: .env.example not found. Please create .env manually." -ForegroundColor $RED
            throw ".env.example not found"
        }
    }
}

# ============================================
# Port Checking
# ============================================
function Test-PortInUse {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [int]$Port
    )
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conn) { return $true }

    # Fallback: try netstat (doesn't require admin)
    $netstat = & netstat -ano 2>$null | Select-String ":$Port " | Select-String "LISTENING"
    return ($null -ne $netstat)
}

# ============================================
# Process Management via PID Files
# ============================================
function Test-ProcessFromPidFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$PidFilePath
    )
    if (-not (Test-Path $PidFilePath)) { return $false }

    $rawPid = (Get-Content $PidFilePath -Raw).Trim()
    if ([string]::IsNullOrWhiteSpace($rawPid)) { return $false }

    try {
        $proc = Get-Process -Id ([int]$rawPid) -ErrorAction Stop
        return (-not $proc.HasExited)
    } catch {
        return $false
    }
}

function Stop-ProcessFromPidFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$PidFilePath,

        [Parameter(Mandatory=$true)]
        [string]$ServiceName
    )
    if (-not (Test-Path $PidFilePath)) {
        Write-Host "$ServiceName PID file not found - may not have been started by this script" -ForegroundColor $YELLOW
        return $false
    }

    $rawPid = (Get-Content $PidFilePath -Raw).Trim()
    if ([string]::IsNullOrWhiteSpace($rawPid)) {
        Write-Host "$ServiceName PID file is empty" -ForegroundColor $YELLOW
        return $false
    }

    $pidInt = [int]$rawPid

    # Check if process is still alive
    try {
        $proc = Get-Process -Id $pidInt -ErrorAction Stop
    } catch {
        Write-Host "$ServiceName (PID $pidInt) was not running" -ForegroundColor $YELLOW
        return $false
    }

    # Use taskkill /T to kill the entire process tree (handles child workers)
    Write-Host "Stopping $ServiceName (PID $pidInt)..." -ForegroundColor $YELLOW
    $result = & taskkill /PID $pidInt /T /F 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  $ServiceName stopped" -ForegroundColor $GREEN
        return $true
    } else {
        Write-Host "  Warning: Could not stop $ServiceName - $result" -ForegroundColor $YELLOW
        return $false
    }
}

# ============================================
# Process Detection via WMI (fallback for externally-started processes)
# ============================================
function Find-ProcessesByCommandLine {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$Pattern
    )
    try {
        $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
                 Where-Object { $_.CommandLine -match $Pattern }
        return @($procs)
    } catch {
        return @()
    }
}

# ============================================
# Service Readiness Polling
# ============================================
function Wait-ForService {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$Name,

        [Parameter(Mandatory=$true)]
        [int]$Port,

        [int]$MaxAttempts = 30
    )
    Write-Host "Waiting for $Name to be ready..." -ForegroundColor $YELLOW
    for ($i = 1; $i -le $MaxAttempts; $i++) {
        if (Test-PortInUse -Port $Port) {
            Write-Host "  $Name is ready" -ForegroundColor $GREEN
            return $true
        }
        Start-Sleep -Seconds 1
    }
    Write-Host "Error: $Name failed to start after $MaxAttempts seconds" -ForegroundColor $RED
    return $false
}

# ============================================
# Background Process Launcher
# ============================================
function Start-BackgroundService {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$ServiceName,

        [Parameter(Mandatory=$true)]
        [string]$PidFileName,

        [Parameter(Mandatory=$true)]
        [string[]]$ArgumentList,

        [string]$WorkingDirectory = $script:ProjectRoot
    )
    $pidFilePath = Join-Path $script:LogsDir $PidFileName

    Write-Host "Starting $ServiceName..." -ForegroundColor $YELLOW

    # Use Start-Process with -WindowStyle Hidden to run in background
    # conda run ensures the correct Python environment
    $proc = Start-Process -FilePath "conda" `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -PassThru

    # Write PID to file
    $proc.Id | Out-File -FilePath $pidFilePath -NoNewline

    return $proc
}

# ============================================
# Interactive Timeout Prompt
# ============================================
function Read-ChoiceWithTimeout {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$Prompt,

        [int]$TimeoutSeconds = 5,

        [string]$DefaultChoice = "N"
    )
    Write-Host "$Prompt (y/N) - Auto-selecting '$DefaultChoice' in $TimeoutSeconds seconds..." -ForegroundColor $YELLOW
    Write-Host "Press Y to confirm, or wait for timeout..." -ForegroundColor $YELLOW

    $endTime = (Get-Date).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $endTime) {
        if ($host.UI.RawUI.KeyAvailable) {
            $key = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            $char = $key.Character
            Write-Host ""  # newline after keypress
            return ($char -eq 'y' -or $char -eq 'Y')
        }
        Start-Sleep -Milliseconds 200
    }

    Write-Host ""  # newline after timeout
    Write-Host "Timeout reached, defaulting to '$DefaultChoice'" -ForegroundColor $YELLOW
    return ($DefaultChoice -eq 'Y')
}

# ============================================
# Banner Printer
# ============================================
function Write-Banner {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [string]$Title,

        [string]$Color = $BLUE
    )
    Write-Host "========================================" -ForegroundColor $Color
    Write-Host "  $Title" -ForegroundColor $Color
    Write-Host "========================================" -ForegroundColor $Color
}
