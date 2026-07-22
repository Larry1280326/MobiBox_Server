<#
.SYNOPSIS
    MobiBox Backend - Restart All Services
.DESCRIPTION
    Restarts all MobiBox backend services by stopping them, waiting 3 seconds,
    then starting them again.
.NOTES
    Windows PowerShell equivalent of restart_services.sh
#>

# Dot-source shared helpers
. "$PSScriptRoot\common.ps1"

$ScriptDir = $PSScriptRoot

Write-Host "Restarting MobiBox Backend services..." -ForegroundColor $BLUE
Write-Host ""

# Stop services
& "$ScriptDir\stop_services.ps1"

Write-Host ""
Write-Host "Waiting 3 seconds..." -ForegroundColor $YELLOW
Start-Sleep -Seconds 3

# Start services
& "$ScriptDir\start_services.ps1"
