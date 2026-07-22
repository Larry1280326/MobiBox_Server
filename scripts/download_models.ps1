<#
.SYNOPSIS
    Pre-download sentence-transformers models for offline use.
.DESCRIPTION
    Downloads the required sentence-transformers models by calling the
    cross-platform Python script. Run this on a machine with internet access,
    then copy the cache to your server.
.PARAMETER Verify
    Instead of downloading, verify that all required models are already cached
    and working.
.NOTES
    Windows PowerShell equivalent of download_models.sh
.EXAMPLE
    .\scripts\download_models.ps1
    .\scripts\download_models.ps1 -Verify
#>

param(
    [switch]$Verify
)

# Dot-source shared helpers
. "$PSScriptRoot\common.ps1"

$ScriptDir = $PSScriptRoot
$PythonScript = Join-Path $ScriptDir "download_sentence_transformers.py"

Write-Banner -Title "MobiBox - Sentence-Transformers Model Download"

if ($Verify) {
    Write-Host "Verifying cached models..." -ForegroundColor $YELLOW
    & python "$PythonScript" --verify
} else {
    Write-Host "Downloading sentence-transformers models..." -ForegroundColor $YELLOW
    & python "$PythonScript"
}

Write-Host ""
Write-Host "Done!" -ForegroundColor $GREEN
