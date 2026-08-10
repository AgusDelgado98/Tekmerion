# Tekmérion portfolio demo (no secrets required)
# Usage:  .\scripts\run_demo.ps1
# Optional: .\scripts\run_demo.ps1 -NoFake

param(
    [switch]$NoFake
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:TEKMERION_DATA_MODE = "showroom"
if (-not $NoFake) {
    $env:TEKMERION_LLM_PROVIDER = "fake"
    Write-Host "Dataset: showroom | LLM: fake (deterministic DEMO, not a real model call)"
} else {
    if (-not $env:TEKMERION_LLM_PROVIDER) {
        $env:TEKMERION_LLM_PROVIDER = "disabled"
    }
    Write-Host "Dataset: showroom | LLM: $($env:TEKMERION_LLM_PROVIDER)"
}

Write-Host "Starting Flask at http://127.0.0.1:5000 ..."
python -m app
