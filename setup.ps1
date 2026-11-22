# Windows PowerShell setup: virtual environment, dependencies, .env template, offline tests.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python -m pip install -q --upgrade pip
.\.venv\Scripts\python -m pip install -q -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
$env:TRACE_MODE = "replay"
.\.venv\Scripts\python -m unittest discover -s tests
Write-Host "Setup done. Next: .\.venv\Scripts\Activate.ps1, then python eval/run_eval.py (replay) or python run.py `"question`""
