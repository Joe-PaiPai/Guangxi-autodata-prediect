$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "E:\power-trading-assistant-venv\Scripts\python.exe"

$existing = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
  $existing |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force }
}

$env:PYTHONPATH = Join-Path $ProjectRoot "backend"
Start-Process `
  -WindowStyle Hidden `
  -FilePath $PythonExe `
  -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
  -WorkingDirectory $ProjectRoot

Write-Host "Server started at http://127.0.0.1:8000/ using $PythonExe"
