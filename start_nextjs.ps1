# CrediFlux Next.js App Startup Script
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  CrediFlux | KNN Loan Prediction & Intelligence Engine" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Start FastAPI ML Backend
Write-Host "`n[1/2] Launching FastAPI Backend Service on port 8000..." -ForegroundColor Yellow
$BackendProcess = Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn api:app --host 127.0.0.1 --port 8000" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

# 2. Launch Next.js Development Server
Write-Host "`n[2/2] Launching Next.js Web App on http://localhost:3000..." -ForegroundColor Green
Set-Location "$PSScriptRoot\frontend"
npm run dev -- -p 3000
