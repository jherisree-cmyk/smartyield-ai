# Start SmartYield AI (API + built UI) on http://localhost:8010
# Usage (PowerShell, project root):  .\run_app.ps1
if (-not (Test-Path "frontend\dist\index.html")) {
    Push-Location frontend
    npm install
    npm run build
    Pop-Location
}
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8010
