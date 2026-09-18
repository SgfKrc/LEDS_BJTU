@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
echo ============================================
echo  QLH edge inference backend service
echo ============================================
echo.
cd /d "%~dp0"
echo [1/2] Activate Python environment...
call .venv\Scripts\activate.bat 2>nul || echo   (using system Python)
echo [2/2] Start API server (port 8000)...
echo   Open http://localhost:8000 when the optional frontend is built.
echo   Otherwise run: cd frontend ^&^& npm run dev
echo.
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
pause
