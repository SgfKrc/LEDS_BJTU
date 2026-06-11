@echo off
chcp 65001 >nul
echo ============================================
echo  边缘推理优化系统 — 后端服务
echo ============================================
echo.
cd /d "%~dp0"
echo [1/2] 激活 Python 环境...
call .venv\Scripts\activate.bat 2>nul || echo   (使用系统 Python)
echo [2/2] 启动 API 服务器 (port 8000)...
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
pause
