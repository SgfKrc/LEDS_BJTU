@echo off
chcp 65001 >nul
cd /d "%~dp0\..\.."
set "PYTHON_CMD=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
%PYTHON_CMD% scripts\demo\defense_reset.py %*
exit /b %errorlevel%
