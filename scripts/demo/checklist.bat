@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0\..\.."
set "PYTHON_CMD=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
if exist ".venv-test\Scripts\python.exe" set "PYTHON_CMD=.venv-test\Scripts\python.exe"
%PYTHON_CMD% scripts\demo\defense_preflight.py %*
exit /b %errorlevel%
