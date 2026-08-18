@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title QLH Patch Listener

set "PYTHON_EXE="
if exist ".venv-packaging\Scripts\python.exe" set "PYTHON_EXE=.venv-packaging\Scripts\python.exe"
if not defined PYTHON_EXE where python >nul 2>nul && set "PYTHON_EXE=python"
if not defined PYTHON_EXE (
  echo [ERROR] Python was not found. Install Python 3.10+ or activate the project environment.
  pause
  exit /b 1
)

if not defined QLH_PATCH_VERIFY_KEY set "QLH_PATCH_VERIFY_KEY=packaging\pubkeys\release-20260809.pub.json"
if not defined QLH_PATCH_PORT set "QLH_PATCH_PORT=19731"
if not defined QLH_PATCH_BRANCH set "QLH_PATCH_BRANCH=dev"

echo [INFO] Listening on IPv4/IPv6 port %QLH_PATCH_PORT% (branch %QLH_PATCH_BRANCH%).
echo [INFO] Press Ctrl+C to stop. Node identity is stored outside this checkout.
"%PYTHON_EXE%" tools\patch_listener.py --verify-key "%QLH_PATCH_VERIFY_KEY%" --port %QLH_PATCH_PORT% --branch "%QLH_PATCH_BRANCH%"
set "RESULT=%ERRORLEVEL%"
echo.
echo [STOPPED] Patch listener returned %RESULT%.
pause
exit /b %RESULT%
