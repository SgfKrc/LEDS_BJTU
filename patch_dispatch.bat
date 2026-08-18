@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title QLH Patch Dispatch

set "PYTHON_EXE="
if exist ".venv-packaging\Scripts\python.exe" set "PYTHON_EXE=.venv-packaging\Scripts\python.exe"
if not defined PYTHON_EXE where python >nul 2>nul && set "PYTHON_EXE=python"
if not defined PYTHON_EXE (
  echo [ERROR] Python was not found. Install Python 3.10+ or activate the project environment.
  pause
  exit /b 1
)

if not defined QLH_PATCH_PROXY_PORT set "QLH_PATCH_PROXY_PORT=7897"
if not defined QLH_PATCH_NODES set /p "QLH_PATCH_NODES=Worker addresses (comma-separated, blank for commit/push only): "
if not defined PATCH_MESSAGE set /p "PATCH_MESSAGE=Patch commit message: "
if not defined PATCH_MESSAGE (
  echo [CANCELLED] Commit message is required.
  pause
  exit /b 2
)
if not defined QLH_PATCH_PATHS set /p "QLH_PATCH_PATHS=Optional paths (comma-separated, blank for all tracked changes): "

set "PATCH_ARGS=--proxy-port %QLH_PATCH_PROXY_PORT%"
if defined QLH_PATCH_NODES set "PATCH_ARGS=%PATCH_ARGS% --nodes %QLH_PATCH_NODES%"
if defined QLH_PATCH_PATHS set "PATCH_ARGS=%PATCH_ARGS% --paths %QLH_PATCH_PATHS%"
if defined QLH_PATCH_DRY_RUN set "PATCH_ARGS=%PATCH_ARGS% --dry-run"
if defined QLH_PATCH_NO_PUSH set "PATCH_ARGS=%PATCH_ARGS% --no-push"

echo.
echo [INFO] Dispatching to branch dev with session proxy 127.0.0.1:%QLH_PATCH_PROXY_PORT%.
"%PYTHON_EXE%" tools\patch_dispatch.py -m "%PATCH_MESSAGE%" %PATCH_ARGS%
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo [FAILED] Patch dispatch returned %RESULT%.
if "%RESULT%"=="0" echo [OK] Patch dispatch completed.
pause
exit /b %RESULT%
