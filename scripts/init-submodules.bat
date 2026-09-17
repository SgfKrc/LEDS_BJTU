@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

rem ============================================================
rem  QLH submodule init (recursive), passing the proxy to EVERY git child
rem
rem  Why: the proxy in this repo's .git/config is NOT inherited by nested
rem  submodule fetches, so a fresh clone fails on the inner repo
rem      android\app\src\main\cpp\llama.cpp
rem  with: fatal: Unable to find current revision in submodule path
rem
rem  Usage:
rem    scripts\init-submodules.bat                proxy from QLH_HTTP_PROXY / http_proxy / 127.0.0.1:7897
rem    scripts\init-submodules.bat --no-proxy     direct connection
rem    scripts\init-submodules.bat --check        print proxy + current submodule status only
rem ============================================================

set "USE_PROXY="
if defined QLH_HTTP_PROXY set "USE_PROXY=%QLH_HTTP_PROXY%"
if not defined USE_PROXY if defined http_proxy set "USE_PROXY=%http_proxy%"
if not defined USE_PROXY if defined HTTPS_PROXY set "USE_PROXY=%HTTPS_PROXY%"
if not defined USE_PROXY set "USE_PROXY=http://127.0.0.1:7897"
set "CHECK_ONLY=0"

if /i "%~1"=="--no-proxy" set "USE_PROXY="
if /i "%~1"=="--check" set "CHECK_ONLY=1"
if /i "%~2"=="--no-proxy" set "USE_PROXY="
if /i "%~2"=="--check" set "CHECK_ONLY=1"

if not defined USE_PROXY goto direct
set "http_proxy=%USE_PROXY%"
set "https_proxy=%USE_PROXY%"
set "HTTP_PROXY=%USE_PROXY%"
set "HTTPS_PROXY=%USE_PROXY%"
rem GIT_CONFIG_* are env vars inherited by git children, so nested fetches use the proxy too
set "GIT_CONFIG_COUNT=1"
set "GIT_CONFIG_KEY_0=http.proxy"
set "GIT_CONFIG_VALUE_0=%USE_PROXY%"
echo [init-submodules] proxy: %USE_PROXY%
goto status

:direct
echo [init-submodules] direct connection ^(--no-proxy^)

:status
echo [init-submodules] current submodule status:
git submodule status --recursive
if "%CHECK_ONLY%"=="1" exit /b 0

echo [init-submodules] sync + update ^(may take a few minutes^)...
git submodule sync --recursive
if errorlevel 1 exit /b 1
git submodule update --init --recursive
if errorlevel 1 exit /b 1

echo [init-submodules] done; status after update:
git submodule status --recursive
exit /b 0
