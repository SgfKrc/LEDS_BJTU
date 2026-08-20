@echo off
chcp 65001 >nul
rem QLH TUI admin one-click launcher (Windows 10+)
rem It will:
rem   1. check whether the backend is already running on port 8000 (QLH_BACKEND_PORT overrides)
rem   2. if not running, start API server (src/api_server.py) in a new window
rem   3. wait for /api/health, then enter the TUI admin menu
rem Backend keeps running after TUI exits; stop it by closing the "QLH Backend API" window or pressing Ctrl+C there
cd /d "%~dp0"
title QLH TUI 管理菜单

rem pick python command (python first, fallback to py -3)
set "PYTHON_CMD=python"
where python >nul 2>nul
if not %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
)

if not "%QLH_BACKEND_PORT%"=="" (
    set "BACKEND_PORT=%QLH_BACKEND_PORT%"
) else (
    set "BACKEND_PORT=8000"
)

echo ============================================
echo   QLH 分布式边缘推理 — TUI 管理菜单
echo ============================================
echo.

rem ---- single-command mode: bjtu shutdown / bjtu /shutdown / bjtu status ...
rem       run one TUI command then exit; never starts the backend.
rem       cmd name list must stay in sync with COMMANDS in src/tui_admin.py.
rem       NOTE: the command must be the FIRST argument (bjtu status --port 9000);
rem       options first (bjtu --port 9000 status) fall back to interactive mode.
set "FIRST_ARG=%~1"
set "SINGLE_CMD="
if defined FIRST_ARG (
    if "%FIRST_ARG:~0,1%"=="/" set "SINGLE_CMD=1"
)
if defined FIRST_ARG if not defined SINGLE_CMD (
    for %%c in (help h quit q exit shutdown halt status st screen goto refresh r model models switch load quant engine presets gpu device nodes connect join dist queue logs log host interval timeout token chat cancel) do (
        if /i "%FIRST_ARG%"=="%%c" set "SINGLE_CMD=1"
    )
)
if not defined SINGLE_CMD goto :interactive_mode
%PYTHON_CMD% src\tui_admin.py %*
exit /b %errorlevel%
:interactive_mode

rem ---- [1/3] check if backend is already running ----
netstat -ano | findstr ":%BACKEND_PORT%" | findstr "LISTENING" >nul 2>nul
if %errorlevel%==0 (
    echo [1/3] 检测到后端已在端口 %BACKEND_PORT% 运行，跳过启动。
) else (
    echo [1/3] 后端未运行，正在新窗口启动 API 服务器（port %BACKEND_PORT%）...
    rem use python src/api_server.py (not -m uvicorn) so that
    rem POST /api/system/shutdown triggers cross-platform graceful exit (resource cleanup).
    start "QLH 后端 API" cmd /k "chcp 65001>nul && cd /d ""%~dp0"" && (if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat") && set QLH_BACKEND_PORT=%BACKEND_PORT% && %PYTHON_CMD% src\api_server.py"
)

rem ---- [2/3] wait for backend readiness (max 120s) ----
echo [2/3] 等待后端就绪（/api/health）...
set /a TRIES=0
:wait_health
%PYTHON_CMD% -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:%BACKEND_PORT%/api/health', timeout=2); sys.exit(0)" >nul 2>nul
if %errorlevel%==0 goto backend_ready
set /a TRIES+=1
if %TRIES% geq 60 (
    echo.
    echo [提示] 后端在 120 秒内未就绪。请检查“QLH 后端 API”窗口日志：
    echo        端口占用、Python 环境、依赖缺失等均会导致启动失败。
    pause
    exit /b 1
)
%PYTHON_CMD% -c "import time; time.sleep(2)" >nul 2>nul
goto wait_health

:backend_ready
echo       后端已就绪。

rem ---- [3/3] launch TUI ----
echo [3/3] 启动 TUI 管理菜单 ...
echo       退出 TUI 后后端继续运行（关闭“QLH 后端 API”窗口可停止后端）。
echo.
%PYTHON_CMD% src\tui_admin.py %*

if errorlevel 1 (
    echo.
    echo [提示] TUI 退出异常。若后端窗口已关闭，请重新运行本脚本。
    pause
)
