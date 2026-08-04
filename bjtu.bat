@echo off
chcp 65001 >nul
rem ============================================================
rem  QLH global bjtu command (Windows)
rem
rem  Usage: bjtu [tui_admin.py args...]
rem         bjtu --help       查看 TUI 命令集与启动参数（不启动后端）
rem
rem  One-click launch: start backend (if not running), wait until
rem  /api/health is ready, then enter TUI. Backend keeps running
rem  after TUI exits; stop it with /shutdown inside TUI, or by
rem  closing the "QLH Backend API" window / pressing Ctrl+C there.
rem
rem  Install: add this file's directory (project root) to PATH.
rem  This file MUST stay in the project root (same dir as src/).
rem ============================================================
cd /d "%~dp0"

if not exist "src\api_server.py" (
    echo [ERROR] bjtu.bat must be placed in the QLH project root.
    echo         Current dir: %~dp0
    pause
    exit /b 1
)

rem ---- help: 仅打印命令集与参数帮助，不启动后端 ----
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help

call start_tui.bat %*
exit /b %errorlevel%

:help
set "PYTHON_CMD=python"
where python >nul 2>nul
if not %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
)
set PYTHONIOENCODING=utf-8
%PYTHON_CMD% src\tui_admin.py --help
exit /b %errorlevel%
