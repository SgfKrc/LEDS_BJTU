@echo off
chcp 65001 >nul
rem QLH TUI one-click launcher (Windows 10+)
rem
rem Interactive mode now goes through the unified `qlh` entry:
rem   1. the backend starts IN-PROCESS (BackendSupervisor) - no separate window;
rem   2. the cold start is carried by the Koakuma splash (animation + live status),
rem      so there is no silent multi-minute black screen while waiting for /api/health;
rem   3. the backend stops when the TUI exits (run `python src\api_server.py` to keep it up).
rem Single-command mode (e.g. `start_tui.bat status`) runs one TUI command and exits.
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

rem ---- single-command mode: run one TUI command then exit; never starts the backend.
rem       cmd name list = the read-only commands in src/tui_commands.py.
rem       NOTE: the command must be the FIRST argument (start_tui.bat status --port 9000);
rem       options first (start_tui.bat --port 9000 status) fall back to interactive mode.
set "FIRST_ARG=%~1"
set "SINGLE_CMD="
if defined FIRST_ARG (
    if "%FIRST_ARG:~0,1%"=="/" set "SINGLE_CMD=1"
)
if defined FIRST_ARG if not defined SINGLE_CMD (
    for %%c in (help h quit q exit shutdown halt status st screen goto refresh r model models switch load quant engine presets gpu device nodes connect join dist queue logs log host interval timeout token chat new sessions resume rename delete-session route thinking cancel) do (
        if /i "%FIRST_ARG%"=="%%c" set "SINGLE_CMD=1"
    )
)
if not defined SINGLE_CMD goto interactive_mode

%PYTHON_CMD% src\tui_commands.py %*
exit /b %errorlevel%

:interactive_mode
rem ---- unified entry: in-process backend + splash (no separate window, no silent wait) ----
set "PORT_ARGS="
if defined QLH_BACKEND_PORT set "PORT_ARGS=--port %BACKEND_PORT%"
%PYTHON_CMD% qlh.py %PORT_ARGS% %*
set "RC=%errorlevel%"
if "%RC%"=="0" exit /b 0
echo.
echo [提示] TUI 退出异常（exit %RC%），请检查上方输出。
pause
exit /b %RC%
