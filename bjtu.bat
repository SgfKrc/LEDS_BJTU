@echo off
chcp 65001 >nul
rem ============================================================
rem  QLH global bjtu command (Windows)
rem
rem  Usage: bjtu [tui_admin.py args...]
rem
rem  One-click launch: start backend (if not running), wait until
rem  /api/health is ready, then enter TUI. Backend keeps running
rem  after TUI exits; stop it by closing the "QLH Backend API"
rem  window or pressing Ctrl+C inside it.
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

call start_tui.bat %*
