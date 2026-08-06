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

rem ---- chat: T9 简化聊天页（可选依赖 Textual/httpx）----
if /i "%~1"=="chat" goto :chat

call start_tui.bat %*
exit /b %errorlevel%

:chat
set "PYTHON_CMD=python"
if exist ".venv-tui\Scripts\python.exe" set "PYTHON_CMD=.venv-tui\Scripts\python.exe"
%PYTHON_CMD% -c "import textual, httpx" >nul 2>nul
if not %errorlevel%==0 (
    echo [T9] 聊天页缺少可选依赖 Textual/httpx。
    echo      安装: python scripts\setup_tui_env.py
    echo      或:   pip install -r packaging\requirements-tui.txt
    echo      管理 TUI（bjtu / start_tui.bat）不受影响。
    exit /b 2
)
set PYTHONIOENCODING=utf-8
%PYTHON_CMD% src\tui_chat.py %2 %3 %4 %5 %6
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
