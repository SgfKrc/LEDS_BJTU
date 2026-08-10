@echo off
chcp 65001 >nul
rem ============================================================
rem  QLH global bjtu command (Windows)
rem
rem  Usage: bjtu [launcher|ui|tui|chat|tui_admin.py args...]
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
    if exist "QLH-Edge-Inference.exe" goto :packaged_launcher
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
if /i "%~1"=="update" goto :update
if /i "%~1"=="version" goto :version
if /i "%~1"=="launcher-status" goto :launcher_n4
if /i "%~1"=="launcher-check" goto :launcher_n4
if /i "%~1"=="launcher-download" goto :launcher_n4
if /i "%~1"=="launcher-install" goto :launcher_n4
if /i "%~1"=="launcher-stage" goto :launcher_n4
if /i "%~1"=="launcher-activate" goto :launcher_n4
if /i "%~1"=="launcher-rollback" goto :launcher_n4
if /i "%~1"=="launcher-recover" goto :launcher_n4
if /i "%~1"=="diagnostics" goto :launcher_n4
if /i "%~1"=="verify" goto :launcher_n4
if /i "%~1"=="diagnose" goto :launcher_n4
if /i "%~1"=="repair" goto :launcher_n4

rem ---- unified launcher modes ----
if /i "%~1"=="launcher" goto :launcher
if /i "%~1"=="ui" goto :ui
if /i "%~1"=="tui" goto :tui

call start_tui.bat %*
exit /b %errorlevel%

:packaged_launcher
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help
if /i "%~1"=="chat" (
    echo [ERROR] 当前安装包未携带 Textual 聊天页，请使用项目环境中的 bjtu chat。
    exit /b 2
)
set "QLH_LAUNCHER_EXE="
if exist "QLH-Launcher.exe" set "QLH_LAUNCHER_EXE=%CD%\QLH-Launcher.exe"
if not defined QLH_LAUNCHER_EXE if exist "%LOCALAPPDATA%\Programs\QLH-Launcher\QLH-Launcher.exe" set "QLH_LAUNCHER_EXE=%LOCALAPPDATA%\Programs\QLH-Launcher\QLH-Launcher.exe"
if not defined QLH_LAUNCHER_EXE if exist "%ProgramFiles%\QLH-Launcher\QLH-Launcher.exe" set "QLH_LAUNCHER_EXE=%ProgramFiles%\QLH-Launcher\QLH-Launcher.exe"
if not defined QLH_LAUNCHER_EXE goto :legacy_packaged_launcher
if /i "%~1"=="update" (
    "%QLH_LAUNCHER_EXE%" check %2 %3 %4 %5 %6
    exit /b %errorlevel%
)
if /i "%~1"=="launcher-status" goto :packaged_launcher_n4
if /i "%~1"=="launcher-check" goto :packaged_launcher_n4
if /i "%~1"=="launcher-download" goto :packaged_launcher_n4
if /i "%~1"=="launcher-install" goto :packaged_launcher_n4
if /i "%~1"=="launcher-stage" goto :packaged_launcher_n4
if /i "%~1"=="launcher-activate" goto :packaged_launcher_n4
if /i "%~1"=="launcher-rollback" goto :packaged_launcher_n4
if /i "%~1"=="launcher-recover" goto :packaged_launcher_n4
if /i "%~1"=="diagnostics" goto :packaged_launcher_n4
if /i "%~1"=="verify" goto :packaged_launcher_n4
if /i "%~1"=="diagnose" goto :packaged_launcher_n4
if /i "%~1"=="repair" goto :packaged_launcher_n4
if /i "%~1"=="version" (
    if exist "version.txt" type version.txt
    if not exist "version.txt" echo unknown
    exit /b 0
)
if /i "%~1"=="tui" (
    "%QLH_LAUNCHER_EXE%" app-tui %2 %3 %4 %5 %6
    exit /b %errorlevel%
)
if /i "%~1"=="ui" (
    "%QLH_LAUNCHER_EXE%" app-ui %2 %3 %4 %5 %6
    exit /b %errorlevel%
)
"%QLH_LAUNCHER_EXE%" %2 %3 %4 %5 %6
exit /b %errorlevel%

:legacy_packaged_launcher
if /i "%~1"=="update" (
    echo [ERROR] 独立 QLH Launcher 尚未安装，无法使用 bjtu update。
    exit /b 2
)
if /i "%~1"=="verify" (
    if exist "tools\QLH-Install-Manifest.exe" (
        "tools\QLH-Install-Manifest.exe" verify --root "%CD%" %2 %3 %4 %5 %6
        exit /b %errorlevel%
    )
    echo [ERROR] 安装完整性校验器缺失；请覆盖安装匹配版本的主应用包。
    exit /b 2
)
if /i "%~1"=="diagnose" (
    echo [ERROR] 独立 QLH Launcher 尚未安装，无法使用 bjtu diagnose。
    exit /b 2
)
if /i "%~1"=="repair" (
    echo [ERROR] 独立 QLH Launcher 尚未安装，无法使用 bjtu repair。
    exit /b 2
)
if /i "%~1"=="version" (
    if exist "version.txt" type version.txt
    if not exist "version.txt" echo unknown
    exit /b 0
)
if /i "%~1"=="tui" (
    QLH-Edge-Inference.exe --tui %2 %3 %4 %5 %6
    exit /b %errorlevel%
)
if /i "%~1"=="ui" (
    QLH-Edge-Inference.exe --ui %2 %3 %4 %5 %6
    exit /b %errorlevel%
)
QLH-Edge-Inference.exe --tui %2 %3 %4 %5 %6
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

:launcher
set "PYTHON_CMD=python"
if exist ".venv-packaging\Scripts\python.exe" set "PYTHON_CMD=.venv-packaging\Scripts\python.exe"
%PYTHON_CMD% packaging\qlh_launcher.py --gui %2 %3 %4 %5 %6
exit /b %errorlevel%

:ui
set "PYTHON_CMD=python"
if exist ".venv-packaging\Scripts\python.exe" set "PYTHON_CMD=.venv-packaging\Scripts\python.exe"
%PYTHON_CMD% packaging\qlh_launcher.py app-ui %2 %3 %4 %5 %6
exit /b %errorlevel%

:tui
set "PYTHON_CMD=python"
if exist ".venv-packaging\Scripts\python.exe" set "PYTHON_CMD=.venv-packaging\Scripts\python.exe"
%PYTHON_CMD% packaging\qlh_launcher.py app-tui %2 %3 %4 %5 %6
exit /b %errorlevel%

:update
set "PYTHON_CMD=python"
if exist ".venv-packaging\Scripts\python.exe" set "PYTHON_CMD=.venv-packaging\Scripts\python.exe"
%PYTHON_CMD% packaging\qlh_launcher.py check %2 %3 %4 %5 %6
exit /b %errorlevel%

:launcher_n4
set "PYTHON_CMD=python"
if exist ".venv-packaging\Scripts\python.exe" set "PYTHON_CMD=.venv-packaging\Scripts\python.exe"
%PYTHON_CMD% packaging\qlh_launcher.py %*
exit /b %errorlevel%

:packaged_launcher_n4
"%QLH_LAUNCHER_EXE%" %*
exit /b %errorlevel%

:version
if exist "version.txt" type version.txt
if not exist "version.txt" echo launcher/app version is managed by the installed package.
exit /b 0

:help
echo QLH BJTU 统一入口:
echo   bjtu launcher   启动选择页（普通界面 / TUI）
echo   bjtu ui         直接启动普通 Web/原生界面
echo   bjtu tui        启动后端并进入 TUI 管理界面
echo   bjtu chat       进入终端对话页
echo   bjtu update     检查更新源中的匹配安装包
echo   bjtu version    显示当前应用版本
echo   bjtu verify [--level quick^|full^|deep] [--json]  校验已安装程序文件
echo   bjtu diagnose [--json]  输出只读故障诊断与人工处理建议
echo   bjtu repair [--json]    修复当前版本中损坏的签名程序文件
echo   bjtu status     执行 TUI 单命令（不自动启动后端）
echo.
set "PYTHON_CMD=python"
where python >nul 2>nul
if not %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
)
set PYTHONIOENCODING=utf-8
%PYTHON_CMD% src\tui_admin.py --help
exit /b %errorlevel%
