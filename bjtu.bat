@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
rem ============================================================
rem  QLH global bjtu command (Windows)
rem
rem  Usage: bjtu [launcher|ui|tui|chat|tui_commands.py args...]
rem         bjtu --help       print TUI commands and options (no backend)
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

if not defined QLH_CORE_ROOT set "QLH_CORE_ROOT=%~dp0"
if not defined QLH_SHELL_ROOT set "QLH_SHELL_ROOT=%~dp0..\qlh-shell"
if not defined QLH_RELEASE_ROOT set "QLH_RELEASE_ROOT=%~dp0..\qlh-release"
set "QLH_RELEASE_LAUNCHER=%QLH_RELEASE_ROOT%\packaging\qlh_launcher.py"

if not exist "src\api_server.py" (
    if exist "QLH-Edge-Inference.exe" goto :packaged_launcher
    echo [ERROR] bjtu.bat must be placed in the QLH project root.
    echo         Current dir: %~dp0
    pause
    exit /b 1
)

rem ---- help: print commands and options; do not start backend ----
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help

rem ---- chat: T9 terminal chat page (optional Textual/httpx) ----
rem ---- aliases --chat / -chat / -c are normalized to chat ----
if /i "%~1"=="--chat" goto :chat
if /i "%~1"=="-chat" goto :chat
if /i "%~1"=="-c" goto :chat
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
if /i "%~1"=="data-status" goto :launcher_n4
if /i "%~1"=="retain-data" goto :launcher_n4
if /i "%~1"=="reassociate-data" goto :launcher_n4
if /i "%~1"=="reinstall" goto :launcher_n4

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
    if not exist "%~dp0QLH-TUI-Chat\QLH-TUI-Chat.exe" (
        echo [ERROR] Installation is incomplete: QLH-TUI-Chat\QLH-TUI-Chat.exe is missing.
        echo         Run bjtu verify --level deep or repair with a trusted installer.
        exit /b 2
    )
    "%~dp0QLH-TUI-Chat\QLH-TUI-Chat.exe" %2 %3 %4 %5 %6
    exit /b %errorlevel%
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
if /i "%~1"=="data-status" goto :packaged_launcher_n4
if /i "%~1"=="retain-data" goto :packaged_launcher_n4
if /i "%~1"=="reassociate-data" goto :packaged_launcher_n4
if /i "%~1"=="reinstall" goto :packaged_launcher_n4
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
    echo [ERROR] Standalone QLH Launcher is not installed; bjtu update is unavailable.
    exit /b 2
)
if /i "%~1"=="verify" (
    if exist "tools\QLH-Install-Manifest.exe" (
        "tools\QLH-Install-Manifest.exe" verify --root "%CD%" %2 %3 %4 %5 %6
        exit /b %errorlevel%
    )
    echo [ERROR] Install integrity verifier is missing; reinstall the matching application package.
    exit /b 2
)
if /i "%~1"=="diagnose" (
    echo [ERROR] Standalone QLH Launcher is not installed; bjtu diagnose is unavailable.
    exit /b 2
)
if /i "%~1"=="repair" (
    echo [ERROR] Standalone QLH Launcher is not installed; bjtu repair is unavailable.
    exit /b 2
)
if /i "%~1"=="data-status" (
    echo [ERROR] Standalone QLH Launcher is not installed; bjtu data-status is unavailable.
    exit /b 2
)
if /i "%~1"=="retain-data" (
    echo [ERROR] Standalone QLH Launcher is not installed; bjtu retain-data is unavailable.
    exit /b 2
)
if /i "%~1"=="reassociate-data" (
    echo [ERROR] Standalone QLH Launcher is not installed; bjtu reassociate-data is unavailable.
    exit /b 2
)
if /i "%~1"=="reinstall" (
    echo [ERROR] Standalone QLH Launcher is not installed; bjtu reinstall is unavailable.
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
if exist "%QLH_SHELL_ROOT%\.venv-tui\Scripts\python.exe" set "PYTHON_CMD=%QLH_SHELL_ROOT%\.venv-tui\Scripts\python.exe"
%PYTHON_CMD% -c "import textual, httpx" >nul 2>nul
if not %errorlevel%==0 (
    echo [T9] Optional Textual/httpx dependencies are missing for chat.
    echo      Install: python "%QLH_SHELL_ROOT%\scripts\setup_tui_env.py"
    echo      Or:      pip install -r "%QLH_SHELL_ROOT%\requirements-tui.txt"
    echo      The management TUI (bjtu / start_tui.bat) is unaffected.
    exit /b 2
)
set PYTHONIOENCODING=utf-8
%PYTHON_CMD% src\tui_chat.py %2 %3 %4 %5 %6
exit /b %errorlevel%

:launcher
set "PYTHON_CMD=python"
if exist "%QLH_RELEASE_ROOT%\.venv-packaging\Scripts\python.exe" set "PYTHON_CMD=%QLH_RELEASE_ROOT%\.venv-packaging\Scripts\python.exe"
%PYTHON_CMD% "%QLH_RELEASE_LAUNCHER%" --gui %2 %3 %4 %5 %6
exit /b %errorlevel%

:ui
set "PYTHON_CMD=python"
if exist "%QLH_RELEASE_ROOT%\.venv-packaging\Scripts\python.exe" set "PYTHON_CMD=%QLH_RELEASE_ROOT%\.venv-packaging\Scripts\python.exe"
%PYTHON_CMD% "%QLH_RELEASE_LAUNCHER%" app-ui %2 %3 %4 %5 %6
exit /b %errorlevel%

:tui
set "PYTHON_CMD=python"
if exist "%QLH_RELEASE_ROOT%\.venv-packaging\Scripts\python.exe" set "PYTHON_CMD=%QLH_RELEASE_ROOT%\.venv-packaging\Scripts\python.exe"
%PYTHON_CMD% "%QLH_RELEASE_LAUNCHER%" app-tui %2 %3 %4 %5 %6
exit /b %errorlevel%

:update
set "PYTHON_CMD=python"
if exist "%QLH_RELEASE_ROOT%\.venv-packaging\Scripts\python.exe" set "PYTHON_CMD=%QLH_RELEASE_ROOT%\.venv-packaging\Scripts\python.exe"
%PYTHON_CMD% "%QLH_RELEASE_LAUNCHER%" check %2 %3 %4 %5 %6
exit /b %errorlevel%

:launcher_n4
set "PYTHON_CMD=python"
if exist "%QLH_RELEASE_ROOT%\.venv-packaging\Scripts\python.exe" set "PYTHON_CMD=%QLH_RELEASE_ROOT%\.venv-packaging\Scripts\python.exe"
%PYTHON_CMD% "%QLH_RELEASE_LAUNCHER%" %*
exit /b %errorlevel%

:packaged_launcher_n4
"%QLH_LAUNCHER_EXE%" %*
exit /b %errorlevel%

:version
if exist "version.txt" type version.txt
if not exist "version.txt" echo launcher/app version is managed by the installed package.
exit /b 0

:help
echo QLH BJTU unified entry:
echo   bjtu launcher   open the launcher (regular UI / TUI)
echo   bjtu ui         start the regular Web/native UI
echo   bjtu tui        start the backend and enter the management TUI
echo   bjtu chat       enter the terminal chat page (packaged builds)
echo   bjtu update     check the update source
echo   bjtu version    show the application version
echo   bjtu verify [--level quick^|full^|deep] [--json]  verify installed files
echo   bjtu diagnose [--json]  print read-only diagnostics
echo   bjtu repair [--json]    repair signed application files
echo   bjtu retain-data --yes  retain user data before uninstall
echo   bjtu reassociate-data --yes  re-associate retained data
echo   bjtu reinstall --yes    retain data and download a verified package
echo   bjtu status     run a TUI read-only command (no backend start)
echo.
set "PYTHON_CMD=python"
where python >nul 2>nul
if not %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
)
set PYTHONIOENCODING=utf-8
%PYTHON_CMD% src\tui_commands.py help
exit /b %errorlevel%
