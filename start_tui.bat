@echo off
chcp 65001 >nul
rem QLH TUI 管理菜单一键启动 (Windows 10+)
rem 自动完成:
rem   1. 检查 8000 端口后端是否已运行（QLH_BACKEND_PORT 可覆盖）
rem   2. 未运行则新开窗口启动 API 服务器 (src/api_server.py)
rem   3. 等待 /api/health 就绪后进入 TUI 管理菜单
rem 退出 TUI 后后端继续运行；停止后端: 关闭“QLH 后端 API”窗口或其中按 Ctrl+C
cd /d "%~dp0"
title QLH TUI 管理菜单

rem 选择 Python 命令（python 优先，缺失时回退 py -3）
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

rem ---- [1/3] 检查后端是否已运行 ----
netstat -ano | findstr ":%BACKEND_PORT%" | findstr "LISTENING" >nul 2>nul
if %errorlevel%==0 (
    echo [1/3] 检测到后端已在端口 %BACKEND_PORT% 运行，跳过启动。
) else (
    echo [1/3] 后端未运行，正在新窗口启动 API 服务器（port %BACKEND_PORT%）...
    rem 用 python src/api_server.py 启动（而非 -m uvicorn），使
    rem POST /api/system/shutdown 能触发跨平台优雅退出（资源清理）。
    start "QLH 后端 API" cmd /k "chcp 65001>nul && cd /d ""%~dp0"" && (if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat") && set QLH_BACKEND_PORT=%BACKEND_PORT% && %PYTHON_CMD% src\api_server.py"
)

rem ---- [2/3] 等待后端就绪（最多 120 秒）----
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

rem ---- [3/3] 启动 TUI ----
echo [3/3] 启动 TUI 管理菜单 ...
echo       退出 TUI 后后端继续运行（关闭“QLH 后端 API”窗口可停止后端）。
echo.
%PYTHON_CMD% src\tui_admin.py %*

if errorlevel 1 (
    echo.
    echo [提示] TUI 退出异常。若后端窗口已关闭，请重新运行本脚本。
    pause
)
