@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
echo ============================================
echo  QLH 边缘推理系统 — 集显版打包脚本
echo ============================================
echo.
cd /d "%~dp0\.."

REM ---- 检查 Python 环境 ----
echo [1/5] 检查 Python 环境...
set "PYTHON=%CD%\.venv-packaging\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo   [错误] 未找到项目打包虚拟环境: %PYTHON%
    echo   请先创建 .venv-packaging 并安装 packaging\requirements-cpu.txt
    pause
    exit /b 1
)
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo   [错误] 项目打包虚拟环境不可用
    pause
    exit /b 1
)
"%PYTHON%" --version
if not defined QLH_SIGNING_KEY (
    echo   [错误] 发布构建必须设置 QLH_SIGNING_KEY，禁止生成无签名安装清单。
    exit /b 1
)
if not exist "packaging\version.txt" (
    echo   [错误] 未找到 packaging\version.txt
    exit /b 1
)
set /p APP_VERSION=<"packaging\version.txt"

REM ---- 安装 CPU-only 依赖 ----
echo.
echo [2/5] 安装 CPU-only 依赖...
echo   提示: 如果已安装 CUDA 版 PyTorch，会被替换为 CPU 版
"%PYTHON%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
"%PYTHON%" -m pip install -r packaging\requirements-cpu.txt --quiet
echo   依赖安装完成。

REM ---- 构建前端 ----
echo.
echo [3/5] 构建 CyberGothic 产品前端...
cd frontend_cybergothic
if not exist "node_modules" (
    echo   安装 npm 依赖...
    call npm install
)
call npx vite build
cd ..
if not exist "frontend_cybergothic\dist\index.html" (
    echo   [错误] 前端构建失败！
    pause
    exit /b 1
)
echo   前端构建完成。

REM ---- 创建必要的目录 ----
echo.
echo [4/5] 准备打包目录...
if not exist "models\qwen-1_8b-chat" mkdir "models\qwen-1_8b-chat"
if not exist "logs" mkdir "logs"

REM ---- PyInstaller 打包 ----
echo.
echo [5/5] PyInstaller 打包...
echo   这可能需要 5-15 分钟，请耐心等待...
"%PYTHON%" -m pip install pyinstaller --quiet
REM ★ 从项目根目录运行，输出到 dist/QLH-Edge-Inference/
"%PYTHON%" -m PyInstaller packaging\qlh-cpu.spec --noconfirm
if errorlevel 1 exit /b 1

REM T9.6: build the companion console before signing the application tree.
"%PYTHON%" -m PyInstaller packaging\qlh-tui-chat.spec --noconfirm ^
    --distpath "dist\QLH-Edge-Inference" ^
    --workpath "build\qlh-tui-chat-cpu"
if errorlevel 1 exit /b 1
if not exist "dist\QLH-Edge-Inference\QLH-TUI-Chat\QLH-TUI-Chat.exe" exit /b 1

REM MODEL-TOOLS P0: build the model-tools CLI console into the same signed tree.
"%PYTHON%" -m PyInstaller packaging\qlh-model-tools.spec --noconfirm ^
    --distpath "dist\QLH-Edge-Inference" ^
    --workpath "build\qlh-model-tools-cpu"
if errorlevel 1 exit /b 1
if not exist "dist\QLH-Edge-Inference\QLH-Model-Tools\QLH-Model-Tools.exe" exit /b 1

if not exist "dist\QLH-Edge-Inference\docs" mkdir "dist\QLH-Edge-Inference\docs"
if not exist "dist\QLH-Edge-Inference\tools" mkdir "dist\QLH-Edge-Inference\tools"
copy /y "bjtu.bat" "dist\QLH-Edge-Inference\bjtu.bat" >nul
if errorlevel 1 exit /b 1
copy /y "packaging\model-tools.bat" "dist\QLH-Edge-Inference\model-tools.bat" >nul
if errorlevel 1 exit /b 1
copy /y "packaging\version.txt" "dist\QLH-Edge-Inference\version.txt" >nul
if errorlevel 1 exit /b 1
copy /y "README.md" "dist\QLH-Edge-Inference\docs\README.md" >nul
if errorlevel 1 exit /b 1
copy /y "docs\整体架构.md" "dist\QLH-Edge-Inference\docs\整体架构.md" >nul
if errorlevel 1 exit /b 1
copy /y "docs\模块接口说明.md" "dist\QLH-Edge-Inference\docs\模块接口说明.md" >nul
if errorlevel 1 exit /b 1
copy /y "docs\核心技术原理.md" "dist\QLH-Edge-Inference\docs\核心技术原理.md" >nul
if errorlevel 1 exit /b 1
copy /y "packaging\scripts\convert_to_gguf.py" "dist\QLH-Edge-Inference\tools\convert_to_gguf.py" >nul
if errorlevel 1 exit /b 1

if exist "dist\QLH-Edge-Inference\QLH-Edge-Inference.exe" (
    if not exist "dist\QLH-Edge-Inference\tools" mkdir "dist\QLH-Edge-Inference\tools"
    "%PYTHON%" -m PyInstaller packaging\qlh-data-retention.spec --noconfirm ^
        --distpath "dist\QLH-Edge-Inference\tools" ^
        --workpath "build\qlh-data-retention-cpu"
    if errorlevel 1 exit /b 1
    if not exist "dist\QLH-Edge-Inference\tools\QLH-Data-Retention.exe" exit /b 1
    "%PYTHON%" -m PyInstaller packaging\qlh-install-manifest.spec --noconfirm ^
        --distpath "dist\QLH-Edge-Inference\tools" ^
        --workpath "build\qlh-install-manifest-cpu"
    if errorlevel 1 exit /b 1
    if not exist "dist\QLH-Edge-Inference\tools\QLH-Install-Manifest.exe" exit /b 1
    xcopy /e /i /y "packaging\pubkeys" "dist\QLH-Edge-Inference\pubkeys" >nul
    if errorlevel 1 exit /b 1
    "%PYTHON%" packaging\install_manifest.py build ^
        --root "dist\QLH-Edge-Inference" ^
        --app-id qlh-edge-inference ^
        --version "!APP_VERSION!" ^
        --platform windows ^
        --variant cpu ^
        --package-kind application ^
        --key "!QLH_SIGNING_KEY!" ^
        --trusted-keys-dir "packaging\pubkeys"
    if errorlevel 1 exit /b 1
    echo.
    echo ============================================
    echo   Build complete.
    echo   Output: dist\QLH-Edge-Inference\
    echo   Application: QLH-Edge-Inference.exe
    echo   TUI chat: QLH-TUI-Chat\QLH-TUI-Chat.exe
    echo   Data retention: tools\QLH-Data-Retention.exe
    echo ============================================
) else (
    echo.
    echo ============================================
    echo   [错误] 打包失败！请检查上方日志。
    echo ============================================
)

endlocal
if not defined QLH_NONINTERACTIVE pause
exit /b 0
