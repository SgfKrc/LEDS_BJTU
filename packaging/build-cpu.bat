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
python --version >nul 2>&1
if errorlevel 1 (
    echo   [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)
python --version

REM ---- 安装 CPU-only 依赖 ----
echo.
echo [2/5] 安装 CPU-only 依赖...
echo   提示: 如果已安装 CUDA 版 PyTorch，会被替换为 CPU 版
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
pip install -r packaging\requirements-cpu.txt --quiet
echo   依赖安装完成。

REM ---- 构建前端 ----
echo.
echo [3/5] 构建 React 前端...
cd frontend
if not exist "node_modules" (
    echo   安装 npm 依赖...
    call npm install
)
call npx vite build
cd ..
if not exist "frontend\dist\index.html" (
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
pip install pyinstaller --quiet
REM ★ 从项目根目录运行，输出到 dist/QLH-Edge-Inference/
pyinstaller packaging\qlh-cpu.spec --noconfirm

if exist "dist\QLH-Edge-Inference\QLH-Edge-Inference.exe" (
    echo.
    echo ============================================
    echo   打包完成！
    echo   输出目录: dist\QLH-Edge-Inference\
    echo   可执行文件: QLH-Edge-Inference.exe
    echo ============================================
) else (
    echo.
    echo ============================================
    echo   [错误] 打包失败！请检查上方日志。
    echo ============================================
)

endlocal
pause
