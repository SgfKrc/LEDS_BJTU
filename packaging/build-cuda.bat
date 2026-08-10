@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
echo ============================================
echo  QLH 边缘推理系统 — 独显版打包脚本
echo ============================================
echo.
cd /d "%~dp0\.."

REM ---- 检查项目专用 CUDA 打包环境 ----
echo [1/5] 检查项目 CUDA 打包环境...
set "PYTHON=%CD%\.venv-packaging-cuda\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo   [错误] 未找到项目打包虚拟环境: %PYTHON%
    echo   请先创建 .venv-packaging-cuda 并安装 packaging\requirements-cpu.txt
    pause
    exit /b 1
)
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo   [错误] 项目 CUDA 打包虚拟环境不可用
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
"%PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'CUDA torch not found'"
if errorlevel 1 (
    echo   [错误] 需要 CUDA 版 torch！
    echo   请先安装: pip install torch  (默认带 CUDA 12.x)
    echo   当前 torch 版本:
    "%PYTHON%" -c "import torch; print(torch.__version__)"
    pause
    exit /b 1
)
echo   CUDA torch: OK
echo.

REM ---- 安装共享打包依赖（SD 1.5 侧车依赖保持可选，不在此处安装）----
echo [2/5] 安装共享打包依赖...
"%PYTHON%" -m pip install -r packaging\requirements-cpu.txt --quiet
"%PYTHON%" -m pip install pyinstaller --quiet
echo   依赖安装完成。
echo.

REM ---- 构建前端 ----
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
echo.

REM ---- 创建必要的目录 ----
echo [4/5] 准备打包目录...
if not exist "models\qwen-1_8b-chat" mkdir "models\qwen-1_8b-chat"
if not exist "logs" mkdir "logs"
echo   目录就绪。
echo.

REM ---- PyInstaller 打包（固定使用项目 CUDA venv） ----
echo [5/5] PyInstaller 打包 (CUDA 独显版)...
echo   这可能需要 10-20 分钟，请耐心等待...
"%PYTHON%" -m PyInstaller packaging\qlh-cuda.spec --noconfirm
if errorlevel 1 exit /b 1

if exist "dist\QLH-Edge-Inference-CUDA\QLH-Edge-Inference.exe" (
    if not exist "dist\QLH-Edge-Inference-CUDA\tools" mkdir "dist\QLH-Edge-Inference-CUDA\tools"
    "%PYTHON%" -m PyInstaller packaging\qlh-install-manifest.spec --noconfirm ^
        --distpath "dist\QLH-Edge-Inference-CUDA\tools" ^
        --workpath "build\qlh-install-manifest-cuda"
    if errorlevel 1 exit /b 1
    if not exist "dist\QLH-Edge-Inference-CUDA\tools\QLH-Install-Manifest.exe" exit /b 1
    xcopy /e /i /y "packaging\pubkeys" "dist\QLH-Edge-Inference-CUDA\pubkeys" >nul
    if errorlevel 1 exit /b 1
    "%PYTHON%" packaging\install_manifest.py build ^
        --root "dist\QLH-Edge-Inference-CUDA" ^
        --app-id qlh-edge-inference ^
        --version "!APP_VERSION!" ^
        --platform windows ^
        --variant cuda ^
        --package-kind application ^
        --key "!QLH_SIGNING_KEY!" ^
        --trusted-keys-dir "packaging\pubkeys" ^
        --include "bjtu.bat=bjtu.bat" ^
        --include "packaging\version.txt=version.txt" ^
        --include "README.md=docs/README.md" ^
        --include "docs\整体架构.md=docs/整体架构.md" ^
        --include "docs\模块接口说明.md=docs/模块接口说明.md" ^
        --include "docs\核心技术原理.md=docs/核心技术原理.md" ^
        --include "packaging\scripts\convert_to_gguf.py=tools/convert_to_gguf.py"
    if errorlevel 1 exit /b 1
    echo.
    echo ============================================
    echo   打包完成！
    echo   输出目录: dist\QLH-Edge-Inference-CUDA\
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
