@echo off
chcp 65001 >nul
setlocal
echo ============================================
echo  QLH 边缘推理系统 — Inno Setup 安装包编译
echo ============================================
echo.

REM ---- 查找 Inno Setup 6 ----
set "ISCC="

REM 默认安装路径
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)

REM 备用路径
if not defined ISCC (
    if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" (
        set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    )
)

REM 如果找不到，让用户手动输入
if not defined ISCC (
    echo [错误] 未找到 Inno Setup 6
    echo   尝试了: C:\Program Files (x86)\Inno Setup 6\
    echo   尝试了: %ProgramFiles(x86)%\Inno Setup 6\
    echo.
    echo 请确保已安装 Inno Setup 6，或手动指定路径：
    echo   set ISCC=你的路径\ISCC.exe
    pause
    exit /b 1
)

echo Inno Setup: %ISCC%
echo.

REM ---- 检查 PyInstaller 输出 ----
if not exist "..\dist\QLH-Edge-Inference\QLH-Edge-Inference.exe" (
    echo [错误] 未找到 PyInstaller 输出！
    echo   请先运行 build-cpu.bat 完成 PyInstaller 打包。
    pause
    exit /b 1
)
echo PyInstaller 输出: OK
echo.

REM ---- 编译 Inno Setup ----
echo 开始编译安装包...
echo 输出: dist\QLH-Edge-Inference-Setup-v0.1.3.exe
echo 这可能需要 2-5 分钟（压缩中）...
echo.

"%ISCC%" setup.iss

if errorlevel 1 (
    echo.
    echo ============================================
    echo   [错误] 编译失败！请检查上方日志。
    echo ============================================
    pause
    exit /b 1
)

echo.
echo ============================================
echo   编译成功！
echo   安装包: dist\QLH-Edge-Inference-Setup-v0.1.3.exe
echo ============================================
echo.
echo 发给用户前，请先在本机测试安装流程！

endlocal
pause
