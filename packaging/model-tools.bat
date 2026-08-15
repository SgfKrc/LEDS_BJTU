@echo off
chcp 65001 >nul
rem ============================================================
rem  QLH MODEL-TOOLS 命令（Windows 打包版入口）
rem
rem  Usage: model-tools <command> [args...]
rem         model-tools --help
rem
rem  打包版只承诺轻量子命令（inspect/verify/sweep/disk-usage/clean/
rem  sync-status/gguf-convert 预检/sd15-lora-inspect）；需要 torch/
rem  transformers 的重型子命令请在源码或 sidecar 环境运行。
rem ============================================================

set "TOOL_DIR=%~dp0QLH-Model-Tools"
if not exist "%TOOL_DIR%\QLH-Model-Tools.exe" (
    echo [错误] 未找到 QLH-Model-Tools.exe，请检查安装完整性。
    exit /b 1
)

"%TOOL_DIR%\QLH-Model-Tools.exe" %*
exit /b %errorlevel%
