@echo off
rem QLH TUI 管理菜单启动脚本 (Windows 10+)
rem 切换到 UTF-8 代码页，保证中文界面正常显示
chcp 65001 >nul
cd /d "%~dp0"
title QLH TUI 管理菜单

where python >nul 2>nul
if %errorlevel%==0 (
    python src\tui_admin.py %*
) else (
    py -3 src\tui_admin.py %*
)

if errorlevel 1 (
    echo.
    echo [提示] TUI 退出异常。若后端未启动，请先运行: python src\api_server.py
    pause
)
