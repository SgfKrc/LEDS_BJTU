@echo off
chcp 65001 >nul
echo ============================================
echo  边缘推理优化系统 — 前端管理平台
echo ============================================
echo.
cd /d "%~dp0\frontend"
echo [1/2] 检查依赖...
if not exist "node_modules" (
    echo   安装 npm 依赖...
    call npm install
)
echo [2/2] 启动 Vite 开发服务器 (port 5173)...
echo   API 请求自动代理到 http://localhost:8000
call npx vite --host
pause
