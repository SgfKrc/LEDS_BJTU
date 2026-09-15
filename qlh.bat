@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_CMD=python"
where python >nul 2>nul
if not %errorlevel%==0 set "PYTHON_CMD=py -3"
%PYTHON_CMD% qlh.py %*
exit /b %errorlevel%
