@echo off
chcp 65001 >nul
rem Koakuma —— `qlh` 的等价别名入口（Windows）
rem 与 qlh.bat 完全一样：同一个 Python 入口，参数原样透传。
cd /d "%~dp0"
set "PYTHON_CMD=python"
where python >nul 2>nul
if not %errorlevel%==0 set "PYTHON_CMD=py -3"
%PYTHON_CMD% qlh.py %*
exit /b %errorlevel%
