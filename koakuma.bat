@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
rem Koakuma is the qlh alias entry point.
rem Keep this launcher ASCII-only for cmd.exe code-page safety.
cd /d "%~dp0"
set "PYTHON_CMD=python"
where python >nul 2>nul
if not %errorlevel%==0 set "PYTHON_CMD=py -3"
%PYTHON_CMD% qlh.py %*
exit /b %errorlevel%
