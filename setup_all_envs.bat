@echo off
REM ============================================================
REM  QLH one-shot environment setup (Windows)
REM
REM  Creates/installs the QLH core runtime, test, and model sidecar
REM  environments. Web, Android, and release environments are external.
REM  Equivalent to:  python scripts\setup_envs.py --all
REM
REM  Variants:
REM    setup_all_envs.bat --only test,qwen3-sidecar selected envs only
REM    setup_all_envs.bat --check                verify only, no install
REM    setup_all_envs.bat --torch-index-url https://download.pytorch.org/whl/cu126
REM ============================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"

python scripts\setup_envs.py --all %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="9009" goto :done
REM fallback to the py launcher when `python` is not on PATH
py -3 scripts\setup_envs.py --all %*
set "RC=%ERRORLEVEL%"
goto :done

:done
exit /b %RC%
