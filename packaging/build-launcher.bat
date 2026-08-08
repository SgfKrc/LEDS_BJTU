@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHON=%CD%\.venv-packaging\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo [ERROR] .venv-packaging was not found.
  exit /b 1
)

rem ---- 版本号唯一来源：packaging/version.txt（与 serve.py 的 /latest.json tag 对齐）----
set "VERSION_FILE=packaging\version.txt"
if not exist "%VERSION_FILE%" (
  echo [ERROR] %VERSION_FILE% was not found.
  exit /b 1
)
set "LAUNCHER_VERSION="
for /f "usebackq delims=" %%V in ("%VERSION_FILE%") do set "LAUNCHER_VERSION=%%V"
if "%LAUNCHER_VERSION%"=="" (
  echo [ERROR] %VERSION_FILE% is empty.
  exit /b 1
)

echo Building standalone QLH Launcher v%LAUNCHER_VERSION%...
"%PYTHON%" -m pip install --disable-pip-version-check --quiet -r packaging\requirements-launcher.txt
if errorlevel 1 exit /b 1
"%PYTHON%" -m PyInstaller packaging\qlh-launcher.spec --noconfirm
if errorlevel 1 exit /b 1
if not exist "dist\QLH-Launcher\QLH-Launcher.exe" (
  echo [ERROR] dist\QLH-Launcher\QLH-Launcher.exe was not generated.
  exit /b 1
)
>"dist\QLH-Launcher\health.ok" echo QLH Launcher %date% %time%

rem ---- 自更新资产落点：packaging\dist\（serve.py DIST_DIR 扫描目录，源站才能发布）----
if not exist "packaging\dist" mkdir "packaging\dist"
set "LAUNCHER_ZIP=packaging\dist\QLH-Launcher-v%LAUNCHER_VERSION%.zip"
if exist "%LAUNCHER_ZIP%" del /q "%LAUNCHER_ZIP%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path 'dist\QLH-Launcher\*' -DestinationPath '%LAUNCHER_ZIP%' -Force"
if errorlevel 1 (
  echo [ERROR] Launcher self-update ZIP was not generated.
  exit /b 1
)
echo Launcher output: dist\QLH-Launcher\
echo Launcher self-update asset: %LAUNCHER_ZIP%

set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  echo [INFO] Inno Setup was not found; skipping Launcher Setup.
  exit /b 0
)

echo Building standalone Launcher Setup...
pushd packaging
"%ISCC%" setup-launcher.iss
if errorlevel 1 (
  popd
  exit /b 1
)
popd
if not exist "packaging\dist\QLH-Launcher-Setup-v%LAUNCHER_VERSION%.exe" (
  echo [ERROR] Launcher Setup was not generated.
  exit /b 1
)
echo Installer output: packaging\dist\QLH-Launcher-Setup-v%LAUNCHER_VERSION%.exe
exit /b 0
