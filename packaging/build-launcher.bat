@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHON=%CD%\.venv-packaging\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo [ERROR] .venv-packaging was not found.
  exit /b 1
)

echo Building standalone QLH Launcher...
"%PYTHON%" -m PyInstaller packaging\qlh-launcher.spec --noconfirm
if errorlevel 1 exit /b 1
if not exist "dist\QLH-Launcher\QLH-Launcher.exe" (
  echo [ERROR] dist\QLH-Launcher\QLH-Launcher.exe was not generated.
  exit /b 1
)
echo Launcher output: dist\QLH-Launcher\

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
if not exist "packaging\dist\QLH-Launcher-Setup-v0.1.8.1.exe" (
  echo [ERROR] Launcher Setup was not generated.
  exit /b 1
)
echo Installer output: packaging\dist\QLH-Launcher-Setup-v0.1.8.1.exe
exit /b 0
