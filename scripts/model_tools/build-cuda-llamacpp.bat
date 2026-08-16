@echo off
REM G4.5: rebuild llama-cpp-python 0.3.28 with CUDA.
REM Configure machine-specific paths through environment variables; do not use
REM vcvars because that caused cudafe++ crashes in the reference environment.
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0..\.."
if not defined QLH_LLAMA_CPP_SRC set "QLH_LLAMA_CPP_SRC=%TEMP%\qlh-llama-cpp-python-src-cuda"
if not defined QLH_GEMMA4_VENV set "QLH_GEMMA4_VENV=%ROOT%\.venv-gemma4-native"
if not defined QLH_VS_ROOT set "QLH_VS_ROOT=%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools"
if not defined QLH_MSVC_VERSION set "QLH_MSVC_VERSION=14.51.36231"
if not defined QLH_WINSDK_VERSION set "QLH_WINSDK_VERSION=10.0.26100.0"
if not defined QLH_CUDA_ROOT set "QLH_CUDA_ROOT=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
if not defined QLH_CUDA_RUNTIME_DIR set "QLH_CUDA_RUNTIME_DIR=%QLH_CUDA_ROOT%\bin"

set "MSVC=%QLH_VS_ROOT%\VC\Tools\MSVC\%QLH_MSVC_VERSION%"
set "WINSDK=%ProgramFiles(x86)%\Windows Kits\10"
set "CUDA=%QLH_CUDA_ROOT%"
for %%I in ("!CUDA!") do set "CUDA_SHORT=%%~sI"
set "CUDA_SHORT=!CUDA_SHORT:\=/!"
set "INCLUDE=!MSVC!\include;!WINSDK!\Include\!QLH_WINSDK_VERSION!\ucrt;!WINSDK!\Include\!QLH_WINSDK_VERSION!\um;!WINSDK!\Include\!QLH_WINSDK_VERSION!\shared"
set "LIB=!MSVC!\lib\x64;!WINSDK!\Lib\!QLH_WINSDK_VERSION!\ucrt\x64;!WINSDK!\Lib\!QLH_WINSDK_VERSION!\um\x64"
set "PATH=!MSVC!\bin\Hostx64\x64;!CUDA!\bin;!PATH!"
set "GGML_CUDA=on"
set "CudaToolkitDir=!CUDA!\"
set "CMAKE_ARGS=-DGGML_CUDA=on -DCUDAToolkit_ROOT=!CUDA_SHORT! -DCMAKE_CUDA_COMPILER=!CUDA_SHORT!/bin/nvcc.exe -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl"

if not exist "!QLH_LLAMA_CPP_SRC!\pyproject.toml" (
  echo ERROR: llama-cpp-python source was not found: !QLH_LLAMA_CPP_SRC!
  exit /b 2
)
if not exist "!QLH_GEMMA4_VENV!\Scripts\python.exe" (
  echo ERROR: Gemma native virtualenv was not found: !QLH_GEMMA4_VENV!
  exit /b 2
)
if not exist "!MSVC!\bin\Hostx64\x64\cl.exe" (
  echo ERROR: MSVC toolchain was not found: !MSVC!
  exit /b 2
)
if not exist "!CUDA!\bin\nvcc.exe" (
  echo ERROR: CUDA toolkit was not found: !CUDA!
  exit /b 2
)
for /f "usebackq delims=" %%R in (`git -C "!QLH_LLAMA_CPP_SRC!\vendor\llama.cpp" rev-parse HEAD 2^>nul`) do set "LLAMA_CPP_REV=%%R"
if /I not "!LLAMA_CPP_REV!"=="47e1de77aa0f06bf73cfd8c5281d95979f89fcbe" (
  echo ERROR: llama.cpp revision mismatch: !LLAMA_CPP_REV!
  exit /b 2
)
findstr /C:"def mtmd_tokenize" "!QLH_LLAMA_CPP_SRC!\llama_cpp\mtmd_cpp.py" >nul
if errorlevel 1 (
  echo ERROR: MTMD binding patch is not applied to llama_cpp\mtmd_cpp.py
  exit /b 2
)

pushd "!QLH_LLAMA_CPP_SRC!"
"!QLH_GEMMA4_VENV!\Scripts\python.exe" -m pip install . --no-build-isolation --force-reinstall --no-deps
set "BUILD_RC=!ERRORLEVEL!"
popd
if not "!BUILD_RC!"=="0" (
  echo BUILD_RC=!BUILD_RC!
  exit /b !BUILD_RC!
)

set "LLAMA_LIB=!QLH_GEMMA4_VENV!\Lib\site-packages\llama_cpp\lib"
for %%D in (cudart64_13.dll cublas64_13.dll cublasLt64_13.dll) do (
  if not exist "!QLH_CUDA_RUNTIME_DIR!\%%D" (
    echo ERROR: required CUDA runtime was not found: !QLH_CUDA_RUNTIME_DIR!\%%D
    exit /b 2
  )
  copy /Y "!QLH_CUDA_RUNTIME_DIR!\%%D" "!LLAMA_LIB!\%%D" >nul
  if errorlevel 1 (
    echo ERROR: failed to stage %%D into llama_cpp\lib
    exit /b 2
  )
)
"!QLH_GEMMA4_VENV!\Scripts\python.exe" -c "import llama_cpp, llama_cpp.mtmd_cpp as m; assert llama_cpp.__version__ == '0.3.28'; assert callable(m.mtmd_tokenize); assert callable(m.mtmd_helper_decode_image_chunk)"
if errorlevel 1 (
  echo ERROR: rebuilt Gemma 4 binding failed the ABI check
  exit /b 2
)
"!QLH_GEMMA4_VENV!\Scripts\python.exe" "%ROOT%\scripts\model_tools\gemma4_native_binding.py" --write-marker --site-packages "!QLH_GEMMA4_VENV!\Lib\site-packages"
if errorlevel 1 (
  echo ERROR: failed to write the frozen Gemma 4 binding marker
  exit /b 2
)
echo BUILD_RC=!BUILD_RC!
exit /b !BUILD_RC!
