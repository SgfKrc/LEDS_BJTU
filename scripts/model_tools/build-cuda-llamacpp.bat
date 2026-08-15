@echo off
REM G4.5: rebuild llama-cpp-python 0.3.28 with CUDA (minimal env, no vcvars to avoid cudafe++ crash)
set VSDIR=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools
set MSVC=%VSDIR%\VC\Tools\MSVC\14.51.36231
set WK=10.0.26100.0
set CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set INCLUDE=%MSVC%\include;C:\Program Files (x86)\Windows Kits\10\Include\%WK%\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\%WK%\um;C:\Program Files (x86)\Windows Kits\10\Include\%WK%\shared
set LIB=%MSVC%\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\%WK%\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\%WK%\um\x64
set PATH=%MSVC%\bin\Hostx64\x64;%CUDA%\bin;%WINDIR%\System32;%WINDIR%\System32\WindowsPowerShell\v1.0;C:\Users\Koakuma\AppData\Local\Programs\Python\Python312\Scripts;C:\Users\Koakuma\AppData\Local\Programs\Python\Python312
set GGML_CUDA=on
set CudaToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\
set CMAKE_ARGS=-DGGML_CUDA=on -DCUDAToolkit_ROOT=C:/PROGRA~1/NVIDIA~2/CUDA/v13.3 -DCMAKE_CUDA_COMPILER=C:/PROGRA~1/NVIDIA~2/CUDA/v13.3/bin/nvcc.exe -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cd /d C:\Users\Koakuma\AppData\Local\Temp\reasonix-session-tmp-3363469897\llama-cpp-python-src-cuda
G:\C\PYT\qlh\.venv-gemma4-native\Scripts\python.exe -m pip install . --no-build-isolation --force-reinstall --no-deps 2>&1 | findstr /V "warning"
echo BUILD_RC=%ERRORLEVEL%
