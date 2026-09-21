# build_keep_head_shim.ps1 - build the QLH keep-head shim (P2 route C)
#
# Why: the pip-bound llama.cpp exposes only the `embeddings` channel, which returns
# `output_norm(H)` and therefore cannot be a layer-relay upstream. The project patch
# `scripts/model_tools/patches/llama-cpp-layer-forward-api.patch` exports the layer-input /
# nextn APIs, but `llama_context_params` is a big by-value struct: calling the self-built
# DLL with the PyPI ctypes declaration mismatches fields (measured: "Unsupported ctx type").
# This shim is compiled against the SAME `include/llama.h` as the DLL it calls, so the
# Python side only needs 4 trivial signatures.
#
# ASCII only: PowerShell 5.1 reads .ps1 as ANSI when there is no BOM.
#
# Usage (from the repository root):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/model_tools/build_keep_head_shim.ps1
param(
    [string]$LlamaCppDir = "android/app/src/main/cpp/llama.cpp",
    [string]$BuildDir    = "build/keephead/build-cpu",
    [string]$OutPath     = "build/keephead/build-cpu/bin/qlh_keep_head.dll",
    [string]$Gcc         = "C:\msys64\ucrt64\bin\gcc.exe"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repo

$src = Join-Path $repo "scripts/model_tools/keep_head_shim/qlh_keep_head.c"
if (-not (Test-Path $src)) { Write-Host "[FAIL] missing shim source: $src"; exit 1 }
if (-not (Test-Path $LlamaCppDir)) { Write-Host "[FAIL] missing llama.cpp tree: $LlamaCppDir"; exit 1 }
if (-not (Test-Path "$BuildDir/src/libllama.dll.a")) {
    Write-Host "[FAIL] missing import library $BuildDir/src/libllama.dll.a"
    Write-Host "       build llama.cpp first, e.g.:"
    Write-Host "         cmake -S $LlamaCppDir -B $BuildDir -G 'MinGW Makefiles' ^"
    Write-Host "               -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ^"
    Write-Host "               -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF ^"
    Write-Host "               -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF"
    Write-Host "         cmake --build $BuildDir --target llama -j 8"
    exit 1
}
if (-not (Test-Path $Gcc)) { Write-Host "[FAIL] gcc not found: $Gcc"; exit 1 }

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutPath) | Out-Null

Write-Host "[build] $OutPath"
& $Gcc -shared -O2 -o $OutPath $src `
    -I"$LlamaCppDir/include" -I"$LlamaCppDir/ggml/include" `
    -L"$BuildDir/src" -lllama
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] gcc exit=$LASTEXITCODE"; exit $LASTEXITCODE }

# MinGW runtime DLLs must sit next to the shim (Windows searches the module directory).
$mingwBin = Split-Path -Parent $Gcc
foreach ($name in @("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll")) {
    $from = Join-Path $mingwBin $name
    if (Test-Path $from) { Copy-Item $from (Join-Path (Split-Path -Parent $OutPath) $name) -Force }
}

Write-Host "[done] $OutPath"
Write-Host "[note] keep this directory's DLLs together (libllama/ggml*/MinGW runtime), and set"
Write-Host "       QLH_KEEP_HEAD_DLL_DIRS=<msys bin dir> if the api-ms-win-crt-* copies are needed."
exit 0
