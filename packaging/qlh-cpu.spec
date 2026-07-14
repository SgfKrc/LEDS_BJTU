# -*- mode: python ; coding: utf-8 -*-
# ============================================================
# PyInstaller spec — QLH 边缘推理系统 集显版 (CPU-only)
# ============================================================
# 构建命令: .venv-packaging/Scripts/python -m PyInstaller packaging/qlh-cpu.spec --noconfirm
# 输出目录: dist/QLH-Edge-Inference/
#
# 双引擎架构:
#   1. llama.cpp + GGUF  — CPU/集显默认引擎（Q4_K_M ~1.16 GB）
#   2. PyTorch CPU       — 备选引擎（FP16 ~3.5 GB）
#
# 前置条件:
#   1. CPU-only PyTorch: pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
#   2. llama-cpp-python: pip install llama-cpp-python
#   3. 其余依赖: pip install -r packaging/requirements-cpu.txt
#   4. 前端构建: cd frontend && npm run build
# ============================================================

import os
import sys
import glob

# 项目根目录（SPECPATH = 当前 spec 文件所在目录）
_PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# src 目录（Python 模块搜索路径）
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

# 前端 dist 目录
_FRONTEND_DIST = os.path.join(_PROJECT_ROOT, "frontend", "dist")

if not os.path.isdir(_FRONTEND_DIST):
    raise FileNotFoundError(
        f"前端 dist 目录未找到: {_FRONTEND_DIST}\n"
        "请先构建前端: cd frontend && npm run build"
    )

# ============================================================
# llama.cpp 原生库（DLL）
# ============================================================
_llama_cpp_dlls = []
try:
    import llama_cpp as _lc
    _lc_dir = os.path.dirname(os.path.abspath(_lc.__file__))
    _lib_dir = os.path.join(_lc_dir, "lib")
    if os.path.isdir(_lib_dir):
        for _dll in glob.glob(os.path.join(_lib_dir, "*.dll")):
            _llama_cpp_dlls.append((_dll, "llama_cpp/lib"))
        # Also grab the .lib files (MSVC import libraries, needed at runtime)
        for _lib in glob.glob(os.path.join(_lib_dir, "*.lib")):
            _llama_cpp_dlls.append((_lib, "llama_cpp/lib"))
        print(f"[spec] Collected {len(_llama_cpp_dlls)} llama.cpp native files")
except Exception as _e:
    print(f"[spec] WARNING: Failed to collect llama.cpp DLLs: {_e}")

a = Analysis(
    ['launcher.py'],
    pathex=[_SRC_DIR, SPECPATH],
    binaries=_llama_cpp_dlls,
    datas=[
        # React 前端静态文件 → 运行时目录 frontend/dist/
        (_FRONTEND_DIST, 'frontend/dist'),
    ],
    hiddenimports=[
        # ============================================================
        # llama.cpp 引擎（CPU/集显推理）
        # ============================================================
        'llama_cpp',
        'llama_cpp._internals',
        'llama_cpp.llama_cpp',
        'llama_engine',

        # ============================================================
        # uvicorn 子模块（动态导入）
        # ============================================================
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',

        # ============================================================
        # FastAPI / Starlette
        # ============================================================
        'fastapi',
        'starlette',
        'starlette.middleware',
        'starlette.middleware.cors',
        # Remote model code may import HTTPX dynamically at runtime.
        'httpx',

        # ============================================================
        # Transformers（动态模型类加载）
        # ============================================================
        'transformers',
        'transformers.models.qwen2',
        'transformers.models.auto',
        'transformers_stream_generator',
        'einops',
        'tiktoken',
        'tiktoken._tiktoken',
        'torch',
        'accelerate',
        'bitsandbytes',

        # ============================================================
        # pywebview 原生窗口（替代外部浏览器）
        # ============================================================
        'webview',
        'webview.platforms.edgechromium',
        'webview.platforms.winforms',
        'webview.platforms.cef',
        'webview.guilib',
        'webview.http',
        'webview.event',
        'webview.menu',
        'webview.util',
        'webview.window',

        # ============================================================
        # Web
        # ============================================================
        'pydantic',
        'python_multipart',
        'pandas',

        # ============================================================
        # SSL / OpenSSL（uvicorn 依赖，需要显式收集避免 DLL 冲突）
        # ============================================================
        'ssl',
        '_ssl',

        # ============================================================
        # 数据库（PostgreSQL + psycopg2）
        # ============================================================
        'psycopg2',
        'psycopg2.pool',
        'psycopg2.sql',
        'psycopg2.extras',
        'psycopg2._psycopg',

        # ============================================================
        # 本地存储（DB 不可用时自动降级）
        # ============================================================
        'local_store',

        # ============================================================
        # 智能编排
        # ============================================================
        'graph_orchestrator',

        # ============================================================
        # 工具
        # ============================================================
        'psutil',
        'tqdm',

        # ============================================================
        # 标准库
        # ============================================================
        'asyncio',
    ],
    hookspath=[],
    runtime_hooks=[os.path.join(SPECPATH, 'pyi_rthook_ssl.py')],
    excludes=[
        'tkinter',
        'test',
        'pydoc',
    ],
)

# CPU/集显版本只保留 bitsandbytes 的 CPU 后端。Windows wheel 同时携带
# 多套 CUDA/XPU DLL，PyInstaller hook 会默认全部收集，既无运行价值又会
# 让程序目录额外膨胀约 168 MB。
_bnb_accelerator_prefixes = (
    'libbitsandbytes_cuda',
    'libbitsandbytes_xpu',
)
_binary_count_before = len(a.binaries)
a.binaries = [
    entry for entry in a.binaries
    if not os.path.basename(entry[0]).lower().startswith(_bnb_accelerator_prefixes)
]
print(
    f"[spec] Excluded {_binary_count_before - len(a.binaries)} "
    "bitsandbytes CUDA/XPU binaries from CPU build"
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],                    # ★ onedir: 二进制 DLL 不嵌入 EXE（由 COLLECT 放入 _internal/）
    [],                    # ★ onedir: 数据文件不嵌入 EXE（由 COLLECT 放入 _internal/）
    [],
    name='QLH-Edge-Inference',
    icon='leds.ico',
    console=False,  # ★ 静默模式：不显示控制台窗口，日志写文件
    debug=False,
    strip=True,
    upx=False,
    exclude_binaries=True, # ★ 关键：EXE 不包含任何二进制依赖
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='QLH-Edge-Inference',
)
