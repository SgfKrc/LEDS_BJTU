# -*- mode: python ; coding: utf-8 -*-
"""QLH 瘦身安装包（SLIM）：不再把 PyTorch 系打进包体。

产物只含：
  - QLH-Edge-Inference.exe：轻量引导器（qlh_launcher.py，不 import 推理运行时）
  - _internal/src/            ：主程序源码（运行时由外部 venv python 以源码方式运行）
  - _internal/frontend_cybergothic/dist/  ：CyberGothic 产品前端静态文件
  - _internal/packaging/      ：外部运行时依赖清单（cpu/cuda）
  - _internal/pubkeys         ：验签公钥
PyTorch / Transformers / llama.cpp / FastAPI / uvicorn 等全部从包体**排除**，
由 qlh_launcher 每次启动用 runtime_guard 检查外部 runtime venv，缺失则
pip 引导（CPU --index-url .../whl/cpu；CUDA 官方默认），再用该 venv python
以 src 源码方式跑 uvicorn。体积从 ~734MB(CPU)/~1.7GB-13GB(CUDA) 降到几乎
仅引导器 + 源码 + 前端（几十 MB），安装包更新不重装大 PyTorch。
"""

import os

_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
_SRC_DIR = os.path.join(_ROOT, "src")
_FRONTEND_DIST = os.path.join(_ROOT, "frontend_cybergothic", "dist")
_PUBKEYS = os.path.join(SPECPATH, "pubkeys")
_ICO = os.path.join(SPECPATH, "leds.ico")

_RUNTIME_REQS = [
    (os.path.join(SPECPATH, "requirements-runtime-cpu.txt"), "packaging"),
    (os.path.join(SPECPATH, "requirements-runtime-cuda.txt"), "packaging"),
]

if not os.path.isdir(_FRONTEND_DIST):
    raise SystemExit(
        "[qlh-slim.spec] frontend_cybergothic/dist 不存在；请先构建前端（cd frontend_cybergothic && npm run build）"
    )

a = Analysis(
    [os.path.join(SPECPATH, "qlh_launcher.py")],
    pathex=[SPECPATH, _ROOT],
    datas=[
        (_ICO, "."),
        (_PUBKEYS, "pubkeys"),
        (_SRC_DIR, "src"),
        (_FRONTEND_DIST, "frontend_cybergothic/dist"),
    ] + _RUNTIME_REQS,
    hiddenimports=[
        "tkinter",
        "tkinter.ttk",
        "cryptography",
        "cryptography.hazmat.primitives.asymmetric.ed25519",
        "launcher_slots",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # 重量级推理运行时：全部移到外部 runtime venv，由 qlh_launcher 引导安装
        "torch", "torchvision", "torchaudio",
        "transformers", "accelerate", "bitsandbytes",
        "llama_cpp", "fastapi", "uvicorn", "pywebview",
    ],
)

pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    [],
    [],
    [],
    name="QLH-Edge-Inference",
    icon=_ICO,
    console=False,
    debug=False,
    strip=True,
    upx=False,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="QLH-Edge-Inference",
)
