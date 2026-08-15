# -*- mode: python ; coding: utf-8 -*-
"""MODEL-TOOLS 统一 CLI 的 Windows 控制台入口。

打包为独立 ``QLH-Model-Tools.exe`` 放入主应用树，受同一份签名安装清单
保护；安装后经 ``model-tools`` 入口 bat 调用（等同源码模式
``python scripts/model_tools.py``）。工具子模块只依赖标准库与受管
工具链（llama-quantize 包等），不携带 torch/llama_cpp。
"""

import os

from PyInstaller.utils.hooks import collect_submodules


_PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

# model_tools CLI 按子命令惰性调度；全量收集 scripts.model_tools 子模块，
# 避免运行时才触发的动态导入在打包版中丢失。部分子模块（qwen3_multimodal_*）
# 顶层 import src 顶层模块（源码模式靠运行时 sys.path 注入），打包版必须让
# Analysis 直接解析到 src 目录并收集这些模块进 PYZ。
_HIDDEN_IMPORTS = (
    collect_submodules("scripts.model_tools")
    + collect_submodules("scripts.experiment_core")
)

a = Analysis(
    [os.path.join(_PROJECT_ROOT, "scripts", "model_tools.py")],
    pathex=[_PROJECT_ROOT, _SRC_DIR],
    binaries=[],
    datas=[],
    hiddenimports=_HIDDEN_IMPORTS,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "test",
        "pydoc",
        "IPython",
        "jedi",
        "matplotlib",
        "PyQt5",
        "traitlets",
        "nbformat",
        "pytest",
        "_pytest",
        "py",
        "zmq",
        # 打包版只承诺轻量子命令（inspect/verify/sweep/disk-usage/clean/
        # sync-status/gguf-convert 预检/sd15-lora-inspect）。需要 torch/
        # transformers 的重型子命令（llm_smoke_matrix、qwen3-* smoke、
        # gguf-convert 真实执行等）在源码或 sidecar 环境运行；显式排除
        # 避免 364MB+ 的 torch 依赖树膨胀安装包。
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "PIL",
        "safetensors",
        "llama_cpp",
        "ollama",
    ],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    [],
    [],
    name="QLH-Model-Tools",
    console=True,
    debug=False,
    strip=False,
    upx=False,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="QLH-Model-Tools",
)
