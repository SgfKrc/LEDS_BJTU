# -*- mode: python ; coding: utf-8 -*-
"""T9 Textual 聊天页的 Windows 控制台伴随程序。

主 GUI 保持静默窗口；该 onedir 程序专门由已安装目录中的 ``bjtu chat``
调用。它被放入同一个主应用树，因此受同一份签名安装清单保护。
"""

import os


_PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

# Textual 在 App.get_driver_class 中按平台动态导入驱动；只显式带上运行时
# 可能走到的驱动，避免 collect_submodules 把开发机的可选调试/图形栈一并带入。
_HIDDEN_IMPORTS = [
    "httpx",
    "tui_chat",
    "tui_sse",
    "tui_shared",
    "textual.drivers.windows_driver",
    "textual.drivers.linux_driver",
    "textual.drivers.headless_driver",
    "textual.drivers.linux_inline_driver",
    "textual.drivers._input_reader_windows",
    "textual.drivers._input_reader_linux",
]

a = Analysis(
    [os.path.join(_SRC_DIR, "tui_chat.py")],
    pathex=[_SRC_DIR],
    binaries=[],
    datas=[],
    hiddenimports=_HIDDEN_IMPORTS,
    hookspath=[],
    runtime_hooks=[],
    # Rich 的 Jupyter 渲染是普通终端的可选分支；开发机若恰好安装 IPython，
    # PyInstaller 会沿其图形/科学计算依赖树误收集数百 MB。Rich 在缺失时已有
    # ModuleNotFoundError 回退，明确排除不会影响终端聊天页。
    excludes=[
        "tkinter",
        "test",
        "pydoc",
        "IPython",
        "jedi",
        "matplotlib",
        "numpy",
        "PIL",
        "PyQt5",
        "traitlets",
        "nbformat",
        "pytest",
        "_pytest",
        "py",
        "zmq",
    ],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    [],
    [],
    name="QLH-TUI-Chat",
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
    name="QLH-TUI-Chat",
)
