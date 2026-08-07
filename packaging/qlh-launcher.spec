# -*- mode: python ; coding: utf-8 -*-
"""Small standalone bootstrap launcher; no inference dependencies."""

import os

a = Analysis(
    [os.path.join(SPECPATH, "qlh_launcher.py")],
    pathex=[SPECPATH],
    datas=[(os.path.join(SPECPATH, "leds.ico"), ".")],
    hiddenimports=["tkinter", "tkinter.ttk"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "transformers", "fastapi", "uvicorn"],
)

pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    [],
    [],
    [],
    name="QLH-Launcher",
    icon=os.path.join(SPECPATH, "leds.ico"),
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
    name="QLH-Launcher",
)
