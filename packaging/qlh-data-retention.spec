# -*- mode: python ; coding: utf-8 -*-
"""Small UP-N6.4W helper used by Windows install/uninstall transactions."""

import os

a = Analysis(
    [os.path.join(SPECPATH, "data_retention.py")],
    pathex=[SPECPATH],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "transformers", "fastapi", "uvicorn"],
)

pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="QLH-Data-Retention",
    console=True,
    debug=False,
    strip=False,
    upx=False,
)
