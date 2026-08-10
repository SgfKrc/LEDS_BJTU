# -*- mode: python ; coding: utf-8 -*-
"""Small UP-N6.0 verifier used only by system installer transactions."""

import os

a = Analysis(
    [os.path.join(SPECPATH, "install_manifest.py")],
    pathex=[SPECPATH],
    datas=[],
    hiddenimports=[
        "cryptography",
        "cryptography.hazmat.primitives.asymmetric.ed25519",
    ],
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
    name="QLH-Install-Manifest",
    console=True,
    debug=False,
    strip=False,
    upx=False,
)
