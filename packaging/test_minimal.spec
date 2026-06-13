# -*- mode: python ; coding: utf-8 -*-
"""最小测试 spec"""
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

a = Analysis(
    ['test_minimal.py'],
    pathex=[_SRC_DIR, SPECPATH],
    binaries=[],
    datas=[],
    hiddenimports=[
        'webview',
        'webview.platforms.edgechromium',
        'webview.guilib',
        'webview.http',
        'webview.event',
        'webview.menu',
        'webview.util',
        'webview.window',
        'torch',
        'uvicorn',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'test', 'unittest', 'pydoc'],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Test-Minimal',
    icon='leds.ico',
    console=True,
    debug=False,
    strip=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='Test-Minimal',
)
