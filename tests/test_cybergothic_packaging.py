"""CY-PKG-01 static-resource selection and packaging input contracts."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import api_server


def test_source_default_uses_cybergothic_dist():
    path = Path(api_server._resolve_frontend_dist()).as_posix()
    assert path.endswith("/frontend_cybergothic/dist")


def test_explicit_frontend_override_is_the_only_compatibility_escape(tmp_path, monkeypatch):
    legacy = tmp_path / "frontend" / "dist"
    legacy.mkdir(parents=True)
    monkeypatch.setenv("QLH_FRONTEND_DIST", str(legacy))
    assert Path(api_server._resolve_frontend_dist()) == legacy.resolve()


def test_frozen_default_matches_packaged_datas_root(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server.sys, "frozen", True, raising=False)
    monkeypatch.setattr(api_server.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert Path(api_server._resolve_frontend_dist()) == (
        tmp_path / "frontend_cybergothic" / "dist"
    )


def test_standard_packaging_inputs_do_not_default_to_legacy_frontend():
    root = Path(__file__).resolve().parents[1]
    inputs = [
        root / "packaging" / "qlh-cpu.spec",
        root / "packaging" / "qlh-cuda.spec",
        root / "packaging" / "qlh-slim.spec",
        root / "packaging" / "linux" / "build-deb.sh",
        root / "packaging" / "build-cpu.bat",
        root / "packaging" / "build-cuda.bat",
    ]
    for path in inputs:
        text = path.read_text(encoding="utf-8")
        assert "frontend_cybergothic" in text, path
        assert "frontend/dist" not in text.replace("frontend_cybergothic/dist", ""), path
