from __future__ import annotations

import os
import json
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_module import ModelManager  # noqa: E402
from scripts.model_tools.gemma4_native_binding import (
    MARKER_FILENAME,
    expected_binding_marker,
)


def _fake_managed_modules(root: Path):
    llama_file = root / "llama_cpp" / "__init__.py"
    mtmd_file = root / "llama_cpp" / "mtmd_cpp.py"
    llama_file.parent.mkdir(parents=True)
    llama_file.touch()
    mtmd_file.touch()
    mtmd = types.ModuleType("llama_cpp.mtmd_cpp")
    mtmd.__file__ = str(mtmd_file)
    for symbol in expected_binding_marker()["abi"]["mtmd_python_symbols"]:
        setattr(mtmd, symbol, lambda *args: 0)
    llama = types.ModuleType("llama_cpp")
    llama.__file__ = str(llama_file)
    llama.__version__ = "0.3.28"
    llama.mtmd_cpp = mtmd
    return llama, mtmd


def _write_marker(root: Path) -> Path:
    marker = root / "llama_cpp" / MARKER_FILENAME
    marker.write_text(json.dumps(expected_binding_marker()), encoding="utf-8")
    return marker


def test_gemma4_binding_missing_managed_directory_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("QLH_GEMMA4_SITE_PACKAGES", str(tmp_path / "missing"))
    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    monkeypatch.delitem(sys.modules, "llama_cpp.mtmd_cpp", raising=False)
    with pytest.raises(RuntimeError, match="existing managed"):
        ModelManager._prepare_gemma4_native_binding(use_cuda=False)


def test_gemma4_binding_rejects_external_same_version_shadow(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    external = tmp_path / "external" / "llama_cpp.py"
    external.parent.mkdir()
    external.touch()
    llama, mtmd = _fake_managed_modules(managed)
    llama.__file__ = str(external)
    monkeypatch.setenv("QLH_GEMMA4_SITE_PACKAGES", str(managed))
    monkeypatch.setitem(sys.modules, "llama_cpp", llama)
    monkeypatch.setitem(sys.modules, "llama_cpp.mtmd_cpp", mtmd)
    with pytest.raises(RuntimeError, match="outside the managed"):
        ModelManager._prepare_gemma4_native_binding(use_cuda=False)


def test_gemma4_binding_accepts_modules_owned_by_managed_root(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    llama, mtmd = _fake_managed_modules(managed)
    _write_marker(managed)
    monkeypatch.setenv("QLH_GEMMA4_SITE_PACKAGES", str(managed))
    monkeypatch.setitem(sys.modules, "llama_cpp", llama)
    monkeypatch.setitem(sys.modules, "llama_cpp.mtmd_cpp", mtmd)
    ModelManager._prepare_gemma4_native_binding(use_cuda=False)


def test_gemma4_binding_requires_frozen_marker(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    llama, mtmd = _fake_managed_modules(managed)
    monkeypatch.setenv("QLH_GEMMA4_SITE_PACKAGES", str(managed))
    monkeypatch.setitem(sys.modules, "llama_cpp", llama)
    monkeypatch.setitem(sys.modules, "llama_cpp.mtmd_cpp", mtmd)
    with pytest.raises(RuntimeError, match="marker is missing"):
        ModelManager._prepare_gemma4_native_binding(use_cuda=False)


def test_gemma4_binding_rejects_marker_identity_change(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    llama, mtmd = _fake_managed_modules(managed)
    marker = _write_marker(managed)
    value = json.loads(marker.read_text(encoding="utf-8"))
    value["upstream"]["revision"] = "0" * 40
    marker.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setenv("QLH_GEMMA4_SITE_PACKAGES", str(managed))
    monkeypatch.setitem(sys.modules, "llama_cpp", llama)
    monkeypatch.setitem(sys.modules, "llama_cpp.mtmd_cpp", mtmd)
    with pytest.raises(RuntimeError, match="does not match"):
        ModelManager._prepare_gemma4_native_binding(use_cuda=False)
