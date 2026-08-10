"""Contracts for the pinned llama-quantize build and managed package."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.build_llama_quantize import cmake_definitions
from scripts.model_tools.llama_quantize_toolchain import (
    ROOT,
    file_sha256,
    host_target_id,
    load_lock,
    normalize_architecture,
    normalize_platform,
    resolve_quantizer,
    verify_managed_package,
)


def _write_package(root: Path, target_id: str, *, revision: str | None = None) -> Path:
    lock = load_lock()
    target = lock["targets"][target_id]
    package = root / "llama-quantize" / target_id
    package.mkdir(parents=True)
    executable = package / target["executable"]
    executable.write_bytes(b"pinned-quantizer-fixture")
    if target["platform"] == "linux":
        executable.chmod(0o755)
    license_file = package / "LICENSE.llama.cpp"
    license_file.write_text("fixture license\n", encoding="ascii")
    files = [
        {"path": item.name, "size_bytes": item.stat().st_size, "sha256": file_sha256(item)}
        for item in sorted((executable, license_file), key=lambda path: path.name)
    ]
    manifest = {
        "schema_version": 1,
        "tool": "llama-quantize",
        "target_id": target_id,
        "upstream": {
            "repository": lock["upstream"]["repository"],
            "revision": revision or lock["upstream"]["revision"],
        },
        "executable": target["executable"],
        "build": {},
        "smoke": {
            "help": True,
            "q4_k_m_listed": True,
            "runtime_dependencies_verified": True,
            "runtime_dependencies": [],
        },
        "files": files,
    }
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return package


def test_lock_matches_checked_out_llama_cpp_revision():
    lock = load_lock()
    source = ROOT / lock["source"]

    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    assert result.stdout.strip() == lock["upstream"]["revision"]


def test_host_target_normalization_contract():
    assert normalize_platform("win32") == "windows"
    assert normalize_platform("Linux") == "linux"
    assert normalize_architecture("AMD64") == "x86_64"
    assert normalize_architecture("x64") == "x86_64"
    assert host_target_id(platform_name="windows", architecture="amd64") == "windows-x86_64"


def test_managed_package_verifies_revision_hash_and_file_set(tmp_path: Path):
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id)

    report = verify_managed_package(package, expected_target=target_id)

    assert report["valid"] is True
    assert report["status"] == "verified"
    assert report["sha256"] == file_sha256(package / str(report["executable"]))
    assert str(tmp_path) not in str(report)


def test_managed_package_rejects_tampering_and_revision_drift(tmp_path: Path):
    target_id = host_target_id()
    tampered = _write_package(tmp_path / "tampered", target_id)
    (tampered / load_lock()["targets"][target_id]["executable"]).write_bytes(b"changed")
    drifted = _write_package(tmp_path / "drifted", target_id, revision="0" * 40)

    tampered_report = verify_managed_package(tampered, expected_target=target_id)
    drifted_report = verify_managed_package(drifted, expected_target=target_id)

    assert tampered_report["valid"] is False
    assert "file_digest_mismatch" in {item["code"] for item in tampered_report["errors"]}
    assert drifted_report["valid"] is False
    assert "revision_mismatch" in {item["code"] for item in drifted_report["errors"]}


def test_resolver_prefers_verified_managed_package(tmp_path: Path, monkeypatch):
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id)
    monkeypatch.setenv("QLH_MODEL_TOOLS_ROOT", str(tmp_path))
    monkeypatch.delenv("QLH_LLAMA_QUANTIZE", raising=False)

    executable, report = resolve_quantizer(project_root=tmp_path / "project")

    assert executable == package / load_lock()["targets"][target_id]["executable"]
    assert report["status"] == "available"
    assert report["provenance"] == "managed_package"
    assert report["verification"] == "verified"
    assert report["revision"] == load_lock()["upstream"]["revision"]
    assert str(tmp_path) not in str(report)


def test_resolver_fails_closed_for_corrupt_managed_package(tmp_path: Path, monkeypatch):
    target_id = host_target_id()
    package = _write_package(tmp_path, target_id)
    (package / "unexpected.dll").write_bytes(b"not-listed")
    monkeypatch.setenv("QLH_MODEL_TOOLS_ROOT", str(tmp_path))

    executable, report = resolve_quantizer(project_root=tmp_path / "project")

    assert executable is None
    assert report["status"] == "invalid"
    assert report["verification"] == "failed"


def test_cmake_contract_is_offline_portable_and_platform_specific():
    common = cmake_definitions("linux-x86_64", "gnu")
    windows_gnu = cmake_definitions("windows-x86_64", "gnu")
    windows_msvc = cmake_definitions("windows-x86_64", "msvc")

    assert "-DLLAMA_USE_PREBUILT_UI=OFF" in common
    assert "-DLLAMA_OPENSSL=OFF" in common
    assert "-DGGML_CUDA=OFF" in common
    assert "-DGGML_NATIVE=OFF" in common
    assert "-DGGML_OPENMP=OFF" in common
    assert not any("CMAKE_EXE_LINKER_FLAGS" in item for item in common)
    assert any("-static-libstdc++" in item for item in windows_gnu)
    assert "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded" in windows_msvc


def test_release_packaging_requires_managed_quantizer():
    for relative in ("packaging/qlh-cpu.spec", "packaging/qlh-cuda.spec"):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert "verify_managed_package" in content
        assert "model-tools/llama-quantize/windows-x86_64" in content
    linux = (ROOT / "packaging/linux/build-deb.sh").read_text(encoding="utf-8")
    assert "scripts/build_llama_quantize.py" in linux
    assert "model-tools/llama-quantize/linux-x86_64" in linux
