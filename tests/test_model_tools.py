"""Contract tests for the read-only MODEL-TOOLS P0 trio."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from scripts.model_tools.cli import main
from scripts.model_tools.gguf import inspect_gguf, verify_gguf
from scripts.model_tools.sweep import sweep_models


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I I", 4, value)


def _write_gguf(path: Path, *, payload: bytes = b"\x00" * 16) -> str:
    metadata = b"".join([
        _kv_string("general.architecture", "test"),
        _kv_string("general.name", "tiny-test"),
        _kv_u32("general.alignment", 32),
        _kv_u32("test.context_length", 2048),
        _kv_u32("test.block_count", 2),
    ])
    tensor_prefix = _string("blk.0.weight") + struct.pack("<I Q I", 1, 4, 0)
    header_prefix = b"GGUF" + struct.pack("<I Q Q", 3, 1, 5) + metadata + tensor_prefix
    data_offset = (len(header_prefix) + 31) // 32 * 32
    tensor = tensor_prefix + struct.pack("<Q", 0)
    header = b"GGUF" + struct.pack("<I Q Q", 3, 1, 5) + metadata + tensor
    content = header + (b"\x00" * (data_offset - len(header))) + payload
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_inspect_reads_metadata_and_tensor_descriptors(tmp_path: Path):
    target = tmp_path / "tiny.gguf"
    _write_gguf(target)

    report = inspect_gguf(target)

    assert report["valid"] is True
    assert report["version"] == 3
    assert report["derived"] == {
        "architecture": "test",
        "name": "tiny-test",
        "context_length": 2048,
        "block_count": 2,
        "vocab_size": None,
        "tokenizer_model": None,
        "tensor_types": {"F32": 1},
    }
    assert report["tensors"] == [{
        "name": "blk.0.weight",
        "shape": [4],
        "type": "F32",
        "type_code": 0,
        "offset": 0,
        "byte_size": 16,
    }]


def test_verify_checks_sidecar_and_detects_tampering(tmp_path: Path):
    target = tmp_path / "tiny.gguf"
    digest = _write_gguf(target)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8")

    good = verify_gguf(target)
    assert good["valid"] is True
    assert good["sha256_checked"] is True
    assert good["sha256"] == digest

    target.write_bytes(target.read_bytes()[:-1] + b"X")
    bad = verify_gguf(target)
    assert bad["valid"] is False
    assert any("sha256 mismatch" in error for error in bad["errors"])


def test_verify_rejects_truncated_tensor_data(tmp_path: Path):
    target = tmp_path / "truncated.gguf"
    _write_gguf(target, payload=b"\x00" * 8)

    report = verify_gguf(target)

    assert report["valid"] is False
    assert any("tensor data truncated" in error for error in report["errors"])


def test_models_sweep_reports_integrity_and_orphan_candidates_without_writes(tmp_path: Path):
    root = tmp_path / "models"
    root.mkdir()
    good = root / "good.gguf"
    digest = _write_gguf(good)
    (root / "good.gguf.sha256").write_text(f"{digest}  good.gguf\n", encoding="utf-8")
    (root / "download.part").write_bytes(b"partial")
    (root / "orphan.sha256").write_text("a" * 64 + "  missing.gguf\n", encoding="utf-8")
    before = {item: item.stat().st_mtime_ns for item in root.iterdir()}

    report = sweep_models(root)

    assert report["valid"] is True
    assert report["read_only"] is True
    assert report["gguf"][0]["valid"] is True
    assert "download.part" in report["orphan_files"]
    assert "orphan.sha256" in report["orphan_files"]
    assert before == {item: item.stat().st_mtime_ns for item in root.iterdir()}


def test_cli_json_output_and_exit_code(tmp_path: Path, capsys):
    target = tmp_path / "tiny.gguf"
    _write_gguf(target)

    assert main(["gguf_inspect", str(target), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"tensor_count": 1' in output
