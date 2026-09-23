"""tests/test_relay_segment_info.py — 段构建标识（§10.2 待办）的自包含测试。

覆盖三件事：
1. `collect_local_build` 对小文件（shim / libllama）算 sha256，对 GB 级段工件**默认不算**
   （只记大小/名字），显式 `digest_artifacts=True` 才算；
2. `load_ready_build` 能从服务端 ready 文件读到 `build`，读不到/老格式时**如实标注原因**；
3. `unknown_remote_build` 显式标 `remote_unknown`（记录必须能区分「未知」与「没写」）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import relay_segment_info as R  # noqa: E402


def test_collect_local_build_digests_small_files_only(tmp_path: Path) -> None:
    shim = tmp_path / "qlh_keep_head.dll"
    shim.write_bytes(b"shim-bytes")
    (tmp_path / "libllama.dll").write_bytes(b"llama-bytes")
    (tmp_path / "ggml-base.dll").write_bytes(b"ggml-bytes")
    model = tmp_path / "artifact.gguf"
    model.write_bytes(b"x" * 32)

    build = R.collect_local_build(shim=shim, model=model, llama_cpp_module=False)

    assert build["schema_version"] == R.RELAY_SEGMENT_INFO_SCHEMA
    assert build["shim"]["sha256"] == hashlib.sha256(b"shim-bytes").hexdigest()
    # 同目录的 libllama / ggml* 才是真正决定数值的构建产物 ⇒ 一并记摘要
    assert set(build["llama_cpp_build"]) == {"libllama.dll", "ggml-base.dll"}
    # 段工件默认**不算** sha256（GB 级会拖慢每轮实验），但大小/名字必须有
    assert build["model"]["sha256"] is None
    assert build["model"]["size"] == 32
    assert build["model"]["name"] == "artifact.gguf"


def test_collect_local_build_digests_artifact_on_demand(tmp_path: Path) -> None:
    model = tmp_path / "artifact.gguf"
    model.write_bytes(b"abc")
    build = R.collect_local_build(model=model, llama_cpp_module=False, digest_artifacts=True)
    assert build["model"]["sha256"] == hashlib.sha256(b"abc").hexdigest()


def test_collect_local_build_marks_missing_paths(tmp_path: Path) -> None:
    build = R.collect_local_build(shim=tmp_path / "nope.dll", llama_cpp_module=False)
    assert build["shim"]["exists"] is False
    assert build["shim"].get("sha256") is None


def test_load_ready_build_reads_service_build(tmp_path: Path) -> None:
    ready = tmp_path / "tail.ready"
    ready.write_text(json.dumps({
        "role": "tail",
        "n_embd": 4096,
        "build": {"schema_version": R.RELAY_SEGMENT_INFO_SCHEMA,
                  "llama_cpp_version": "0.3.35"},
    }), encoding="utf-8")

    build = R.load_ready_build(ready)

    assert build["source"] == "ready_file"
    assert build["llama_cpp_version"] == "0.3.35"
    assert build["role"] == "tail"
    assert build["schema_version"] == R.RELAY_SEGMENT_INFO_SCHEMA


def test_load_ready_build_marks_unreadable_and_legacy(tmp_path: Path) -> None:
    missing = R.load_ready_build(tmp_path / "nope.ready")
    assert missing["source"] == "ready_file_unreadable"

    # 老版本 ready 文件没有 `build` 字段 ⇒ 显式标注，而不是静默给空 dict
    legacy = tmp_path / "legacy.ready"
    legacy.write_text(json.dumps({"role": "middle", "n_embd": 896}), encoding="utf-8")
    old = R.load_ready_build(legacy)
    assert old["source"] == "ready_file_without_build"
    assert old["role"] == "middle"


def test_unknown_remote_build_is_explicit() -> None:
    info = R.unknown_remote_build("127.0.0.1:50190")
    assert info["source"] == "remote_unknown"
    assert info["endpoint"] == "127.0.0.1:50190"
