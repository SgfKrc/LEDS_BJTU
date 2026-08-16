"""从节点补丁监听器（M2）单测。

隔离 git/socket；用临时 Ed25519 密钥对与签名帧做真实验签闭环。覆盖：
验签拒绝（篡改/错 schema/错 repo/key 不匹配）、dirty 告警后强制覆盖、
强拉对齐 commit_sha、fetch 代理重试 fail-closed、ack 协议、
restart_requested 转交标记。
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))

import patch_listener as pl  # noqa: E402
from signing import canonical_json  # noqa: E402


@pytest.fixture
def keys(tmp_path):
    """临时 Ed25519 密钥对：返回 (key_path, pub_json_path, pub_bytes)。"""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )
    private = Ed25519PrivateKey.generate()
    pub_raw = private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw)
    key_path = tmp_path / "node-patch.key"
    key_path.write_bytes(private.private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()))
    pub_path = tmp_path / "node-patch.pub.json"
    pub_path.write_text(json.dumps({
        "key_id": "node-patch",
        "public_key": base64.b64encode(pub_raw).decode("ascii"),
    }), encoding="utf-8")
    return key_path, pub_path, pub_raw


def _sign_frame(payload: dict, key_path: Path) -> dict:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    private = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
    body = canonical_json(payload)
    return {**payload, "signature": base64.b64encode(
        private.sign(body)).decode("ascii"), "key_id": "node-patch"}


def _valid_payload(**over):
    base = {
        "schema": "qlh.patch_frame.v1",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": "https://github.com/sgfd8134/leds_-bjtu_-gitee.git",
        "branch": "dev",
        "commit_sha": "a" * 40,
        "proxy_port": 7897,
        "note": "fix: x",
    }
    base.update(over)
    return base


@pytest.fixture
def args(keys, tmp_path):
    class A:
        verify_key = str(keys[1])
        no_clean = False
        branch = "dev"
    return A()


def test_verify_valid_frame(keys, args):
    key_path, _, _ = keys
    frame = _sign_frame(_valid_payload(), key_path)
    assert pl._verify_frame(dict(frame), Path(args.verify_key), pl._log_setup()) is None


def test_verify_rejects_tampered_commit(keys, args):
    key_path, _, _ = keys
    frame = _sign_frame(_valid_payload(), key_path)
    frame["commit_sha"] = "b" * 40
    reason = pl._verify_frame(dict(frame), Path(args.verify_key), pl._log_setup())
    assert reason and "signature invalid" in reason


def test_verify_rejects_wrong_schema(keys, args):
    key_path, _, _ = keys
    frame = _sign_frame(_valid_payload(schema="qlh.evil.v1"), key_path)
    reason = pl._verify_frame(dict(frame), Path(args.verify_key), pl._log_setup())
    assert reason and "unknown schema" in reason


def test_verify_rejects_repo_mismatch(keys, args, monkeypatch):
    key_path, _, _ = keys
    monkeypatch.setattr(pl, "_ACCEPTED_REPO",
                        "https://github.com/only/this.git")
    frame = _sign_frame(_valid_payload(), key_path)
    reason = pl._verify_frame(dict(frame), Path(args.verify_key), pl._log_setup())
    assert reason and "repo mismatch" in reason


def test_verify_rejects_wrong_key(keys, args, tmp_path):
    # 用另一把密钥签名 -> 验签失败
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    other = Ed25519PrivateKey.generate()
    other_path = tmp_path / "other.key"
    other_path.write_bytes(other.private_bytes(
        __import__("cryptography.hazmat.primitives.serialization",
                   fromlist=["Encoding", "PrivateFormat", "NoEncryption"]
                   ).Encoding.Raw,
        __import__("cryptography.hazmat.primitives.serialization",
                   fromlist=["Encoding", "PrivateFormat", "NoEncryption"]
                   ).PrivateFormat.Raw,
        __import__("cryptography.hazmat.primitives.serialization",
                   fromlist=["Encoding", "PrivateFormat", "NoEncryption"]
                   ).NoEncryption()))
    frame = _sign_frame(_valid_payload(), other_path)
    reason = pl._verify_frame(dict(frame), Path(args.verify_key), pl._log_setup())
    assert reason and "signature invalid" in reason


def test_apply_patch_dirty_warns_and_force_overwrites(keys, args, monkeypatch):
    """dirty 告警不拒绝：继续 fetch/reset 强拉对齐。"""
    calls = []

    def fake_git(gargs, *, env=None, check=True):
        calls.append(gargs)
        if gargs[0] == "status":
            return " M docs/x.md"  # dirty
        if gargs[0] == "fetch":
            assert env and env["HTTPS_PROXY"] == "http://127.0.0.1:7897"
            return ""
        if gargs[0] == "rev-parse":
            return "a" * 40  # 与帧 commit 对齐
        return ""

    monkeypatch.setattr(pl, "_git", fake_git)
    frame = _valid_payload()
    status, detail = pl._apply_patch(frame, no_clean=False, logger=pl._log_setup())
    assert status == "applied"
    assert detail == "a" * 40
    assert ["reset", "--hard", "origin/dev"] in calls
    assert ["clean", "-fd"] in calls


def test_apply_patch_head_mismatch_fails(keys, args, monkeypatch):
    def fake_git(gargs, *, env=None, check=True):
        if gargs[0] == "status":
            return ""
        if gargs[0] == "fetch":
            return ""
        if gargs[0] == "rev-parse":
            return "f" * 40  # 不等于目标
        return ""

    monkeypatch.setattr(pl, "_git", fake_git)
    status, detail = pl._apply_patch(_valid_payload(), no_clean=False,
                                     logger=pl._log_setup())
    assert status == "failed"
    assert "!=" in detail


def test_apply_patch_fetch_retries_then_fails(keys, args, monkeypatch):
    attempts = {"n": 0}

    def fake_git(gargs, *, env=None, check=True):
        if gargs[0] == "fetch":
            attempts["n"] += 1
            raise RuntimeError("proxy unreachable")
        return ""

    monkeypatch.setattr(pl, "_git", fake_git)
    status, detail = pl._apply_patch(_valid_payload(), no_clean=False,
                                     logger=pl._log_setup())
    assert status == "failed"
    assert "fetch 失败" in detail
    assert attempts["n"] == pl.FETCH_RETRIES


def test_handle_frame_restart_requested_flag(keys, args, monkeypatch):
    """restart_requested 只转交标记（ack 带后缀），不执行重启。"""
    key_path, _, _ = keys
    frame = _sign_frame(_valid_payload(restart_requested=True), key_path)
    monkeypatch.setattr(pl, "_git", lambda gargs, **kw: {
        "status": "", "fetch": "", "rev-parse": "a" * 40,
    }.get(gargs[0], ""))
    ack = pl._handle_frame(frame, args, pl._log_setup())
    assert ack.startswith("applied:")
    assert "restart_requested" in ack


def test_recv_frame_rejects_oversized_and_bad_header():
    class FakeConn:
        def __init__(self, payload):
            self._payload = payload
            self._pos = 0

        def recv(self, n):
            chunk = self._payload[self._pos:self._pos + n]
            self._pos += len(chunk)
            return chunk

    # 超长帧头（超过 MAX_FRAME_BYTES）-> None
    oversized = f"{pl.MAX_FRAME_BYTES + 1:010d}".encode("ascii")
    assert pl._recv_frame(FakeConn(oversized + b"x")) is None
    # 非法 header -> None
    assert pl._recv_frame(FakeConn(b"not-a-number")) is None
    # 非 JSON body -> None
    bad = f"{5:010d}".encode("ascii") + b"hello"
    assert pl._recv_frame(FakeConn(bad)) is None


def test_apply_patch_handles_malicious_proxy_port(monkeypatch):
    """proxy_port 非法值回退默认，不崩溃。"""
    def fake_git(gargs, *, env=None, check=True):
        if gargs[0] == "status":
            return ""
        if gargs[0] == "fetch":
            assert env["HTTPS_PROXY"].endswith(":7897")  # 回退默认端口
            return ""
        if gargs[0] == "rev-parse":
            return "a" * 40
        return ""

    monkeypatch.setattr(pl, "_git", fake_git)
    frame = _valid_payload(proxy_port="evil")
    status, detail = pl._apply_patch(frame, no_clean=False, logger=pl._log_setup())
    assert status == "applied"


def test_recv_frame_length_header():
    import socket
    frame = {"schema": "qlh.patch_frame.v1", "commit_sha": "x" * 40}
    data = json.dumps(frame).encode("utf-8")
    wire = f"{len(data):010d}".encode("ascii") + data

    class FakeConn:
        def __init__(self, payload):
            self._payload = payload
            self._pos = 0

        def recv(self, n):
            chunk = self._payload[self._pos:self._pos + n]
            self._pos += len(chunk)
            return chunk

    parsed = pl._recv_frame(FakeConn(wire))
    assert parsed["commit_sha"] == "x" * 40
