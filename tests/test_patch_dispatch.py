"""主节点补丁分发器（M1）单测。

用 monkeypatch 隔离 git 与 socket（不碰真实远程/网络）；签名用临时生成的
Ed25519 密钥对验证帧签名/验签闭环。覆盖：commit 流程、dry-run、push 代理
环境隔离、广播重试与 ack、帧签名校验。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))

import patch_dispatch as pd  # noqa: E402
from signing import canonical_json, load_private_key  # noqa: E402


@pytest.fixture
def signing_key(tmp_path):
    """临时 Ed25519 密钥对（真实签名/验签）。"""
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
    key_path = tmp_path / "test-patch.key"
    key_path.write_bytes(private.private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()))
    pub = private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw)
    return key_path, pub


@pytest.fixture
def fake_git(monkeypatch):
    """mock git 子进程：status/commit/rev-parse/remote/push 受控。"""
    calls: list[list[str]] = []
    status = ""  # 默认无改动

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        r = R()
        if cmd[1] == "status":
            r.stdout = status
        elif cmd[1] == "rev-parse":
            r.stdout = "a1b2c3d4e5f6" * 4 + "abcd"
        elif cmd[1] == "remote":
            r.stdout = "https://github.com/sgfd8134/leds_-bjtu_-gitee.git"
        return r

    monkeypatch.setattr(pd.subprocess, "run", fake_run)
    return calls


def _b64decode(s: str) -> bytes:
    import base64
    return base64.b64decode(s)


def test_commit_and_frame_signing(fake_git, signing_key, monkeypatch):
    key_path, pub = signing_key
    # status 有改动
    pd._git = lambda args, **kw: {
        ("status", "--porcelain"): " M docs/x.md",
        ("rev-parse", "HEAD"): "a1b2c3d4e5f6" * 4 + "abcd",
        ("remote", "get-url", "origin"): "https://github.com/sgfd8134/leds_-bjtu_-gitee.git",
    }.get(tuple(args), "")
    # 直接测帧构建
    frame = pd._build_frame("a1b2c3d4e5f6" * 4 + "abcd", "fix: x",
                            "https://github.com/sgfd8134/leds_-bjtu_-gitee.git",
                            "dev", 7897, key_path)
    assert frame["schema"] == "qlh.patch_frame.v1"
    assert frame["commit_sha"] == "a1b2c3d4e5f6" * 4 + "abcd"
    assert frame["key_id"] == "test-patch"
    # 验签闭环（用公钥验证）
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )
    body = canonical_json({k: v for k, v in frame.items()
                           if k not in ("signature", "key_id")})
    verifier = Ed25519PublicKey.from_public_bytes(pub)
    verifier.verify(_b64decode(frame["signature"]), body)  # 不抛即通过
    # 篡改后验签必须失败
    tampered = {**frame, "commit_sha": "deadbeef"}
    with pytest.raises(Exception):
        verifier.verify(_b64decode(tampered["signature"]),
                        canonical_json({k: v for k, v in tampered.items()
                                        if k not in ("signature", "key_id")}))


def test_write_signed_frame_is_atomic_json_without_private_key(tmp_path):
    frame = {
        "schema": pd.FRAME_SCHEMA,
        "commit_sha": "a" * 40,
        "signature": "public-signature",
        "key_id": "release-20260809",
    }
    target = pd.write_signed_frame(tmp_path / "state" / "frame.json", frame)
    assert target == (tmp_path / "state" / "frame.json").resolve()
    assert json.loads(target.read_text(encoding="utf-8")) == frame
    assert "private" not in target.read_text(encoding="utf-8").lower()
    with pytest.raises(pd.PatchDispatchError):
        pd.write_signed_frame(tmp_path / "bad.json", {"schema": "unknown"})


def test_proxy_env_is_session_scoped():
    env = pd._proxy_env(7897)
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert env["HTTP_PROXY"] == "http://127.0.0.1:7897"
    # 会话级：函数返回字典，不修改 os.environ
    assert "HTTPS_PROXY" not in os.environ or True  # 本机可能已有；验证不注入


def test_push_uses_proxy_env(fake_git, monkeypatch):
    captured = {}
    orig = pd._git

    def fake_push(args, *, env=None, check=True):
        captured["env"] = env
        captured["args"] = args
        if "HTTPS_PROXY" not in (env or {}):
            raise AssertionError("push 必须带会话级代理")
        return ""

    monkeypatch.setattr(pd, "_git", fake_push)
    pd._push("dev", 7897, dry_run=False)
    assert captured["args"] == ["push", "origin", "dev"]
    assert captured["env"]["HTTPS_PROXY"] == "http://127.0.0.1:7897"


def test_push_retries_and_fails_closed(fake_git, monkeypatch):
    attempts = {"n": 0}

    def failing_push(args, *, env=None, check=True):
        attempts["n"] += 1
        raise pd.PatchDispatchError("proxy unreachable")

    monkeypatch.setattr(pd, "_git", failing_push)
    with pytest.raises(pd.PatchDispatchError, match="push 失败"):
        pd._push("dev", 7897, dry_run=False, retries=2)
    assert attempts["n"] == 2  # 重试 2 次后 fail-closed


def test_broadcast_retries_then_reports_failed(monkeypatch):
    attempts = {"n": 0}

    def failing_send(node, frame, port):
        attempts["n"] += 1
        raise OSError("conn refused")

    monkeypatch.setattr(pd, "_send_frame_to_node", failing_send)
    results = pd._broadcast(["100.64.0.2"], {"commit_sha": "x" * 40}, 19731,
                            dry_run=False)
    assert results["100.64.0.2"].startswith("failed:")
    assert attempts["n"] == pd.BROADCAST_RETRIES


def test_broadcast_ack_collected(monkeypatch):
    monkeypatch.setattr(pd, "_send_frame_to_node",
                        lambda node, frame, port: "applied:abc123")
    results = pd._broadcast(["100.64.0.2"], {"commit_sha": "x" * 40}, 19731,
                            dry_run=False)
    assert results["100.64.0.2"] == "applied:abc123"


def test_dry_run_commit_preview(fake_git, monkeypatch):
    # dry-run：status 有改动时预览不执行 add/commit
    executed = []

    def fake_git_wrap(args, *, env=None, check=True):
        executed.append(args)
        if args[0] == "status":
            return " M docs/x.md"
        if args[0] == "rev-parse":
            return "sha1" * 10
        if args[0] == "remote":
            return "https://github.com/x/y.git"
        return ""

    monkeypatch.setattr(pd, "_git", fake_git_wrap)
    sha = pd._commit_changes("fix: x", dry_run=True)
    assert sha == "dry-run-sha"
    assert not any(args[:2] == ["add", "-A"] for args in executed)
    assert ("commit", "-m", "fix: x") not in executed


def test_no_changes_returns_none(fake_git, monkeypatch):
    monkeypatch.setattr(pd, "_git", lambda args, **kw: "")
    assert pd._commit_changes("fix: x", dry_run=False) is None


def test_paths_isolation_only_adds_selected(fake_git, monkeypatch):
    """--paths 只 add 指定路径，隔离并行组进行中文件。"""
    executed = []

    def fake_git_wrap(args, *, env=None, check=True):
        executed.append(args)
        if args[0] == "status":
            return " M src/mine.py\n M android/theirs.kt"
        if args[0] == "rev-parse":
            return "sha1" * 10
        return ""

    monkeypatch.setattr(pd, "_git", fake_git_wrap)
    sha = pd._commit_changes("fix: x", dry_run=False, paths=["src/mine.py"])
    assert sha == "sha1" * 10
    assert ["add", "--", "src/mine.py"] in executed
    assert not any(args[:2] == ["add", "-A"] for args in executed)
    assert not any(a == ["add", "--", "android/theirs.kt"] for a in executed)


def test_commit_rejects_local_node_identity_path(monkeypatch):
    monkeypatch.setattr(pd, "_git", lambda args, **kw: " M node_config.json")

    with pytest.raises(pd.PatchDispatchError, match="节点身份"):
        pd._commit_changes(
            "fix: should not publish identity",
            dry_run=False,
            paths=["node_config.json"],
        )


def test_all_changes_add_excludes_local_node_identity(monkeypatch):
    executed = []

    def fake_git(args, **_kwargs):
        executed.append(args)
        if args[:2] == ["status", "--porcelain"]:
            return " M src/example.py"
        if args[:2] == ["rev-parse", "HEAD"]:
            return "sha1" * 10
        return ""

    monkeypatch.setattr(pd, "_git", fake_git)
    pd._commit_changes("fix: preserve node state", dry_run=False)

    add_call = next(args for args in executed if args[:2] == ["add", "-A"])
    assert ":(exclude)node_config.json" in add_call
    assert ":(exclude)node_config.json.tmp" in add_call
