#!/usr/bin/env python3
"""从节点补丁监听器（M2）——按《双机补丁分发工具专项计划》§3.2。

常驻监听 TCP 端口，接收主节点补丁帧：
  验签（Ed25519）→ 校验 schema/repo/branch → dirty 告警（不拒绝，强制覆盖）
  → 强拉 dev 分支（reset --hard，7897 会话级代理）→ HEAD 对齐校验 → ack。

帧内 restart_requested 标记**只转交节点管理模块**（不自行重启，见 §2.0）。

用法：
  python tools/patch_listener.py --verify-key <pub.json>          # 前台运行
  python tools/patch_listener.py --port 19731 --branch dev
  python tools/patch_listener.py --no-clean                       # 强拉不 clean
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import select
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packaging"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from signing import (  # noqa: E402
    SigningError,
    canonical_json,
    load_public_key_file,
)

DEFAULT_PORT = 19731
DEFAULT_BRANCH = "dev"
DEFAULT_PROXY_PORT = 7897
FRAME_SCHEMA = "qlh.patch_frame.v1"
FETCH_RETRIES = 2
LOG_PATH = REPO_ROOT / "build" / "patch-listener.log"
PROTECTED_REPO_PATHS = ("node_config.json", "node_config.json.tmp")
_ACCEPTED_REPO = os.environ.get("QLH_PATCH_REPO", "")  # 空 = 使用 origin
PATCH_STATE_SCHEMA = "qlh.patch_listener.state.v1"
PATCH_STATE_FILE = "patch-listener-state.json"
MAX_PATCH_HISTORY = 32
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class PatchStateError(RuntimeError):
    """The local patch recovery journal is unavailable or invalid."""


def _log_setup() -> logging.Logger:
    logger = logging.getLogger("patch-listener")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    REPO_ROOT.joinpath("build").mkdir(exist_ok=True)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _git(args: list[str], *, env: dict | None = None,
         check: bool = True) -> str:
    merged = {**os.environ, **(env or {})}
    r = subprocess.run(["git", *args], cwd=str(REPO_ROOT), env=merged,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=180)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败: {r.stderr[-400:]}")
    return r.stdout.strip()


def _proxy_env(proxy_port: int) -> dict:
    proxy = f"http://127.0.0.1:{proxy_port}"
    return {"HTTPS_PROXY": proxy, "HTTP_PROXY": proxy,
            "https_proxy": proxy, "http_proxy": proxy}


def _configured_repo() -> str:
    """Return the configured repository, defaulting to this checkout origin."""
    if _ACCEPTED_REPO.strip():
        return _ACCEPTED_REPO.strip()
    try:
        return _git(["remote", "get-url", "origin"])
    except RuntimeError:
        return ""


def _patch_state_path() -> Path:
    """Store recovery metadata outside the checkout and its force-clean scope."""
    configured = os.environ.get("QLH_PATCH_STATE_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser()
    else:
        from config import STATE_DIR

        root = Path(STATE_DIR) / "patch-listener"
    return root.resolve() / PATCH_STATE_FILE


def _read_patch_state() -> dict:
    path = _patch_state_path()
    if not path.exists():
        return {
            "schema": PATCH_STATE_SCHEMA,
            "active": None,
            "history": [],
        }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatchStateError("patch recovery journal is unreadable") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != PATCH_STATE_SCHEMA
        or not isinstance(raw.get("history"), list)
    ):
        raise PatchStateError("patch recovery journal has an unsupported schema")
    active = raw.get("active")
    if active is not None and not isinstance(active, dict):
        raise PatchStateError("patch recovery journal has invalid active state")
    return raw


def _write_patch_state(state: dict) -> None:
    path = _patch_state_path()
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False,
        ) as handle:
            json.dump(state, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    except OSError as exc:
        try:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PatchStateError("patch recovery journal cannot be persisted") from exc


def _patch_operation_id(frame: dict) -> str:
    import hashlib

    payload = "\0".join([
        str(frame.get("repo", "")),
        str(frame.get("branch", "")),
        str(frame.get("commit_sha", "")),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_patch_state(
    frame: dict,
    phase: str,
    *,
    previous_head: str = "",
    current_head: str = "",
    error_code: str = "",
) -> None:
    state = _read_patch_state()
    record = {
        "operation_id": _patch_operation_id(frame),
        "target": str(frame["commit_sha"]),
        "branch": str(frame["branch"]),
        "phase": phase,
        "updated_at": int(time.time()),
    }
    if previous_head:
        record["previous_head"] = previous_head
    if current_head:
        record["current_head"] = current_head
    if error_code:
        record["error_code"] = error_code
    history = [item for item in state["history"] if isinstance(item, dict)]
    history.append(record)
    state["history"] = history[-MAX_PATCH_HISTORY:]
    state["active"] = record if phase in {"received", "verified", "applying"} else None
    state["latest"] = record
    _write_patch_state(state)


def _pending_state_for(frame: dict) -> dict | None:
    state = _read_patch_state()
    active = state.get("active")
    if active is None:
        return None
    if active.get("operation_id") != _patch_operation_id(frame):
        raise PatchStateError(
            "pending patch recovery requires replay of the existing signed target"
        )
    return active


def inspect_patch_recovery() -> dict:
    """Read-only recovery classification for listener startup/diagnostics."""
    state = _read_patch_state()
    active = state.get("active")
    if active is None:
        return {"status": "idle", "active": None}
    try:
        current_head = _git(["rev-parse", "HEAD"])
    except RuntimeError as exc:
        return {
            "status": "unavailable",
            "active": dict(active),
            "error": str(exc),
        }
    target = str(active.get("target", ""))
    previous_head = str(active.get("previous_head", ""))
    if current_head == target:
        status = "target_present_replay_required"
    elif previous_head and current_head == previous_head:
        status = "not_applied_replay_required"
    else:
        status = "manual_review_required"
    return {
        "status": status,
        "active": dict(active),
        "current_head": current_head,
    }


def _clean_args() -> list[str]:
    """Build a force-clean command that cannot remove node identity state."""
    args = ["clean", "-fd"]
    for path in PROTECTED_REPO_PATHS:
        args.extend(["-e", path])
    configured = os.environ.get("QLH_NODE_CONFIG_PATH", "").strip()
    if configured:
        try:
            relative = Path(configured).expanduser().resolve().relative_to(REPO_ROOT)
        except (OSError, ValueError):
            relative = None
        if relative is not None:
            value = relative.as_posix()
            if value not in PROTECTED_REPO_PATHS:
                args.extend(["-e", value])
    return args


def _verify_frame(frame: dict, public_key_path: Path, logger: logging.Logger,
                  branch: str = DEFAULT_BRANCH) -> str | None:
    """验签 + schema/repo/branch 校验。返回拒绝原因（None=通过）。"""
    if frame.get("schema") != FRAME_SCHEMA:
        return f"unknown schema: {frame.get('schema')}"
    try:
        key_info = load_public_key_file(public_key_path)
    except SigningError as exc:
        return f"public key unreadable: {exc}"
    expected_repo = _configured_repo()
    if not frame.get("repo"):
        return "repo missing"
    if not expected_repo:
        return "origin repo unavailable"
    if expected_repo and frame.get("repo") != expected_repo:
        return f"repo mismatch: {frame.get('repo')}"
    if frame.get("branch") != branch:
        return f"branch mismatch: {frame.get('branch')}"
    if not _COMMIT_SHA_RE.fullmatch(str(frame.get("commit_sha", ""))):
        return "commit_sha invalid"
    signature = frame.pop("signature", None)
    key_id = frame.pop("key_id", None)
    try:
        public_key = _load_public_key(key_info)
        public_key.verify(base64.b64decode(signature),
                          canonical_json(frame))
    except Exception as exc:
        return f"signature invalid: {exc}"
    finally:
        frame["signature"] = signature
        frame["key_id"] = key_id
    if key_info.get("key_id") and key_id != key_info["key_id"]:
        return f"key_id mismatch: {key_id}"
    return None


def _load_public_key(key_info: dict):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )
    raw = base64.b64decode(key_info["public_key"])
    return Ed25519PublicKey.from_public_bytes(raw)


def _current_branch() -> str:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"])


def _verify_target_on_remote(target: str, branch: str) -> str | None:
    """Require the signed target to be reachable from the fetched branch."""
    remote_tip = _git(["rev-parse", "--verify", f"origin/{branch}"])
    resolved_target = _git(["rev-parse", "--verify", f"{target}^{{commit}}"])
    if resolved_target != target:
        return "signed target commit is unavailable after fetch"
    merge_base = _git(["merge-base", target, remote_tip])
    if merge_base != target:
        return "signed target is not reachable from fetched branch"
    return None


def _apply_patch(frame: dict, *, no_clean: bool, logger: logging.Logger
                 ) -> tuple[str, str]:
    """Apply exactly the signed SHA with durable, replay-safe local recovery."""
    branch = frame["branch"]
    apply_started = False
    previous_head = ""
    try:
        proxy_port = int(frame.get("proxy_port") or DEFAULT_PROXY_PORT)
        if not 1 <= proxy_port <= 65535:
            raise ValueError("proxy port out of range")
    except (TypeError, ValueError):
        proxy_port = DEFAULT_PROXY_PORT  # 恶意/非法值回退默认，不崩溃
    target = frame["commit_sha"]

    try:
        pending = _pending_state_for(frame)
        if pending:
            logger.warning(
                "检测到未完成补丁操作，重放同一签名目标: target=%s phase=%s",
                target[:12], pending.get("phase", ""),
            )
            if pending.get("phase") == "applying":
                head = _git(["rev-parse", "HEAD"])
                if head == target and no_clean:
                    _record_patch_state(frame, "applied", current_head=head)
                    return "applied", target

        previous_head = _git(["rev-parse", "HEAD"])
        _record_patch_state(frame, "received", previous_head=previous_head)

        # dirty 告警（从节点为纯部署机，正常无改动；不拒绝，强制覆盖）
        dirty = _git(["status", "--porcelain"])
        if dirty:
            logger.warning("工作区 dirty（%d 项）——纯部署机异常，继续强制覆盖",
                           len(dirty.splitlines()))

        # fetch（代理会话级，重试）
        last_err = ""
        for attempt in range(FETCH_RETRIES):
            try:
                _git(["fetch", "origin", branch], env=_proxy_env(proxy_port))
                break
            except RuntimeError as exc:
                last_err = str(exc)
                if attempt < FETCH_RETRIES - 1:
                    time.sleep(2.0)
        else:
            _record_patch_state(frame, "failed", previous_head=previous_head,
                                error_code="fetch_failed")
            return "failed", f"fetch 失败（代理 {proxy_port} 不可达？）: {last_err}"

        reason = _verify_target_on_remote(target, branch)
        if reason:
            _record_patch_state(frame, "failed", previous_head=previous_head,
                                error_code="target_unavailable")
            return "failed", reason
        _record_patch_state(frame, "verified", previous_head=previous_head)
        _record_patch_state(frame, "applying", previous_head=previous_head)
        apply_started = True

        # Reset to the signed commit, never a moving branch ref.
        _git(["reset", "--hard", target])
        if not no_clean:
            _git(_clean_args())
        head = _git(["rev-parse", "HEAD"])
        if head != target:
            _record_patch_state(frame, "applying", previous_head=previous_head,
                                current_head=head, error_code="head_mismatch")
            return "failed", f"HEAD {head[:12]} != 目标 {target[:12]}"
        _record_patch_state(frame, "applied", previous_head=previous_head,
                            current_head=head)
        return "applied", target
    except PatchStateError as exc:
        logger.error("补丁恢复状态不可用: %s", exc)
        return "failed", str(exc)
    except RuntimeError as exc:
        if not apply_started:
            try:
                _record_patch_state(
                    frame, "failed", previous_head=previous_head,
                    error_code="git_apply_failed",
                )
            except PatchStateError:
                pass
        return "failed", f"patch apply failed: {exc}"


def _handle_frame(frame: dict, args, logger: logging.Logger) -> str:
    reason = _verify_frame(frame, Path(args.verify_key), logger,
                           branch=args.branch)
    if reason:
        logger.warning("帧拒绝: %s", reason)
        return f"rejected:{reason}"
    status, detail = _apply_patch(frame, no_clean=args.no_clean,
                                  logger=logger)
    logger.info("补丁应用: %s %s", status, detail)
    ack = f"{status}:{detail}"
    if frame.get("restart_requested"):
        # 转交节点管理模块（§2.0）：本工具不重启，仅记录
        logger.info("restart_requested=true——已转交节点管理模块（不自行重启）")
        ack += ";restart_requested"
    return ack


MAX_FRAME_BYTES = 4 * 1024 * 1024  # 帧大小上限（防恶意超大帧内存膨胀）


def _recv_frame(conn: socket.socket) -> dict | None:
    header = b""
    while len(header) < 10:
        chunk = conn.recv(10 - len(header))
        if not chunk:
            return None
        header += chunk
    try:
        length = int(header.decode("ascii"))
    except ValueError:
        return None
    if not 0 < length <= MAX_FRAME_BYTES:
        return None
    data = b""
    while len(data) < length:
        chunk = conn.recv(min(length - len(data), 1 << 20))
        if not chunk:
            return None
        data += chunk
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _serve(args, logger: logging.Logger) -> None:
    """Serve frames on IPv4 and IPv6 sockets when the platform supports both."""
    from network_address import canonical_host, create_listen_sockets

    bind_host = canonical_host(args.host)
    hosts = ["0.0.0.0", "::"] if bind_host in {"", "0.0.0.0"} else [bind_host]
    servers = create_listen_sockets(hosts, args.port, backlog=8, allow_partial=True)
    try:
        bound = ", ".join(
            f"{sock.getsockname()[0]}:{sock.getsockname()[1]}" for sock in servers
        )
        logger.info("补丁监听器就绪：%s（branch=%s, key=%s）",
                    bound, args.branch, Path(args.verify_key).name)
        while True:
            readable, _, _ = select.select(servers, [], [])
            for server in readable:
                conn, _addr = server.accept()
                with conn:
                    try:
                        frame = _recv_frame(conn)
                        if frame is None:
                            conn.sendall(b"failed:empty")
                            continue
                        ack = _handle_frame(frame, args, logger)
                        conn.sendall(ack.encode("utf-8", errors="replace"))
                    except Exception as exc:  # noqa: BLE001 - single frame must not kill listener
                        logger.error("处理帧异常: %s", exc)
                        try:
                            conn.sendall(f"failed:{exc}".encode("utf-8"))
                        except OSError:
                            pass
    finally:
        for server in servers:
            server.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从节点补丁监听器（M2）")
    ap.add_argument("--verify-key", required=True, help="Ed25519 公钥文件（.pub.json）")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--no-clean", action="store_true", help="强拉后不 clean")
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args(argv)

    logger = _log_setup()
    try:
        recovery = inspect_patch_recovery()
        if recovery["status"] != "idle":
            logger.warning("补丁恢复探针: %s", recovery["status"])
    except PatchStateError as exc:
        logger.error("补丁恢复探针不可用: %s", exc)
    return _serve(args, logger)


if __name__ == "__main__":
    sys.exit(main())
