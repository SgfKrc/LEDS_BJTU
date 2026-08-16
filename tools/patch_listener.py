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
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packaging"))

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
_ACCEPTED_REPO = os.environ.get("QLH_PATCH_REPO", "")  # 空 = 接受任意（默认）


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


def _verify_frame(frame: dict, public_key_path: Path, logger: logging.Logger,
                  branch: str = DEFAULT_BRANCH) -> str | None:
    """验签 + schema/repo/branch 校验。返回拒绝原因（None=通过）。"""
    if frame.get("schema") != FRAME_SCHEMA:
        return f"unknown schema: {frame.get('schema')}"
    try:
        key_info = load_public_key_file(public_key_path)
    except SigningError as exc:
        return f"public key unreadable: {exc}"
    if _ACCEPTED_REPO and frame.get("repo") != _ACCEPTED_REPO:
        return f"repo mismatch: {frame.get('repo')}"
    if frame.get("branch") and frame["branch"] != branch:
        return f"branch mismatch: {frame.get('branch')}"
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


def _apply_patch(frame: dict, *, no_clean: bool, logger: logging.Logger
                 ) -> tuple[str, str]:
    """强拉 dev 分支对齐 commit_sha；返回 (status, detail)。"""
    branch = frame["branch"]
    try:
        proxy_port = int(frame.get("proxy_port") or DEFAULT_PROXY_PORT)
    except (TypeError, ValueError):
        proxy_port = DEFAULT_PROXY_PORT  # 恶意/非法值回退默认，不崩溃
    target = frame["commit_sha"]

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
        return "failed", f"fetch 失败（代理 {proxy_port} 不可达？）: {last_err}"

    # 强拉覆盖
    _git(["reset", "--hard", f"origin/{branch}"])
    if not no_clean:
        _git(["clean", "-fd"])
    head = _git(["rev-parse", "HEAD"])
    if head != target:
        return "failed", f"HEAD {head[:12]} != 目标 {target[:12]}"
    return "applied", target


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从节点补丁监听器（M2）")
    ap.add_argument("--verify-key", required=True, help="Ed25519 公钥文件（.pub.json）")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--no-clean", action="store_true", help="强拉后不 clean")
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args(argv)

    logger = _log_setup()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(8)
        logger.info("补丁监听器就绪 %s:%d（branch=%s, key=%s）",
                    args.host, args.port, args.branch,
                    Path(args.verify_key).name)
        while True:
            conn, addr = server.accept()
            with conn:
                try:
                    frame = _recv_frame(conn)
                    if frame is None:
                        conn.sendall(b"failed:empty")
                        continue
                    ack = _handle_frame(frame, args, logger)
                    conn.sendall(ack.encode("utf-8", errors="replace"))
                except Exception as exc:  # noqa: BLE001 - 单帧失败不杀监听
                    logger.error("处理帧异常: %s", exc)
                    try:
                        conn.sendall(f"failed:{exc}".encode("utf-8"))
                    except OSError:
                        pass


if __name__ == "__main__":
    sys.exit(main())
