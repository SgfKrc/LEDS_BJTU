#!/usr/bin/env python3
"""主节点补丁分发器（M1）——按《双机补丁分发工具专项计划》§3.1。

一键把修复代码提交并分发到从节点：
  git add -A + commit → 生成补丁帧（Ed25519 签名）→ push 远程 dev 分支
  （走 7897 会话级代理，不写全局 git config）→ TCP 推帧给从节点
  （静态清单模式）→ 收集 ack。

用法：
  python tools/patch_dispatch.py -m "fix: xxx"                    # 全流程
  python tools/patch_dispatch.py -m "fix: xxx" --nodes 100.64.0.2 # 指定从节点
  python tools/patch_dispatch.py -m "fix: xxx" --no-push          # 只 commit+广播
  python tools/patch_dispatch.py -m "fix: xxx" --dry-run          # 预览不执行
  python tools/patch_dispatch.py -m "fix: xxx" --proxy-port 7897  # 代理端口
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packaging"))

from signing import (  # noqa: E402
    SigningError,
    canonical_json,
    load_private_key,
)

DEFAULT_KEY = REPO_ROOT / "packaging" / ".signing-keys" / "release-20260809.key"
DEFAULT_PROXY_PORT = 7897
DEFAULT_BRANCH = "dev"
DEFAULT_PORT = 19731
FRAME_SCHEMA = "qlh.patch_frame.v1"
PUSH_RETRIES = 2          # push 重试次数（含首次）
BROADCAST_RETRIES = 3     # 每节点 TCP 推送重试
BROADCAST_RETRY_DELAY = 2.0
ACK_TIMEOUT = 10.0
PROTECTED_REPO_PATHS = ("node_config.json", "node_config.json.tmp")


class PatchDispatchError(RuntimeError):
    pass


def _git(args: list[str], *, env: dict | None = None, check: bool = True) -> str:
    """执行 git（可带会话级代理 env），返回 stdout。"""
    merged = {**os.environ, **(env or {})}
    r = subprocess.run(["git", *args], cwd=str(REPO_ROOT), env=merged,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    if check and r.returncode != 0:
        raise PatchDispatchError(f"git {' '.join(args)} 失败: {r.stderr[-400:]}")
    return r.stdout.strip()


def _proxy_env(proxy_port: int) -> dict:
    proxy = f"http://127.0.0.1:{proxy_port}"
    return {"HTTPS_PROXY": proxy, "HTTP_PROXY": proxy,
            "https_proxy": proxy, "http_proxy": proxy}


def _git_remote_url() -> str:
    url = _git(["remote", "get-url", "origin"])
    if not url:
        raise PatchDispatchError("未配置 git remote origin")
    return url


def _commit_changes(message: str, *, dry_run: bool,
                    paths: list[str] | None = None) -> str | None:
    """git add（--paths 或 -A）+ commit；无改动返回 None。

    --paths 用于隔离并行组进行中的文件：只提交本次补丁涉及的文件，
    避免 add -A 把别人的未提交代码混进补丁 commit。
    """
    staged = _git(["status", "--porcelain"])
    if dry_run:
        if staged:
            shown = [ln for ln in staged.splitlines()
                     if not paths or any(p in ln for p in paths)]
            print(f"  [dry-run] 将提交 {len(shown)} 个文件"
                  + (f"（指定 {len(paths)} 个路径）" if paths else "") + "：")
            for line in shown[:10]:
                print(f"    {line}")
        else:
            print("  [dry-run] 无工作区改动，跳过 commit")
        return "dry-run-sha" if staged else None
    if not staged:
        return None
    if paths:
        protected = {
            p.replace("\\", "/").lstrip("./")
            for p in paths
        } & set(PROTECTED_REPO_PATHS)
        if protected:
            raise PatchDispatchError(
                "拒绝提交本地节点身份文件: " + ", ".join(sorted(protected))
            )
        for p in paths:
            _git(["add", "--", p])
    else:
        # Keep the guard even though .gitignore excludes these files: a local
        # checkout may have an older/modified ignore file, and identity data
        # must never enter a patch commit.
        add_args = ["add", "-A", "--", "."]
        for protected in PROTECTED_REPO_PATHS:
            add_args.extend([f":(exclude){protected}"])
        _git(add_args)
    _git(["commit", "-m", message])
    return _git(["rev-parse", "HEAD"])


def _build_frame(commit_sha: str, note: str, repo: str, branch: str,
                 proxy_port: int, key_path: Path) -> dict:
    """生成签名补丁帧（canonical JSON + Ed25519）。"""
    payload = {
        "schema": FRAME_SCHEMA,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": repo,
        "branch": branch,
        "commit_sha": commit_sha,
        "proxy_port": proxy_port,
        "fallback_repo": os.environ.get("QLH_PATCH_FALLBACK_REPO", ""),
        "note": note,
    }
    try:
        _key_id, private = load_private_key(key_path)
    except SigningError as exc:
        raise PatchDispatchError(f"签名密钥不可用: {exc}") from exc
    body = canonical_json(payload)
    payload["signature"] = _b64encode(private.sign(body))
    payload["key_id"] = _key_id
    return payload


def _b64encode(raw: bytes) -> str:
    import base64
    return base64.b64encode(raw).decode("ascii")


def _push(branch: str, proxy_port: int, *, dry_run: bool, retries: int = PUSH_RETRIES) -> None:
    """push 到远程分支（会话级代理，不写全局 git config）。"""
    if dry_run:
        print(f"  [dry-run] 将 push origin {branch}（代理 127.0.0.1:{proxy_port} 会话级）")
        return
    last_err = ""
    for attempt in range(retries):
        try:
            _git(["push", "origin", branch], env=_proxy_env(proxy_port))
            return
        except PatchDispatchError as exc:
            last_err = str(exc)
            if attempt < retries - 1:
                time.sleep(2.0)
    raise PatchDispatchError(f"push 失败（代理 {proxy_port} 不可达？）: {last_err}")


def _send_frame_to_node(node: str, frame: dict, port: int) -> str:
    """TCP 推送帧到单节点，返回 ack 文本（循环读到连接关闭，防分包）。"""
    data = json.dumps(frame, ensure_ascii=False).encode("utf-8")
    header = f"{len(data):010d}".encode("ascii")
    with socket.create_connection((node, port), timeout=ACK_TIMEOUT) as sock:
        sock.sendall(header + data)
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _broadcast(nodes: list[str], frame: dict, port: int, *, dry_run: bool) -> dict:
    """静态清单模式：逐节点 TCP 推送（重试），返回 {node: ack}。"""
    results: dict[str, str] = {}
    if dry_run:
        for node in nodes:
            print(f"  [dry-run] 将推帧到 {node}:{port}（commit {frame['commit_sha'][:10]}）")
            results[node] = "dry-run"
        return results
    for node in nodes:
        last_err = ""
        for attempt in range(BROADCAST_RETRIES):
            try:
                ack = _send_frame_to_node(node, frame, port)
                results[node] = ack
                print(f"  -> {node}: {ack}")
                break
            except OSError as exc:
                last_err = str(exc)
                if attempt < BROADCAST_RETRIES - 1:
                    time.sleep(BROADCAST_RETRY_DELAY)
        else:
            results[node] = f"failed:{last_err}"
            print(f"  -> {node}: 推送失败（重试 {BROADCAST_RETRIES} 次）: {last_err}")
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="主节点补丁分发器（M1）")
    ap.add_argument("-m", "--message", required=True, help="补丁提交信息")
    ap.add_argument("--nodes", default="", help="从节点 IP 列表（逗号分隔，如 100.64.0.2,100.64.0.3）")
    ap.add_argument("--no-push", action="store_true", help="只 commit+广播，不 push")
    ap.add_argument("--dry-run", action="store_true", help="预览不执行")
    ap.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="从节点监听端口")
    ap.add_argument("--key", default=str(DEFAULT_KEY), help="Ed25519 签名私钥路径")
    ap.add_argument("--paths", default="", help="只提交指定路径（逗号分隔，隔离并行组进行中文件）")
    args = ap.parse_args(argv)

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]

    current_branch = _git(["branch", "--show-current"])
    if current_branch != args.branch:
        raise PatchDispatchError(
            f"当前分支为 {current_branch or '<detached>'}，与目标分支 {args.branch} 不一致"
        )

    # 1. commit（--paths 隔离并行组文件）
    paths = [p.strip() for p in args.paths.split(",") if p.strip()] or None
    sha = _commit_changes(args.message, dry_run=args.dry_run, paths=paths)
    if sha is None:
        print("无工作区改动，无事可做")
        return 0

    # 2. 帧
    repo = _git_remote_url()
    frame = _build_frame(sha, args.message, repo, args.branch,
                         args.proxy_port, Path(args.key))
    print(f"补丁帧: commit {sha[:12]} branch={args.branch} repo={repo}")
    if args.dry_run:
        print(f"  [dry-run] 帧摘要: {json.dumps({k: v for k, v in frame.items() if k != 'signature'}, ensure_ascii=False)[:160]}")
        print(f"  [dry-run] 签名: {frame['signature'][:24]}... key_id={frame['key_id']}")

    # 3. push（除非 --no-push）
    if not args.no_push:
        _push(args.branch, args.proxy_port, dry_run=args.dry_run)
    elif not args.dry_run:
        print("  --no-push：跳过 push（从节点将拉不到新 commit，除非远程已存在）")

    # 4. 广播
    if not nodes:
        print("未指定 --nodes，跳过广播（当前仅支持静态节点清单）")
        return 0
    results = _broadcast(nodes, frame, args.port, dry_run=args.dry_run)
    failed = [n for n, a in results.items() if a.startswith("failed")]
    if failed:
        print(f"广播部分失败: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
