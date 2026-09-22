#!/usr/bin/env python
"""跨机接力的**健康检查**：把「TCP 探活 + 心跳新鲜度」变成可被驱动 / CI 调用的命令。

## 为什么需要这个独立命令

远端段会**静默消失**，而且它消失时的现象与「模型算错」难以区分 —— 这在本项目里实际发生过两次：

1. 远端服务跑在 ssh 会话里 ⇒ 一次网络抖动把会话带走、**服务随之退出**，而本机隧道端口
   **仍在监听** ⇒ 新连接被"接受"后立刻 reset（`ConnectionResetError`），看起来像模型层错误；
2. `CLOSE` 帧误触发 `runner.close()` ⇒ 引擎被销毁，此后**每个连接**都失败（`rc=-5 参数非法`），
   帧却是完全正确的。

两种故障各需要一半的检查手段：

- **TCP 探活**能发现「端口通、但服务其实不在」（案例 1 的对端已死）；
- **心跳新鲜度**能发现「进程还在、但已经不再服务」（案例 2 的引擎已废）。

所以本命令把两者合在一起，并给出**可进 CI 的退出码**：0 = 全健康，1 = 有异常。

## 用法

    # 只探 TCP
    python scripts/relay_health.py --check head=tcp:127.0.0.1:50187 \
                                   --check middle=tcp:127.0.0.1:50190 \
                                   --check tail=tcp:127.0.0.1:50188

    # 结合服务端心跳（经 ssh 读远端 ready 文件，检查 alive_at 是否新鲜）
    python scripts/relay_health.py --ssh-ready head=y700:/data/data/com.termux/files/home/qlh-keephead/head.ready \
                                   --ssh-ready surface=surface@100.100.52.106:C:/Users/surface/qlh-keephead/mid.ready \
                                   --stale-seconds 30

    # 机器可读（CI / 驱动调用）
    python scripts/relay_health.py --check tail=tcp:127.0.0.1:50188 --json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import socket
import subprocess
import sys
from typing import Any

STALE_DEFAULT = 30.0
_SAFE_REMOTE_PATH = re.compile(r"^[A-Za-z0-9_./:\\-]+$")


def _remote_cat_command(path: str) -> str:
    """Build a shell-safe read command for the configured SSH endpoint."""
    if not isinstance(path, str) or not path:
        raise ValueError("ready path must be non-empty")
    # SSH executes the final argument through the remote user's shell.  The
    # target may be POSIX or Windows, so a shell-specific quote alone is not
    # sufficient; reject metacharacters before applying POSIX quoting.
    if not _SAFE_REMOTE_PATH.fullmatch(path):
        raise ValueError("ready path contains shell metacharacters")
    return f"cat -- {shlex.quote(path)}"


def _enable_utf8_stdout() -> None:
    """让输出在 GBK 控制台下也**不会因为一个符号就崩掉**。

    真踩过：`⇒`（U+21D2）在 cp936 控制台无法编码 ⇒ `UnicodeEncodeError` 直接把健康检查打成
    traceback —— 一个本该只报"健康/异常"的工具，反而比不检查更吵。这里降级为 replace，
    并且输出里只用 ASCII 箭头（见 `=>`）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _check_tcp(endpoint: str, *, timeout: float) -> dict[str, Any]:
    """TCP 探活。**只判断"有没有东西在监听并能接受连接"**，不发送任何协议帧。"""
    host, _, port_text = endpoint.rpartition(":")
    if not host or not port_text.isdigit():
        return {"ok": False, "reason": "bad_endpoint", "detail": f"需要 host:port，实得 {endpoint!r}"}
    try:
        with socket.create_connection((host, int(port_text)), timeout=timeout):
            return {"ok": True, "reason": "listening", "endpoint": endpoint}
    except OSError as exc:
        return {"ok": False, "reason": "unreachable", "endpoint": endpoint, "detail": str(exc),
                "hint": "远端服务可能已退出（例如随 ssh 会话被网络抖动静默带走），或隧道已断"}


def _check_ready(alias: str, path: str, *, timeout: float, stale_seconds: float) -> dict[str, Any]:
    """经 ssh 读远端 ready 文件，检查**心跳新鲜度**。

    ready 文件由服务端 `--heartbeat-interval` 定期刷新（含 `alive_at` / `pid`）。
    若 `alive_at` 距今超过 `stale_seconds` ⇒ **进程很可能已经不在了**（或卡死）。
    """
    target = f"{alias}:{path}"
    try:
        command = _remote_cat_command(path)
        done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(timeout)}",
                               alias, command],
                              capture_output=True, text=True, timeout=timeout + 10, check=False)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "reason": "ssh_failed", "target": target, "detail": str(exc)}
    if done.returncode != 0 or not done.stdout.strip():
        return {"ok": False, "reason": "no_ready_file", "target": target,
                "detail": (done.stderr or "").strip()[:200] or "ready 文件不存在或为空"}
    try:
        payload = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return {"ok": False, "reason": "bad_ready_json", "target": target, "detail": str(exc)}

    alive_at = str(payload.get("alive_at") or "")
    age: float | None = None
    if alive_at:
        try:
            stamp = dt.datetime.fromisoformat(alive_at.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("heartbeat timestamp has no timezone")
            age = (dt.datetime.now(dt.timezone.utc) - stamp.astimezone(dt.timezone.utc)).total_seconds()
        except ValueError:
            age = None
    fresh = age is not None and age <= stale_seconds
    reason = "heartbeat_ok" if fresh else "heartbeat_stale" if age is not None else "heartbeat_invalid"
    return {"ok": bool(fresh), "reason": reason,
            "target": target, "role": payload.get("role"), "pid": payload.get("pid"),
            "alive_at": alive_at, "age_s": None if age is None else round(age, 1),
            "stale_seconds": stale_seconds,
            "hint": None if fresh else "心跳过期 ⇒ 该段服务很可能已停止（或卡死）；重启前先确认进程"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="跨机接力健康检查（TCP 探活 + 心跳新鲜度）")
    ap.add_argument("--check", action="append", default=[], metavar="NAME=tcp:HOST:PORT",
                    help="TCP 探活一项，可重复")
    ap.add_argument("--ssh-ready", action="append", default=[], metavar="NAME=ALIAS:PATH",
                    help="经 ssh 读远端 ready 文件并检查心跳新鲜度，可重复")
    ap.add_argument("--stale-seconds", type=float, default=STALE_DEFAULT,
                    help=f"心跳过期阈值（秒，默认 {STALE_DEFAULT:g}）")
    ap.add_argument("--timeout", type=float, default=5.0, help="单项超时（秒）")
    ap.add_argument("--json", action="store_true", help="输出 JSON（供驱动 / CI 解析）")
    args = ap.parse_args(argv)
    _enable_utf8_stdout()

    if not args.check and not args.ssh_ready:
        ap.error("至少给一个 --check 或 --ssh-ready")

    results: list[dict[str, Any]] = []
    for item in args.check:
        name, _, endpoint = item.partition("=")
        if not name or not endpoint:
            ap.error(f"--check 需要 NAME=tcp:HOST:PORT，实得 {item!r}")
        results.append({"name": name, "kind": "tcp",
                        **_check_tcp(endpoint.removeprefix("tcp:"), timeout=args.timeout)})
    for item in args.ssh_ready:
        name, _, target = item.partition("=")
        alias, _, path = target.partition(":")
        if not name or not alias or not path:
            ap.error(f"--ssh-ready 需要 NAME=ALIAS:PATH，实得 {item!r}")
        results.append({"name": name, "kind": "ready",
                        **_check_ready(alias, path, timeout=args.timeout,
                                       stale_seconds=args.stale_seconds)})

    healthy = all(item["ok"] for item in results)
    if args.json:
        print(json.dumps({"healthy": healthy, "checks": results}, ensure_ascii=False))
    else:
        for item in results:
            mark = "OK  " if item["ok"] else "FAIL"
            detail = item.get("endpoint") or item.get("target") or ""
            extra = ""
            if item["kind"] == "ready" and item.get("age_s") is not None:
                extra = f" alive_at={item['alive_at']} age={item['age_s']}s pid={item.get('pid')}"
            print(f"  [{mark}] {item['name']:<8} {item['kind']:<5} {detail}{extra}"
                  f"{'' if item['ok'] else '  => ' + str(item.get('detail') or item.get('reason'))}")
            if not item["ok"] and item.get("hint"):
                print(f"         hint: {item['hint']}")
        print(f"[verdict] {'全部健康' if healthy else '存在异常段（详见上）'}")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
