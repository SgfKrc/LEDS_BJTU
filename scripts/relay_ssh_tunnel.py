#!/usr/bin/env python
"""relay_ssh_tunnel.py —— 跨机 relay 段的 **SSH 隧道编排**（三机验收 T3/T4 的前提）。

为什么需要它（`docs/边缘最小发行包审计清单-2026-09-26.md` 的「三机组网验收」缺口 ①）
------------------------------------------------------------------------------------
`relay_l2l_net_probe.py` 的 `--tail-endpoint` / `--middle-endpoint` **只接受 loopback 端点**
（这是有意的安全约束：跨机必须走本地 SSH 隧道，见 `src/relay_transport.py` 的
`is_loopback_host`）。而 `y700-1` / `tablet-2tlucnu8` 在 Tailnet 上、不在本机 loopback
⇒ **必须**把远端的段服务经 `ssh -L` 映射到本机 loopback。

此前这件事只**手工**做过（每次敲一长串 `ssh -N -L ...`），既无法复现也容易写错端口
⇒ 本脚本把它固化，并把"探活"接到**协议级**的 `relay_health.py --probe`（不是"端口能连"）。

用法::

    # 起隧道并探活（默认动作）
    python scripts/relay_ssh_tunnel.py open --target y700 --remote-port 50310 --local-port 50311

    # 只探活（隧道已在别处起好）
    python scripts/relay_ssh_tunnel.py check --local-port 50311

    # 停掉（按 pid 文件）
    python scripts/relay_ssh_tunnel.py close --local-port 50311

目标别名（`--target`）走本机 `~/.ssh/config`；Y700 的 `Port 8022` 已在该 config 里
（Tailscale SSH **不支持 Android 作 server** ⇒ 只能走 Termux 的 sshd）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

PID_DIR = ROOT / "build" / "relay-tunnels"

#: 已知目标别名（仅作提示，实际连接由 ssh 自己读 config 决定）。
KNOWN_TARGETS = {
    "surface": "surface@100.100.52.106（Windows；Tailscale 直连）",
    "y700": "y700-1（Android/Termux；config 里 Port 8022，Tailscale SSH 不支持 Android 作 server）",
    "y700-ip": "同上（按 IP 直连）",
    "y700-lan": "同上（局域网直连，可能不同网段而超时）",
}


def _pid_path(local_port: int) -> Path:
    return PID_DIR / f"tunnel-{local_port}.pid"


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ssh_executable() -> str:
    found = shutil.which("ssh")
    if not found:
        raise SystemExit("FAIL: 找不到 ssh 可执行文件")
    return found


def _wait_port(port: int, timeout: float, *, expect_open: bool = True) -> bool:
    """等本机端口达到期望状态。**只证"端口状态"，不证"对端协议活着"** —— 后者交给 `check`。"""
    deadline = time.monotonic() + max(0.1, timeout)
    while time.monotonic() < deadline:
        if _port_open("127.0.0.1", port) == expect_open:
            return True
        time.sleep(0.25)
    return False


def open_tunnel(target: str, remote_port: int, local_port: int, *,
                ssh_port: int | None = None, wait: float = 20.0) -> dict:
    """起一条 `ssh -N -L 127.0.0.1:local:127.0.0.1:remote <target>` 隧道（后台 + pid 文件）。

    远端侧监听地址**固定为 `127.0.0.1`** —— 段服务本身也按 loopback 起（见 `relay_mid_service`
    的 `--listen`），隧道只负责把"远端的 loopback"铺到"本机的 loopback"。
    """
    if target not in KNOWN_TARGETS:
        print(f"[warn] --target {target!r} 不在已知别名里（{sorted(KNOWN_TARGETS)}）；"
              "仍会按原样交给 ssh（它自己读 ~/.ssh/config）", file=sys.stderr)
    if _port_open("127.0.0.1", local_port):
        raise SystemExit(
            f"FAIL: 本机 127.0.0.1:{local_port} 已被占用 —— 换 --local-port，"
            "或先 close 掉旧隧道"
        )

    PID_DIR.mkdir(parents=True, exist_ok=True)
    log_path = PID_DIR / f"tunnel-{local_port}.log"
    argv = [_ssh_executable(), "-N", "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15",
            "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}"]
    if ssh_port is not None:
        argv += ["-p", str(ssh_port)]
    argv.append(target)

    # ★ 2026-09-28（三机验收实测）：Windows 上 `DETACHED_PROCESS` **留不住** ssh —— 隧道进程
    #   仍随本 shell 结束被回收（实测：起完立刻探活 healthy=True，几分钟后端口已释放）。
    #   ⇒ Windows 走 **schtasks**（与 Surface 上起长服务的做法一致）：写一个 `.bat` 再注册计划任务，
    #   由任务计划程序托管 ⇒ 真正脱离本 shell。
    #   POSIX 侧保留原逻辑（`start_new_session=True` 足够）。
    task_name = f"qlh-relay-tunnel-{local_port}"
    process = None
    if os.name == "nt":  # pragma: no cover - 平台相关
        bat_path = PID_DIR / f"tunnel-{local_port}.bat"
        quoted = " ".join(f'"{part}"' if " " in part else part for part in argv)
        bat_path.write_text(
            "@echo off\r\n" + quoted + f" > \"{log_path}\" 2>&1\r\n", encoding="ascii"
        )
        subprocess.run(["schtasks", "/create", "/tn", task_name, "/tr", str(bat_path),
                        "/sc", "once", "/st", "23:59", "/f"],
                       capture_output=True, check=False)
        subprocess.run(["schtasks", "/run", "/tn", task_name],
                       capture_output=True, check=False)
    else:
        log_handle = log_path.open("wb")
        process = subprocess.Popen(argv, stdout=log_handle, stderr=log_handle,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    _pid_path(local_port).write_text(
        str(process.pid) if process is not None else task_name, encoding="utf-8"
    )

    alive = _wait_port(local_port, wait)
    result = {
        "action": "open", "target": target, "local_port": local_port,
        "remote_port": remote_port,
        "pid": process.pid if process is not None else None,
        "task": task_name if process is None else None,
        "listening": alive, "log": str(log_path),
        "argv": argv,
    }
    if not alive:
        # 别把死隧道留在后台（pid 文件/计划任务都可能指向已退出的东西）
        detail = ""
        try:
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
        except OSError:
            pass
        print(f"FAIL: 隧道未在 {wait}s 内监听 127.0.0.1:{local_port}\n{detail}", file=sys.stderr)
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass
        else:  # pragma: no cover - 平台相关
            subprocess.run(["schtasks", "/end", "/tn", task_name], capture_output=True, check=False)
            subprocess.run(["schtasks", "/delete", "/tn", task_name, "/f"],
                           capture_output=True, check=False)
        result["listening"] = False
    else:
        where = f"pid={process.pid}" if process is not None else f"task={task_name}"
        print(f"[tunnel] {where} 127.0.0.1:{local_port} -> {target}:{remote_port}")
    return result


def check_tunnel(local_port: int, *, timeout: float = 8.0) -> dict:
    """**协议级**探活 —— 复用 `relay_health.py --probe`（`CLOSE`→`TOKEN(-1)` 握手）。

    ⚠️ 刻意不用"端口能连"当判据：`relay_health --self-check` 里已如实复现过那种漏报
    （**端口在监听但对端已死**会判健康）。
    """
    probe = ROOT / "scripts" / "relay_health.py"
    if not probe.is_file():
        raise SystemExit(f"FAIL: 找不到 {probe}")
    argv = [sys.executable, str(probe), "--probe", f"tunnel=tcp:127.0.0.1:{local_port}",
            "--json"]
    proc = subprocess.run(argv, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=max(5.0, timeout))
    payload: dict = {}
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except ValueError:
                continue
    healthy = bool(payload.get("healthy"))
    out = {"action": "check", "local_port": local_port, "healthy": healthy,
           "reason": payload.get("reason"), "exit_code": proc.returncode}
    print(f"[check] 127.0.0.1:{local_port} healthy={healthy} reason={payload.get('reason')}")
    return out


def close_tunnel(local_port: int) -> dict:
    """停掉隧道（**先列出将停的对象**，与本仓"删除前 dry-run"的规矩一致）。

    ⚠️ pid 文件里存的可能是 **pid**（POSIX）或**计划任务名**（Windows，见 `open_tunnel`）
    —— 两种都要能停。
    """
    path = _pid_path(local_port)
    if not path.is_file():
        print(f"[close] 没有 pid 文件（{path}）⇒ 无事可做")
        return {"action": "close", "local_port": local_port, "pid": None, "stopped": False}
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        raw = ""
    if not raw:
        path.unlink(missing_ok=True)
        return {"action": "close", "local_port": local_port, "pid": None, "stopped": False}

    stopped = False
    if raw.lstrip("-").isdigit():
        pid = int(raw)
        print(f"[close] 将停 pid={pid}（127.0.0.1:{local_port} 的隧道）")
        try:
            if os.name == "nt":  # pragma: no cover - 平台相关
                subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                               capture_output=True, check=False)
            else:  # pragma: no cover - 平台相关
                os.kill(pid, 15)
            stopped = True
        except OSError as exc:
            print(f"[close] 停 pid={pid} 失败：{exc}", file=sys.stderr)
    else:
        task_name = raw
        print(f"[close] 将停计划任务 {task_name}（127.0.0.1:{local_port} 的隧道）")
        subprocess.run(["schtasks", "/end", "/tn", task_name], capture_output=True, check=False)
        subprocess.run(["schtasks", "/delete", "/tn", task_name, "/f"],
                       capture_output=True, check=False)
        stopped = True
        pid = None
    path.unlink(missing_ok=True)
    if stopped and _port_open("127.0.0.1", local_port):
        print(f"[close] ⚠️ pid 已停但 127.0.0.1:{local_port} 仍在监听（可能是别的进程）",
              file=sys.stderr)
    return {"action": "close", "local_port": local_port, "pid": pid, "stopped": stopped}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    ap = argparse.ArgumentParser(description="跨机 relay 段的 SSH 隧道编排")
    ap.add_argument("action", choices=("open", "check", "close"))
    ap.add_argument("--target", default=None, help=f"ssh 别名；已知：{sorted(KNOWN_TARGETS)}")
    ap.add_argument("--remote-port", type=int, default=None, help="远端段服务端口")
    ap.add_argument("--local-port", type=int, required=True, help="本机 loopback 端口")
    ap.add_argument("--ssh-port", type=int, default=None, help="ssh 端口（缺省走 config）")
    ap.add_argument("--wait", type=float, default=20.0, help="等隧道监听的最长秒数")
    args = ap.parse_args(argv)

    if args.action == "open":
        if not args.target or not args.remote_port:
            raise SystemExit("FAIL: open 需要 --target 与 --remote-port")
        result = open_tunnel(args.target, args.remote_port, args.local_port,
                             ssh_port=args.ssh_port, wait=args.wait)
        if result["listening"]:
            result["check"] = check_tunnel(args.local_port)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["listening"] and result.get("check", {}).get("healthy") else 1
    if args.action == "check":
        result = check_tunnel(args.local_port)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["healthy"] else 1
    result = close_tunnel(args.local_port)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
