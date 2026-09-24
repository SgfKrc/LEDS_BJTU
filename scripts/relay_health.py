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
import threading
from pathlib import Path
from typing import Any

def _ensure_transport_importable() -> None:
    """把本仓的 `src/` 放进 `sys.path` —— **多点探索**，不假定脚本一定在 `scripts/` 下。

    为什么不能只写 `parents[1] / "src"`：脚本被复制/搬迁时（例如
    `tests/test_ci_relay_gates.py` 用 `tmp_path` 副本做**变异测试**）那一条会**指飞** ⇒
    `import relay_transport` 失败 ⇒ 探活退化成 `relay_transport_unavailable` ——
    看着"判死"，其实**什么都没测**。候选按"离得近"排，且只在真看到 `relay_transport.py`
    时才插入条目。
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "src",   # 常规：脚本在 <repo>/scripts/ 下
        here.parents[1],
        Path.cwd() / "src",        # 从仓库根调用（测试与 CI 的 cwd 都是仓库根）
        Path.cwd(),
    ]
    for candidate in candidates:
        try:
            if (candidate / "relay_transport.py").is_file() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        except OSError:
            continue


_ensure_transport_importable()

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


def _probe_relay(endpoint: str, *, timeout: float) -> dict[str, Any]:
    """★ R-R9：**协议级**探活 —— 真做一次最小 Relay 握手（`CLOSE` → 服务端回 `TOKEN(-1)`）。

    ## 为什么 `_check_tcp` 不够（§8.6 的实测教训）

    远端段跑在 ssh 会话里时，一次网络抖动会把会话带走、**服务随之退出**，而**隧道端口仍在本机
    监听** ⇒ 新连接被"接受"后**立刻 reset**。此时 `_check_tcp`（只 `connect`、**不发送任何帧**）
    **会判健康** ⇒ 漏报；而现象与「模型层错误」难以区分（`docs/跨框架接力-当前有效基线与后续
    优化计划-2026-09-21.md` §8.6 末段）。

    补上这一半的手段是：**真的按协议问一句**。握手用 `CLOSE` —— 服务端只做会话收尾
    （**不经 runner、不加载模型**），却足以证明"对端真的在按协议服务"。帧格式**不在这里重复
    实现**（与 `src/relay_transport.py` 同源），避免两套 wire 定义漂移。

    与 `_check_ready`（心跳新鲜度）互补：后者发现"进程还在但引擎已废"，本函数发现
    "端口通但对端已死/已废"。
    """
    host, _, port_text = endpoint.rpartition(":")
    if not host or not port_text.isdigit():
        return {"ok": False, "reason": "bad_endpoint", "detail": f"需要 host:port，实得 {endpoint!r}"}

    sock = None
    try:
        from relay_transport import (
            RelayFrame,
            RelayFrameKind,
            _decode_token,
            recv_frame,
            send_frame,
        )

        sock = socket.create_connection((host, int(port_text)), timeout=timeout)
        sock.settimeout(timeout)
        send_frame(sock, RelayFrame(RelayFrameKind.CLOSE, 0))
        response = recv_frame(sock, max_payload_bytes=64)
        token = _decode_token(response, 0)
        if token != -1:
            return {"ok": False, "reason": "unexpected_close_ack", "endpoint": endpoint,
                    "detail": f"CLOSE 的应答应为 -1，实得 {token}"}
        return {"ok": True, "reason": "protocol_handshake_ok", "endpoint": endpoint}
    except ImportError as exc:      # 缺 src/relay_transport ⇒ 不谎报健康
        return {"ok": False, "reason": "relay_transport_unavailable", "endpoint": endpoint,
                "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001 - 任何协议/网络异常都意味着"该段不在服务"
        return {"ok": False, "reason": "handshake_failed", "endpoint": endpoint,
                "detail": f"{type(exc).__name__}: {exc}",
                "hint": "端口能连上但握手失败 => 对端很可能已退出（§8.6 的「接受后立即 reset」）；"
                        "注意这会与「模型层错误」现象相似"}
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


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


def self_check() -> int:
    """★ A10 + R-R9：用**本机 loopback** 验证判定方向（CI 兜底，不需要真机 / 网络）。

    四条方向断言：
    1. **活监听**（有人 accept）⇒ `_check_tcp` 判健康；
    2. **已释放**端口 ⇒ `_check_tcp` 判死；
    3. ★ **假服务**（accept 后**立刻 close**，复现 §8.6 的「端口在监听但服务已死」）⇒
       `_check_tcp` **判健康**（如实复现**已知漏报**）、`_probe_relay` **判死** ⇒
       证明协议级探活真的补上了那一半；
    4. **只监听但不应答** ⇒ `_probe_relay` 也必须判死（证明它真的在**等应答**，不是连上就算过）。

    只覆盖本机可判定的两路：`--ssh-ready`（心跳新鲜度）依赖真实设备，不适合进 CI。
    """
    failures: list[str] = []
    live = socket.socket()
    live.bind(("127.0.0.1", 0))
    live.listen(1)
    live_endpoint = f"127.0.0.1:{int(live.getsockname()[1])}"

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_endpoint = f"127.0.0.1:{int(probe.getsockname()[1])}"
    probe.close()                     # 立刻释放 ⇒ 该端口上没有监听者

    try:
        live_result = _check_tcp(live_endpoint, timeout=2.0)
    finally:
        live.close()
    dead_result = _check_tcp(dead_endpoint, timeout=2.0)

    if live_result.get("ok") is not True:
        failures.append(f"活监听被判为不健康：{live_result}")
    if dead_result.get("ok") is not False:
        failures.append(f"已释放端口被判为健康：{dead_result}")

    # ★ R-R9 断言 3 + 4：一个「接受后立刻 close」的假服务 —— 正是 §8.6 描述的现象。
    fake = socket.socket()
    fake.bind(("127.0.0.1", 0))
    fake.listen(8)
    fake_endpoint = f"127.0.0.1:{int(fake.getsockname()[1])}"
    stop = threading.Event()

    def _serve_fake() -> None:
        while not stop.is_set():
            try:
                conn, _ = fake.accept()
            except OSError:
                return
            try:
                conn.close()          # 接受后立刻关闭 ⇒ §8.6 的「accept 后 reset」
            except OSError:
                pass

    thread = threading.Thread(target=_serve_fake, daemon=True)
    thread.start()
    try:
        fake_tcp = _check_tcp(fake_endpoint, timeout=2.0)
        fake_probe = _probe_relay(fake_endpoint, timeout=2.0)
    finally:
        stop.set()
        fake.close()
    thread.join(timeout=3)

    if fake_tcp.get("ok") is not True:
        failures.append(f"假服务应让 _check_tcp 判健康（这是**已知漏报**，用于证明它不够）："
                        f"{fake_tcp}")
    if fake_probe.get("ok") is not False:
        failures.append(f"假服务必须被 _probe_relay 判死（否则协议级探活失效）：{fake_probe}")

    # 断言 4：只监听、不应答（`live.listen` 但不 accept/不回帧）⇒ 握手必须失败
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)                 # 监听但从不 accept ⇒ 连接能建立、却永远等不到应答
    silent_endpoint = f"127.0.0.1:{int(silent.getsockname()[1])}"
    try:
        silent_probe = _probe_relay(silent_endpoint, timeout=1.0)
    finally:
        silent.close()
    if silent_probe.get("ok") is not False:
        failures.append(f"只监听、不应答的端口必须被 _probe_relay 判死：{silent_probe}")

    for item in failures:
        print(f"  - {item}")
    print(f"[verdict] 健康检查自检{'失败' if failures else '通过'}"
          f"（loopback：活={live_result.get('reason')} 死={dead_result.get('reason')} "
          f"假服务 tcp={fake_tcp.get('reason')}/probe={fake_probe.get('reason')} "
          f"沉默={silent_probe.get('reason')}）")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="跨机接力健康检查（TCP 探活 + 心跳新鲜度）")
    ap.add_argument("--check", action="append", default=[], metavar="NAME=tcp:HOST:PORT",
                    help="TCP 探活一项，可重复")
    ap.add_argument("--probe", action="append", default=[], metavar="NAME=tcp:HOST:PORT",
                    help="★ R-R9：**协议级**探活（真做一次 Relay 握手，不经 runner、不加载模型）—— "
                         "能发现 §8.6 那种「端口还在监听但对端已死」的情形；可重复")
    ap.add_argument("--ssh-ready", action="append", default=[], metavar="NAME=ALIAS:PATH",
                    help="经 ssh 读远端 ready 文件并检查心跳新鲜度，可重复")
    ap.add_argument("--stale-seconds", type=float, default=STALE_DEFAULT,
                    help=f"心跳过期阈值（秒，默认 {STALE_DEFAULT:g}）")
    ap.add_argument("--timeout", type=float, default=5.0, help="单项超时（秒）")
    ap.add_argument("--json", action="store_true", help="输出 JSON（供驱动 / CI 解析）")
    ap.add_argument("--self-check", action="store_true",
                    help="★ A10 + R-R9：CI 兜底自检 —— 用本机 loopback（活监听 / 已释放端口 / "
                         "「接受后立刻 close」的假服务 / 只监听不应答）验证判定方向，"
                         "不需要真机 / 网络；失败返回 1")
    args = ap.parse_args(argv)
    _enable_utf8_stdout()

    if args.self_check:
        # ★ A10：CI 兜底自检 —— 不探任何真实目标。
        return self_check()

    if not args.check and not args.ssh_ready and not args.probe:
        ap.error("至少给一个 --check / --probe / --ssh-ready")

    results: list[dict[str, Any]] = []
    for item in args.check:
        name, _, endpoint = item.partition("=")
        if not name or not endpoint:
            ap.error(f"--check 需要 NAME=tcp:HOST:PORT，实得 {item!r}")
        results.append({"name": name, "kind": "tcp",
                        **_check_tcp(endpoint.removeprefix("tcp:"), timeout=args.timeout)})
    for item in args.probe:
        name, _, endpoint = item.partition("=")
        if not name or not endpoint:
            ap.error(f"--probe 需要 NAME=tcp:HOST:PORT，实得 {item!r}")
        results.append({"name": name, "kind": "probe",
                        **_probe_relay(endpoint.removeprefix("tcp:"), timeout=args.timeout)})
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
