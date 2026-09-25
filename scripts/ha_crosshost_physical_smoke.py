"""Run the HA-CROSSHOST-01 physical Surface/y700 smoke gate.

The Surface worker is streamed over SSH stdin and never written to the remote
disk.  The control frames travel through an SSH reverse TCP tunnel.  The y700
probe uses Android wireless debugging through an explicitly supplied or
currently online ``adb`` serial; it never assumes a fixed Android port.  This
is a physical process/network check, not a production availability claim.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cluster_transport import TransportEnvelope  # noqa: E402

from cluster_control_contract import (  # noqa: E402
    VoterSet,
    VoterSignature,
    validate_certificate,
)
from cluster_quorum import (  # noqa: E402
    QuorumCollector,
    QuorumError,
    QuorumVoter,
    SQLiteVoterLedger,
)

try:  # ★ 跨机 quorum 用 Ed25519 签票；缺 cryptography 时该模式 fail-loud（默认不跑）
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
        Ed25519PrivateKey,
    )
except Exception:  # noqa: BLE001
    Ed25519PrivateKey = None  # type: ignore[assignment]


REMOTE_WORKER = r'''
import argparse
import base64
import hashlib
import json
import socket
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--port", required=True, type=int)
args = parser.parse_args()
sys.path.insert(0, args.root + r"\src")
from cluster_transport import TransportContractError, TransportEnvelope

def send(out, value):
    out.sendall((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))

def read_line(inp):
    value = inp.readline()
    if not value:
        return None
    return json.loads(value.decode("utf-8"))

generation = 0
attempt_id = ""
received = {}
with socket.create_connection(("127.0.0.1", args.port), timeout=15) as sock:
    sock.settimeout(15)
    inp = sock.makefile("rb")
    send(sock, {"ok": True, "event": "ready"})
    while True:
        command = read_line(inp)
        if command is None:
            break
        try:
            operation = str(command.get("op", ""))
            if operation == "stop":
                send(sock, {"ok": True, "event": "stopped"})
                break
            if operation == "set_generation":
                generation = int(command["generation"])
                attempt_id = str(command["attempt_id"])
                received = {}
                send(sock, {"ok": True, "event": "generation_set", "generation": generation})
                continue
            if operation != "receive":
                raise TransportContractError("worker_operation_invalid", "unsupported operation")
            envelope = TransportEnvelope.decode(json.dumps(command["envelope"], sort_keys=True))
            payload = base64.b64decode(command["payload_b64"].encode("ascii"), validate=True)
            now_ms = int(command["now_ms"])
            if envelope.connection_generation != generation:
                raise TransportContractError("generation_stale", "old generation")
            if envelope.attempt_id != attempt_id:
                raise TransportContractError("attempt_fenced", "old attempt")
            if envelope.is_expired(now_ms=now_ms):
                raise TransportContractError("deadline_exceeded", "expired envelope")
            if envelope.payload_size != len(payload) or envelope.payload_digest != hashlib.sha256(payload).hexdigest():
                raise TransportContractError("payload_mismatch", "payload does not match envelope")
            previous = received.get(envelope.channel, -1)
            if envelope.sequence <= previous:
                raise TransportContractError("sequence_duplicate", "duplicate sequence")
            if envelope.sequence != previous + 1:
                raise TransportContractError("sequence_out_of_order", "non-contiguous sequence")
            received[envelope.channel] = envelope.sequence
            send(sock, {"ok": True, "event": "accepted", "generation": generation, "sequence": envelope.sequence})
        except TransportContractError as exc:
            send(sock, {"ok": False, "code": exc.code})
        except Exception as exc:
            send(sock, {"ok": False, "code": getattr(exc, "code", "worker_error")})
'''


class PhysicalSmokeError(RuntimeError):
    pass


#: ★ 跨机 quorum 的**远端 voter** worker：在 Surface 侧持有自己的 durable 账本（SQLite）与私钥，
#: 通过 SSH 反向 TCP 响应 `reserve_term` / `prepare` / `sign_vote` / `commit_certificate` / `snapshot`。
#: 设计意图与 `src/cluster_quorum.py` 里 `QuorumVoter` 的注释一致（"used by the collector and
#: **future transport**"）⇒ 本机侧用 `_RemoteVoter` 做同接口代理，**不改 quorum 代码**。
#: ⚠️ 私钥经 SSH 通道传入（不落远端磁盘、不进证据）；这是冒烟工装，不是生产密钥分发方式。
_REMOTE_VOTER = r'''
import argparse
import base64
import json
import socket
import sys

# ★ 配置**不经命令行**传入：SSH→cmd 的多层引号会把 JSON 的 {}"/, 吃掉（项目里反复踩过），
#   改为在下发源码时用 Python repr 字面量插入下面这一行（见 `_run_quorum_exchange`）。
# __QLH_VOTER_CONFIG__
parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--port", required=True, type=int)
args = parser.parse_args()
sys.path.insert(0, args.root + r"\src")
from cluster_control_contract import QuorumCertificate, VoterSet
from cluster_quorum import QuorumError, QuorumVoter, SQLiteVoterLedger
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

padding = "=" * (-len(PRIVATE_KEY_B64) % 4)
private_key = Ed25519PrivateKey.from_private_bytes(
    base64.urlsafe_b64decode(PRIVATE_KEY_B64 + padding))
voter_set = VoterSet.from_dict(json.loads(VOTER_SET_JSON))
ledger = SQLiteVoterLedger(LEDGER_PATH, voter_id=VOTER_ID,
                           cluster_id=voter_set.cluster_id,
                           voter_set_epoch=voter_set.voter_set_epoch)
voter = QuorumVoter(voter_id=VOTER_ID, private_key=private_key,
                    voter_set=voter_set, ledger=ledger)


def send(sock, value):
    sock.sendall((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def read(sock):
    data = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            return None
        if chunk == b"\n":
            break
        data.extend(chunk)
    return json.loads(bytes(data).decode("utf-8"))


with socket.create_connection(("127.0.0.1", args.port), timeout=30) as sock:
    sock.settimeout(30)
    send(sock, {"ok": True, "event": "voter_ready", "voter_id": VOTER_ID})
    while True:
        request = read(sock)
        if request is None:
            break
        op = request.get("op")
        try:
            if op == "reserve_term":
                term = voter.reserve_term(request["leader_id"], now_ms=request.get("now_ms"))
                send(sock, {"ok": True, "term": int(term)})
            elif op == "prepare":
                voter.prepare(request["leader_id"], int(request["term"]),
                              now_ms=request.get("now_ms"))
                send(sock, {"ok": True})
            elif op == "sign_vote":
                signature = voter.sign_vote(QuorumCertificate.from_dict(request["certificate"]),
                                            now_ms=request.get("now_ms"))
                send(sock, {"ok": True, "signature": signature.to_dict()})
            elif op == "commit_certificate":
                voter.commit_certificate(QuorumCertificate.from_dict(request["certificate"]),
                                         now_ms=request.get("now_ms"))
                send(sock, {"ok": True})
            elif op == "snapshot":
                send(sock, {"ok": True, "snapshot": voter.snapshot().to_dict()})
            elif op == "close":
                send(sock, {"ok": True})
                break
            else:
                send(sock, {"ok": False, "code": "unknown_op"})
        except QuorumError as exc:
            send(sock, {"ok": False, "code": str(getattr(exc, "code", "quorum_error"))})
        except Exception as exc:  # noqa: BLE001 - 只回稳定码，不回异常内容
            send(sock, {"ok": False, "code": "remote_voter_error",
                        "detail": type(exc).__name__})
'''


#: ★ HA-CROSSHOST-01 的 open 项「weak-network」：对**每一次控制帧往返**注入固定延迟。
#: ⚠️ 这是**应用层注入**，只验证「控制面在 RTT 被抬高时的行为（RTO / 超时 / 重连）」，
#: **不等于真实弱网**（没有丢包、抖动、带宽限制，也没有真机链路）—— 证据里必须带上注入值，
#: 不得当网络证据引用（见 `docs/主节点动态选举与分布式管理-P4.5立项-2026-09-21.md` §18）。
_WEAKNET_DELAY_S = 0.0


def _inject_delay() -> None:
    if _WEAKNET_DELAY_S > 0:
        time.sleep(_WEAKNET_DELAY_S)


# ---------------------------------------------------------------- ★ R-R2（§26）：丢包下的重试/退避
# §26 实测：丢包 5% 起冒烟即 failed，失败点是 `listener.accept()` 处**单次**连接不够 ——
# 经同一代理的 SSH 单次成功率约 1/3。⇒ 修法是**重试 + 指数退避**，且退避要有上限
# （弱网下"多试几次"胜过"一直等"，也不能让冒烟无限挂住）。
# ★ 2026-09-24（R-R2 物理复测，Surface 上线后）：5% 丢包下**端到端冒烟**（含反向隧道 + 控制帧）
#   的成功率**远低于** §26 记录的"SSH 单次建连 1/3"（实测 6 轮全失败）⇒ 3 次重试不足。
#   次数改为**可配**，便于扫参而不必改代码：`QLH_SSH_RETRY_ATTEMPTS`（默认仍 3 ⇒ 行为与旧版一致）。
def _retry_attempts_from_env(default: int = 3) -> int:
    """读 `QLH_SSH_RETRY_ATTEMPTS`（非法或缺失 ⇒ 用默认值，不抛）。"""
    raw = (os.environ.get("QLH_SSH_RETRY_ATTEMPTS") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, min(value, 16))


SSH_RETRY_ATTEMPTS = _retry_attempts_from_env()
SSH_RETRY_BASE_S = 0.5
SSH_RETRY_MAX_S = 4.0


def _control_timeout_from_env(default: float = 15.0) -> float:
    """读 `QLH_SSH_CONTROL_TIMEOUT_S`（**连上之后**等控制帧的秒数；非法或缺失 ⇒ 用默认值）。

    ★ R-R2 物理复测（Surface 上线后）：把 `QLH_SSH_RETRY_ATTEMPTS` 抬到 8 后，失败点从
    「建不起反向连」**后移**成「连上但控制帧超时」⇒ 弱网下真正偏紧的是**连上之后**的等待。
    ⚠️ **默认值保持不变** —— 本仓纪律是不擅自放宽既有阈值，故只做成可配，由验收方决定何时放宽。
    """
    raw = (os.environ.get("QLH_SSH_CONTROL_TIMEOUT_S") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(1.0, min(value, 600.0))


def retry_delays(attempts: int, *, base_s: float = SSH_RETRY_BASE_S,
                 max_s: float = SSH_RETRY_MAX_S) -> list[float]:
    """指数退避序列（**纯函数**，便于单测）：`base, base*2, base*4, …` 截到 `max_s`。

    返回长度 == `attempts`；**最后一次失败后不再睡**（调用方按 `index < len(delays)` 判）。
    """
    if attempts < 0:
        raise ValueError("attempts must be >= 0")
    return [min(base_s * (2 ** index), max_s) for index in range(int(attempts))]


def _first_frame_timeout_from_env(default: float = 15.0) -> float:
    """读 `QLH_SSH_FIRST_FRAME_TIMEOUT_S`（首帧等待秒数；非法或缺失 ⇒ 用默认值，不抛）。

    ★ R-R2 物理复测：`--weaknet-bridge` 的"整块丢弃交 TCP 重传"在 5% 丢包下会让**反向隧道建立**
    显著变慢 ⇒ 15 s 可能偏紧。做成可配即可扫参，不必改代码。
    """
    raw = (os.environ.get("QLH_SSH_FIRST_FRAME_TIMEOUT_S") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(1.0, min(value, 600.0))


def _connect_with_retry(listener: socket.socket, start_worker: Callable[[], Any], *,
                        attempts: int = SSH_RETRY_ATTEMPTS,
                        first_frame_timeout_s: float = None,
                        sleeper: Callable[[float], None] | None = None) -> tuple[Any, Any]:
    """**起 worker（SSH）+ 等它连回反向端口**，失败就重起重等（★ R-R2）。

    为什么必须重试：丢包链路下 SSH 单次建连成功率约 1/3（§26）⇒ 反向 TCP 端口从未连通 ⇒
    `listener.accept()` 干等超时 ⇒ 整个冒烟 failed。这里把「起进程 + accept」当作**一个可重试
    单元**：单次 `accept()` 超时不够，必须**换一条新 SSH 连接**再等。

    ⚠️ 每次失败都**回收**刚起的进程；全部失败时也**不留孤儿**（抛 `PhysicalSmokeError`）。
    返回 `(进程, 已连上的 socket)` —— 成功那次的进程**不**回收。
    """
    first_frame_timeout_s = (first_frame_timeout_s if first_frame_timeout_s is not None
                             else _first_frame_timeout_from_env())
    sleeper = sleeper or time.sleep
    total = max(1, int(attempts))
    # 退避只发生在两次尝试之间 ⇒ 共 total-1 次（最后一次失败后立刻放弃）
    delays = retry_delays(max(0, total - 1))
    last_error: Exception | None = None
    last_process: Any = None
    for index in range(total):
        process = last_process = start_worker()
        try:
            listener.settimeout(first_frame_timeout_s)
            connection, _address = listener.accept()
            return process, connection
        except (TimeoutError, OSError) as exc:
            last_error = exc
            if index < total - 1:
                # 中间失败：回收，再换一条新连接重试
                try:
                    process.kill()
                except Exception:  # noqa: BLE001  # 回收失败不掩盖原始连接错误
                    pass
        if index < len(delays):
            sleeper(delays[index])
    error = PhysicalSmokeError(
        f"worker_connect_retry_exhausted:{type(last_error).__name__}")
    # 最后一次的进程**不回收**：调用方要用它的 stderr 诊断，并负责收尾
    error.process = last_process  # type: ignore[attr-defined]
    raise error


def _send_line(sock: socket.socket, value: dict[str, Any]) -> None:
    sock.sendall((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def _recv_line(sock: socket.socket) -> dict[str, Any]:
    data = bytearray()
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise PhysicalSmokeError("physical worker disconnected")
        if chunk == b"\n":
            break
        data.extend(chunk)
        if len(data) > 64 * 1024:
            raise PhysicalSmokeError("physical worker response is too large")
    value = json.loads(bytes(data).decode("utf-8"))
    if not isinstance(value, dict):
        raise PhysicalSmokeError("physical worker response is not an object")
    return value


def _request(sock: socket.socket, value: dict[str, Any]) -> dict[str, Any]:
    # ★ 弱网模拟：发送前 + 接收前各注入一次 ⇒ 等效把控制面 RTT 抬高 2×delay（默认 0 = 旧行为）。
    _inject_delay()
    _send_line(sock, value)
    _inject_delay()
    return _recv_line(sock)


def _expect_ok(sock: socket.socket, value: dict[str, Any]) -> dict[str, Any]:
    response = _request(sock, value)
    if not response.get("ok"):
        raise PhysicalSmokeError(str(response.get("code", "worker_error")))
    return response


def _ssh_process(target: str, remote_root: str, remote_port: int, local_port: int,
                 proxy_command: str | None = None) -> subprocess.Popen[bytes]:
    command = f'cd /d "{remote_root}" && python - --root "{remote_root}" --port {remote_port}'
    options = ["BatchMode=yes", "ConnectTimeout=8", "ExitOnForwardFailure=yes"]
    if proxy_command:
        # ★ 真实弱网（网络层）：整条 SSH 通道经本地 TCP 代理（延迟/丢包注入）
        options.append(f"ProxyCommand={proxy_command}")
    process = subprocess.Popen(
        [
            "ssh",
            *[arg for option in options for arg in ("-o", option)],
            "-R",
            f"127.0.0.1:{remote_port}:127.0.0.1:{local_port}",
            target,
            command,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(REMOTE_WORKER.encode("utf-8"))
    process.stdin.close()
    return process


def _ssh_worker(target: str, remote_root: str, remote_port: int, local_port: int,
                source: str, extra_args: str = "") -> subprocess.Popen[bytes]:
    """与 `_ssh_process` 同机制（源码经 stdin 流式传入、控制帧走 SSH 反向 TCP），
    但可注入**任意 worker 源码**与附加参数 —— 供跨机 quorum voter 使用。"""
    command = (f'cd /d "{remote_root}" && python - {extra_args} '
               f'--root "{remote_root}" --port {remote_port}')
    process = subprocess.Popen(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=8",
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"127.0.0.1:{remote_port}:127.0.0.1:{local_port}",
            target,
            command,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(source.encode("utf-8"))
    process.stdin.flush()
    process.stdin.close()
    return process


def _stop_ssh(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=8)


def _public_key_b64(private_key: Any) -> str:
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415

    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _private_key_b64(private_key: Any) -> str:
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415

    raw = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption())
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class _RemoteVoter:
    """把 `QuorumVoter` 的同名方法转发到对端 voter worker —— 跨机代理，**不改 quorum 代码**。

    `QuorumCollector` 只按 duck typing 调用 `reserve_term` / `prepare` / `sign_vote` /
    `commit_certificate`，因此远端代理天然可替换本地 voter（这正是 `src/cluster_quorum.py`
    里 `QuorumVoter` 注释所说的 "future transport"）。
    """

    def __init__(self, sock: socket.socket, *, voter_id: str) -> None:
        self._sock = sock
        self.voter_id = voter_id

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        response = _request(self._sock, request)     # 复用既有收发（受弱网注入开关影响）
        if not response.get("ok"):
            raise QuorumError(str(response.get("code") or "quorum_error"),
                              "remote voter rejected the request")
        return response

    def reserve_term(self, leader_id: str, *, now_ms: int | None = None) -> int:
        return int(self._call({"op": "reserve_term", "leader_id": leader_id,
                               "now_ms": now_ms})["term"])

    def prepare(self, leader_id: str, term: int, *, now_ms: int | None = None) -> None:
        self._call({"op": "prepare", "leader_id": leader_id, "term": int(term),
                    "now_ms": now_ms})

    def sign_vote(self, certificate: Any, *, now_ms: int | None = None) -> VoterSignature:
        response = self._call({"op": "sign_vote", "certificate": certificate.to_dict(),
                               "now_ms": now_ms})
        return VoterSignature.from_dict(response["signature"])

    def commit_certificate(self, certificate: Any, *, now_ms: int | None = None) -> None:
        self._call({"op": "commit_certificate", "certificate": certificate.to_dict(),
                    "now_ms": now_ms})

    def snapshot(self) -> dict[str, Any]:
        return self._call({"op": "snapshot"})["snapshot"]


def _run_quorum_exchange(target: str, remote_root: str, *,
                         remote_state: str) -> dict[str, Any]:
    """★ HA-CROSSHOST-01 的 open 项「quorum certificate exchange」的**跨机实测**。

    配 3 个 voter（`QuorumPolicy.election_allowed` 要求 ≥3 ⇒ 两节点场景必须配 3 个），
    **只联系本机 `voter-a` + Surface `voter-b`** ⇒ `quorum_size == 2` ⇒ 证书的每一票都来自
    不同主机。三条判据：

    1. **跨机颁证**：`acquire` 成功、`signed_voters == (voter-a, voter-b)`、`validate_certificate` 通过；
    2. **fail-closed**：只联系 `voter-a` ⇒ `quorum_unavailable`（不许单机自签）；
    3. **对端 durability**：向 Surface 要 `snapshot`，确认该 term / 证书摘要**持久化在它的 SQLite 账本里**。
    """
    if Ed25519PrivateKey is None:
        raise PhysicalSmokeError("quorum_crypto_unavailable")
    voter_ids = ("voter-a", "voter-b", "voter-c")
    keys = {voter_id: Ed25519PrivateKey.generate() for voter_id in voter_ids}
    voter_set = VoterSet(cluster_id="qlh-physical-quorum", voter_set_epoch=1,
                         voters={voter_id: _public_key_b64(keys[voter_id])
                                 for voter_id in voter_ids})
    local_dir = tempfile.mkdtemp(prefix="qlh-quorum-a-")
    local_voter = QuorumVoter(
        voter_id="voter-a", private_key=keys["voter-a"], voter_set=voter_set,
        ledger=SQLiteVoterLedger(Path(local_dir) / "voter-a.sqlite", voter_id="voter-a",
                                 cluster_id=voter_set.cluster_id,
                                 voter_set_epoch=voter_set.voter_set_epoch))

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(20)
    local_port = int(listener.getsockname()[1])
    remote_port = 44000 + (local_port % 1000)
    voter_set_json = json.dumps(voter_set.to_dict(), sort_keys=True, separators=(",", ":"))
    source = _REMOTE_VOTER.replace(
        "# __QLH_VOTER_CONFIG__",
        "\n".join((
            f"VOTER_ID = {'voter-b'!r}",
            f"PRIVATE_KEY_B64 = {_private_key_b64(keys['voter-b'])!r}",
            f"VOTER_SET_JSON = {voter_set_json!r}",
            f"LEDGER_PATH = {remote_state!r}",
        )))
    events: list[dict[str, Any]] = []
    worker = None
    try:
        try:
            # ★ R-R2：起 voter 同样走「重起 + 重等」（`accept()` 超时是这里最常见的失败点）
            worker, connection = _connect_with_retry(
                listener,
                lambda: _ssh_worker(target, remote_root, remote_port, local_port, source))
        except (OSError, PhysicalSmokeError) as exc:
            # 远端 voter 没连上来 ⇒ 附上它的 stderr 便于诊断
            worker = getattr(exc, "process", None)
            detail = ""
            if worker is not None and worker.stderr is not None:
                try:
                    detail = worker.stderr.read(2000).decode("utf-8", "replace").strip()
                except OSError:
                    detail = ""
            raise PhysicalSmokeError(
                f"remote voter did not connect: {exc}; stderr={detail[:400]!r}") from exc
        connection.settimeout(20)
        with connection:
            ready = _recv_line(connection)
            if not ready.get("ok") or ready.get("event") != "voter_ready":
                raise PhysicalSmokeError("remote voter did not become ready")
            events.append({"event": "remote_voter_ready", "voter_id": ready.get("voter_id")})
            remote_voter = _RemoteVoter(connection, voter_id="voter-b")
            now_ms = int(time.time() * 1000)
            collector = QuorumCollector(voter_set=voter_set,
                                        voters={"voter-a": local_voter,
                                                "voter-b": remote_voter})
            single = collector.acquire("voter-a", available_voter_ids=("voter-a",),
                                       now_ms=now_ms)
            if single.accepted:
                raise PhysicalSmokeError("single voter was allowed to issue a certificate")
            events.append({"event": "single_voter_is_read_only", "reason": single.reason})
            outcome = collector.acquire("voter-a",
                                        available_voter_ids=("voter-a", "voter-b"),
                                        now_ms=now_ms)
            certificate = outcome.certificate
            if not outcome.accepted or certificate is None:
                raise PhysicalSmokeError(f"cross-host quorum failed: {outcome.reason}")
            signed = tuple(sorted(signature.voter_id for signature in certificate.signatures))
            if signed != ("voter-a", "voter-b"):
                raise PhysicalSmokeError(f"unexpected signers: {signed}")
            validate_certificate(certificate, voter_set, now_ms=now_ms)
            digest = certificate.digest()
            events.append({"event": "cross_host_certificate_issued", "term": outcome.term,
                           "signed_voters": list(signed),
                           "certificate_digest_prefix": digest[:16]})
            snapshot = remote_voter.snapshot()
            if (snapshot.get("active_leader") != "voter-a"
                    or not snapshot.get("active_certificate_digest")):
                raise PhysicalSmokeError("remote voter did not persist the certificate")
            if str(snapshot.get("active_certificate_digest")) != digest:
                raise PhysicalSmokeError("remote ledger digest does not match the certificate")
            events.append({"event": "remote_ledger_persisted",
                           "term": snapshot.get("active_term"),
                           "digest_prefix": digest[:16]})
            _expect_ok(connection, {"op": "close"})
    finally:
        if worker is not None:
            _stop_ssh(worker)
        listener.close()
    return {
        "status": "passed",
        "transport": "ssh_reverse_tcp",
        "voter_ids": list(voter_ids),
        "contacted_voters": ["voter-a", "voter-b"],
        "quorum_size": voter_set.quorum_size,
        "term": outcome.term,
        "signed_voters": list(signed),
        "certificate_digest_prefix": digest[:16],
        "remote_ledger": {"voter_id": snapshot.get("voter_id"),
                          "active_term": snapshot.get("active_term"),
                          "active_leader": snapshot.get("active_leader")},
        "events": events,
    }


def _run_surface(target: str, remote_root: str, *, long_steps: int = 1,
                 proxy_command: str | None = None) -> dict[str, Any]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(20)
    local_port = int(listener.getsockname()[1])
    remote_port = 43000 + (local_port % 1000)
    events: list[dict[str, Any]] = []
    payload = b"physical-ha-control-metadata"
    now_ms = int(time.time() * 1000)
    # ★ R-R2：丢包链路上**单次** SSH 建连常失败（§26 实测成功率 ~1/3）⇒ 把
    # 「起 worker + 等它连回反向端口」当作**可重试单元**（失败即换连接重来）。
    first_process = None
    try:
        first_process, connection = _connect_with_retry(
            listener,
            lambda: _ssh_process(target, remote_root, remote_port, local_port,
                                 proxy_command=proxy_command))
        connection.settimeout(_control_timeout_from_env())
        with connection:
            ready = _recv_line(connection)
            if not ready.get("ok") or ready.get("event") != "ready":
                raise PhysicalSmokeError("surface worker did not become ready")
            _expect_ok(connection, {"op": "set_generation", "generation": 1, "attempt_id": "physical-1"})
            first = TransportEnvelope.from_payload(
                payload,
                request_id="physical-accepted",
                connection_generation=1,
                attempt_id="physical-1",
                channel="control",
                sequence=0,
                deadline_ms=now_ms + 20_000,
            )
            accepted = _expect_ok(connection, {
                "op": "receive",
                "envelope": first.to_dict(),
                "payload_b64": base64.b64encode(payload).decode("ascii"),
                "now_ms": now_ms,
            })
            events.append({"event": "surface_control_frame_accepted", "generation": accepted["generation"]})
            requested_frames = max(1, int(long_steps))
            for sequence in range(1, requested_frames):
                long_frame = TransportEnvelope.from_payload(
                    payload,
                    request_id=f"physical-long-{sequence}",
                    connection_generation=1,
                    attempt_id="physical-1",
                    channel="control",
                    sequence=sequence,
                    deadline_ms=int(time.time() * 1000) + max(20_000, requested_frames * 100),
                )
                _expect_ok(connection, {
                    "op": "receive",
                    "envelope": long_frame.to_dict(),
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "now_ms": int(time.time() * 1000),
                })
            long_running = {
                "requested_frames": requested_frames,
                "accepted_frames": requested_frames,
                "transport": "ssh_reverse_tcp",
            }
            old = TransportEnvelope.from_payload(
                payload,
                request_id="physical-stale",
                connection_generation=1,
                attempt_id="physical-1",
                channel="control",
                sequence=0,
                deadline_ms=now_ms + 20_000,
            )
        failure_started = time.perf_counter()
        _stop_ssh(first_process)
        events.append({"event": "surface_worker_stopped", "reason": "simulated_crash"})

        # ★ R-R2：换新的 worker 同样走「重起 + 重等」
        second_process = None
        try:
            second_process, connection = _connect_with_retry(
                listener,
                lambda: _ssh_process(target, remote_root, remote_port, local_port,
                                     proxy_command=proxy_command))
            connection.settimeout(_control_timeout_from_env())
            with connection:
                ready = _recv_line(connection)
                if not ready.get("ok"):
                    raise PhysicalSmokeError("surface replacement worker did not become ready")
                _expect_ok(connection, {"op": "set_generation", "generation": 2, "attempt_id": "physical-2"})
                stale = _request(connection, {
                    "op": "receive",
                    "envelope": old.to_dict(),
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "now_ms": int(time.time() * 1000),
                })
                if stale.get("ok") or stale.get("code") != "generation_stale":
                    raise PhysicalSmokeError("Surface accepted an old generation")
                events.append({"event": "surface_old_generation_rejected", "code": "generation_stale"})
                second = TransportEnvelope.from_payload(
                    payload,
                    request_id="physical-reconnected",
                    connection_generation=2,
                    attempt_id="physical-2",
                    channel="control",
                    sequence=0,
                    deadline_ms=int(time.time() * 1000) + 20_000,
                )
                recovered = _expect_ok(connection, {
                    "op": "receive",
                    "envelope": second.to_dict(),
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "now_ms": int(time.time() * 1000),
                })
                events.append({"event": "surface_control_frame_accepted_after_reconnect", "generation": recovered["generation"]})
        finally:
            if second_process is not None:
                _stop_ssh(second_process)
        return {
            "status": "passed",
            "target": "surface",
            "physical_nodes": True,
            "transport": "ssh_reverse_tcp",
            "events": events,
            "rto_ms": max(0, int((time.perf_counter() - failure_started) * 1000)),
            "rpo": {"last_durable_sequence": 0, "lost_events": 0, "scope": "transport_only"},
            "long_running": long_running,
            "weaknet": {
                "inject_delay_ms_per_direction": int(_WEAKNET_DELAY_S * 1000),
                "effective_rtt_delta_ms": int(_WEAKNET_DELAY_S * 2000),
                "scope": "app_layer_control_frame_only",
                "note": "应用层延迟注入；不含丢包/抖动/带宽限制，**不是**真实弱网证据",
            },
        }
    finally:
        if first_process is not None:
            _stop_ssh(first_process)
        listener.close()


def _find_adb(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    discovered = shutil.which("adb")
    if discovered:
        return discovered
    sdk_candidates = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT")]
    if os.name == "nt":
        sdk_candidates.extend([
            str(Path.home() / "AppData" / "Local" / "Android" / "Sdk"),
            str(Path.home() / "Android" / "Sdk"),
        ])
    for sdk in sdk_candidates:
        if sdk:
            candidate = Path(sdk) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
            if candidate.is_file():
                return str(candidate)
    return None


def _discover_y700_serial(adb: str, host: str) -> str | None:
    completed = subprocess.run(
        [adb, "devices", "-l"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    online: list[str] = []
    for line in completed.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "device":
            online.append(fields[0])
    matching = [item for item in online if item == host or item.startswith(f"{host}:")]
    if len(matching) == 1:
        return matching[0]
    if not matching and len(online) == 1:
        return online[0]
    return None


def _ssh_works(target: str, *, timeout: int = 6, attempts: int = SSH_RETRY_ATTEMPTS,
               runner: Callable[..., Any] | None = None,
               sleeper: Callable[[float], None] | None = None) -> bool:
    """探测目标是否可用 SSH 登录 —— **带重试与指数退避**（★ R-R2，§26 丢包硬边界）。

    `runner` / `sleeper` 可注入（单测用）；默认走真 `subprocess.run` + `time.sleep`。
    """
    runner = runner or subprocess.run
    sleeper = sleeper or time.sleep
    # 同上：退避只在两次尝试之间发生 ⇒ 最后一次失败后不睡
    delays = retry_delays(max(0, int(attempts) - 1))
    for index in range(max(1, int(attempts))):
        try:
            completed = runner(
                ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
                 "-o", "StrictHostKeyChecking=accept-new", target, "true"],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
                check=False,
            )
            if completed.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
        if index < len(delays):
            sleeper(delays[index])
    return False
    """探测 Termux sshd 是否可达。

    为什么优先它：Android 无线调试的端口**每次都会变**（设备 `ro.debuggable=0`
    且 `persist.adb.tcp.port` 为空 ⇒ 无法在设备侧固定），而 Termux 的 sshd 端口是
    **固定 8022**（实测可用）。所以「能用 SSH 就用 SSH」才能真正摆脱动态端口。
    """
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
             "-o", "StrictHostKeyChecking=accept-new", target, "true"],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


# 探针命令与传输方式无关（`adb shell` 与 `ssh` 都能跑），因此两条路径共用同一份。
Y700_PROBE_COMMAND = (
    "printf 'model=%s\\n' \"$(getprop ro.product.model)\"; "
    "printf 'sdk=%s\\n' \"$(getprop ro.build.version.sdk)\"; "
    "printf 'abi=%s\\n' \"$(getprop ro.product.cpu.abilist)\"; "
    "printf 'nproc=%s\\n' \"$(nproc)\"; "
    "printf 'available_mem_kb=%s\\n' \"$(awk '/MemAvailable/ {print $2}' /proc/meminfo)\"; "
    "printf 'model_root=/sdcard/Download/QLH/models\\n'; "
    "if test -d /sdcard/Download/QLH/models; then printf 'model_root_exists=true\\n'; "
    "else printf 'model_root_exists=false\\n'; fi; "
    "printf 'gguf_count=%s\\n' \"$(find /sdcard/Download/QLH/models -maxdepth 1 -type f -name '*.gguf' 2>/dev/null | wc -l | tr -d ' ')\""
)


def _y700_report(
    *,
    transport: str,
    serial: str | None,
    completed: subprocess.CompletedProcess[str],
    failure_code: str,
) -> dict[str, Any]:
    """把一次探针结果整理成报告 —— ADB 与 SSH 两条路径共用的收尾逻辑。"""
    observations: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            observations[key.strip()] = value.strip()
    online = completed.returncode == 0 and "arm64-v8a" in {
        item.strip() for item in observations.get("abi", "").split(",")
    }
    try:
        gguf_count = int(observations.get("gguf_count", "0"))
    except ValueError:
        gguf_count = 0
    return {
        "status": "passed" if online else "failed",
        "target": "y700",
        "physical_nodes": True,
        "transport": transport,
        "serial": serial,
        "observations": observations,
        "model_gate": (
            "ready_for_arm64_model_smoke"
            if gguf_count > 0
            else "blocked_no_gguf"
        ),
        "error_code": "" if online else failure_code,
        "stderr_tail": completed.stderr[-500:] if not online else "",
    }


def _run_y700_ssh(target: str) -> dict[str, Any]:
    """走 Termux sshd 的 y700 探针（端口固定 8022，推荐路径）。"""
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
             "-o", "StrictHostKeyChecking=accept-new", target, Y700_PROBE_COMMAND],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "failed",
            "target": "y700",
            "physical_nodes": True,
            "transport": "termux_ssh",
            "serial": target,
            "error_code": "termux_ssh_unreachable",
            "model_gate": "blocked_no_gguf",
            "hint": ("Termux sshd 不可达：端口 8022 无监听通常意味着 Termux 进程被系统回收，"
                     "在设备上重新执行 `sshd` 即可（装 Termux:Boot 可开机自启）"),
        }
    return _y700_report(transport="termux_ssh", serial=target, completed=completed,
                        failure_code="ssh_probe_failed")


def _run_y700_probe(
    ssh_target: str | None,
    serial: str | None,
    *,
    adb: str | None = None,
    host: str = "100.99.211.13",
) -> dict[str, Any]:
    """y700 探针选路：**能用 Termux SSH 就用 SSH**（端口固定 8022），否则退回 ADB。

    `--y700-ssh` 显式指定时只用它；否则依次探测 `y700`、`y700-ip` 两个 ssh 别名
    （局域网别名优先，已在 `~/.ssh/config` 里），都不可达才走 ADB 无线调试。
    这样"动态 ADB 端口"只在 SSH 不可用时才需要面对。
    """
    if ssh_target:
        return _run_y700_ssh(ssh_target)
    if not serial:
        # 顺序很关键：**局域网别名排最前**。设备刚重启时 Tailscale 往往还在重建打洞
        # （实测 `tailscale status` 会短暂显示 `relay`），此时 Tailscale IP 超时、
        # 而局域网 IP 仍然通。所以按 lan -> ip -> ts.net 逐个探测。
        for candidate in ("y700-lan", "y700-ip", "y700"):
            if _ssh_works(candidate):
                return _run_y700_ssh(candidate)
    return _run_y700(serial, adb=adb, host=host)


def _run_y700(
    serial: str | None = None,
    *,
    adb: str | None = None,
    host: str = "100.99.211.13",
) -> dict[str, Any]:
    adb_path = _find_adb(adb)
    if not adb_path:
        return {
            "status": "failed",
            "target": "y700",
            "physical_nodes": True,
            "transport": "adb_wireless_debugging",
            "error_code": "adb_not_found",
            "model_gate": "blocked_no_gguf",
        }
    serial = serial or _discover_y700_serial(adb_path, host)
    if not serial:
        return {
            "status": "failed",
            "target": "y700",
            "physical_nodes": True,
            "transport": "adb_wireless_debugging",
            "error_code": "dynamic_adb_serial_required",
            "model_gate": "blocked_no_gguf",
            "hint": f"run adb connect {host}:<dynamic_port>, then pass --y700-serial",
        }
    completed = subprocess.run(
        [adb_path, "-s", serial, "shell", Y700_PROBE_COMMAND],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return _y700_report(transport="adb_wireless_debugging", serial=serial,
                        completed=completed, failure_code="adb_failed")


def _failed_result(target: str, error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "target": target,
        "physical_nodes": True,
        "error_type": type(error).__name__,
        "error": str(error)[:500],
    }


def _read_stream(src: Any) -> bytes:
    """统一读一块：socket 用 `recv`，stdio 用 `read`（`BufferedReader` 没有 `recv` —— 踩过）。"""
    if hasattr(src, "recv"):
        return src.recv(65536)
    return src.read1(65536) if hasattr(src, "read1") else src.read(65536)


def _write_stream(dst: Any, chunk: bytes) -> None:
    """统一写一块：socket 用 `sendall`，stdio 用 `write` + `flush`。"""
    if hasattr(dst, "sendall"):
        dst.sendall(chunk)
    else:
        dst.write(chunk)
        dst.flush()


def _weaknet_bridge(argv: list[str]) -> int:
    """★ **真实弱网（网络层）**注入：本脚本以 SSH `ProxyCommand` 代理的形式运行。

    在 stdio 与 `HOST:PORT` 之间双向转发，并按 `--delay-ms`（每块单向延迟）与 `--loss-pct`
    （按比例**整块丢弃**、交由 TCP 重传 —— 等效链路丢包）注入。

    与 `--inject-delay-ms` 的**区别**：后者只在**应用层**给控制帧加延迟；本模式作用在**整条 SSH
    通道**上（含其中的控制帧、也含 worker 源码下发），因此才算 §21 里 open 的「真实弱网」的一张证据。
    """
    parser = argparse.ArgumentParser(prog="ha_crosshost_physical_smoke --weaknet-bridge")
    parser.add_argument("--weaknet-bridge", required=True, metavar="HOST:PORT")
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--loss-pct", type=float, default=0.0)
    args = parser.parse_args(argv)
    host, _, port_text = args.weaknet_bridge.rpartition(":")
    if not host or not port_text.isdigit():
        print(f"weaknet bridge: bad target {args.weaknet_bridge!r}", file=sys.stderr)
        return 2
    delay_s = max(0.0, float(args.delay_ms) / 1000.0)
    loss = min(max(float(args.loss_pct), 0.0), 100.0) / 100.0
    remote = socket.create_connection((host, int(port_text)), timeout=15)

    def _read_chunk(src: Any) -> bytes:
        return _read_stream(src)

    def _write_chunk(dst: Any, chunk: bytes) -> None:
        _write_stream(dst, chunk)

    def _pump(dst: Any, src: Any) -> None:
        try:
            while True:
                chunk = _read_chunk(src)
                if not chunk:
                    break
                if loss and random.random() < loss:
                    continue                      # 丢这一块 ⇒ 上层 TCP 重传
                if delay_s:
                    time.sleep(delay_s)
                _write_chunk(dst, chunk)
        except (OSError, AttributeError, ValueError):
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except (OSError, AttributeError):
                pass

    stdin = getattr(sys.stdin, "buffer", sys.stdin)
    stdout = getattr(sys.stdout, "buffer", sys.stdout)
    upstream = threading.Thread(target=_pump, args=(remote, stdin), daemon=True)
    upstream.start()
    try:
        _pump(stdout, remote)
    finally:
        upstream.join(timeout=5)
        remote.close()
    return 0


def main() -> int:
    # ★ 真实弱网（网络层）：本脚本同时可作 SSH 的 `ProxyCommand` 代理运行 —— 见 `--weaknet-bridge`。
    if "--weaknet-bridge" in sys.argv[1:]:
        return _weaknet_bridge(sys.argv[1:])
    # GBK 控制台下 `--help` / 日志里的非 ASCII 字符会抛 UnicodeEncodeError（项目里踩过，
    # 见 `scripts/relay_health.py` 的同类兜底）⇒ 这里降级为 replace。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-target", default="surface@100.100.52.106")
    parser.add_argument("--surface-root", default=r"C:\Users\surface\Documents\LEDS_BJTU")
    parser.add_argument("--y700-serial", help="adb wireless serial, for example IP:<dynamic_port>")
    parser.add_argument("--y700-ssh", default=None,
                        help="Termux sshd target (fixed port 8022). Default: auto-detect the "
                             "`y700` / `y700-ip` ssh aliases; falls back to adb when unreachable")
    parser.add_argument("--y700-host", default="100.99.211.13")
    parser.add_argument("--adb", help="path to adb; defaults to PATH or the local Android SDK")
    parser.add_argument("--long-steps", type=int, default=1,
                        help="Surface control frames before restart (default: 1)")
    parser.add_argument("--inject-delay-ms", type=int, default=0,
                        help="弱网模拟：对每次控制帧往返注入该延迟（毫秒，单向 = RTT +2x）。"
                             "属应用层注入，证据里会标注注入值；不得当真实网络证据引用")
    parser.add_argument("--quorum-exchange", action="store_true",
                        help="跨机 quorum 证书交换：在 Surface 侧起真正的 voter（自带 SQLite 账本），"
                             "本机 collector 只联系 voter-a + 远端 voter-b（quorum_size=2），"
                             "证书两票来自两台主机，并核对对端账本已持久化")
    parser.add_argument("--quorum-remote-state",
                        default=r"C:/Users/surface/qlh-keephead/voter-b.sqlite",
                        help="远端 voter 账本路径（SQLite）")
    parser.add_argument("--weaknet-delay-ms", type=int, default=0,
                        help="真实弱网（网络层）：SSH 经本地 TCP 代理注入的单向延迟（毫秒）")
    parser.add_argument("--weaknet-loss-pct", type=float, default=0.0,
                        help="真实弱网（网络层）：SSH 代理注入的丢包率（%%）")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    long_steps = max(1, min(int(args.long_steps), 10_000))
    global _WEAKNET_DELAY_S
    _WEAKNET_DELAY_S = max(0.0, min(float(args.inject_delay_ms), 30_000.0) / 1000.0)
    proxy_command: str | None = None
    network_delay_ms = max(0, int(args.weaknet_delay_ms))
    network_loss_pct = min(max(float(args.weaknet_loss_pct), 0.0), 100.0)
    if network_delay_ms or network_loss_pct:
        # ★ 真实弱网（网络层）：SSH 经本脚本的 `--weaknet-bridge` 代理（延迟 + 丢包注入）
        surface_host = args.surface_target.rpartition("@")[2] or args.surface_target
        proxy_command = (
            f'"{sys.executable}" "{Path(__file__).resolve()}" --weaknet-bridge '
            f"{surface_host}:22 --delay-ms {network_delay_ms} "
            f"--loss-pct {network_loss_pct}")
    try:
        surface = _run_surface(args.surface_target, args.surface_root, long_steps=long_steps,
                               proxy_command=proxy_command)
    except Exception as exc:
        surface = _failed_result("surface", exc)
    try:
        y700 = _run_y700_probe(args.y700_ssh, args.y700_serial, adb=args.adb,
                               host=args.y700_host)
    except Exception as exc:
        y700 = _failed_result("y700", exc)
    quorum: dict[str, Any] | None = None
    if args.quorum_exchange:
        try:
            quorum = _run_quorum_exchange(args.surface_target, args.surface_root,
                                          remote_state=args.quorum_remote_state)
        except Exception as exc:
            quorum = _failed_result("quorum", exc)
    report = {
        "schema_version": "qlh.cluster.crosshost.physical.v1",
        "scenario": "physical_surface_y700_smoke",
        "surface": surface,
        "y700": y700,
        "long_steps": long_steps,
        "quorum": quorum,
        "weaknet_network": {
            "via_ssh_proxy": bool(proxy_command),
            "delay_ms_per_direction": network_delay_ms,
            "loss_pct": network_loss_pct,
            "scope": "whole_ssh_channel",
            "note": "网络层注入（本地 TCP 代理 + TCP 重传补丢包）；与 --inject-delay-ms（仅应用层控制帧）不同",
        },
        "production_availability_claim": False,
    }
    rendered = json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        partial = args.evidence.with_name(f".{args.evidence.name}.part")
        partial.write_text(rendered, encoding="utf-8")
        partial.replace(args.evidence)
    print(rendered, end="")
    return 0 if report["surface"]["status"] == "passed" and report["y700"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
